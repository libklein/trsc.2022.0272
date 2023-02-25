# coding=utf-8
import math
from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Generator
from copy import copy, deepcopy
from itertools import product
from tempfile import NamedTemporaryFile
from bidict import bidict
from contexttimer import Timer

from evrpscp.models import DiscretePeriod
from column_generation import DiscretizedInstance, Column
from column_generation.util import setup_default_logger
from column_generation.constraints import Constraint, BlockChargerConstraint
from column_generation.subproblem.network import *
from funcy import rest

from .util import *

import evspnl

logger = setup_default_logger(__name__)

_arc_type_mapper = {
    ArcType.Charging: evspnl.ArcType.Charging,
    ArcType.Source: evspnl.ArcType.Source,
    ArcType.Idle: evspnl.ArcType.Idle,
    ArcType.Service: evspnl.ArcType.Service,
}


class CPPSubproblem:
    def __init__(self, instance: DiscretizedInstance, vehicle: int, **cli_args):
        if any((cli_args.get('ignore_energy_price'), cli_args.get('ignore_degradation'),
                cli_args.get('ignore_route_costs'))):
            raise NotImplementedError

        # Instance data
        self._instance = instance
        self.vehicle = vehicle
        self.tours = instance.tours[vehicle]
        self.params = instance.parameters
        self.battery = self.params.battery
        self.chargers = instance.chargers
        self.periods = instance.periods

        self._cli_args = cli_args

        # Last generated column
        self.generated_column: Optional[Column] = None

        # Create mappings and network
        self._cpp_charger_mapping = bidict({charger: to_cpp_charger(charger) for charger in self.chargers})
        self._cpp_tour_mapping = bidict({tour: to_cpp_tour(tour, id=i) for i, tour in enumerate(self.tours)})
        self._cpp_wdf = to_cpp_wdf(self.battery.wearCostDensityFunction)

        # Network
        self.network = TimeExpandedNetwork(instance=self._instance, vehicle=self.vehicle)
        self.__cpp_network = None

        self._apply_optimizations()
        self._initialize_network()

    def _initialize_network(self):
        # Compute lower bounds etc.
        compute_required_service(network=self.network, operations=self.tours)
        compute_lower_bound(network=self.network, wdf=self.battery.wearCostDensityFunction)

    @property
    def _cpp_network(self):
        if self.__cpp_network is None:
            self.__cpp_network, self._cpp_vertex_mapping, self._cpp_arc_mapping = self._create_network_mappings()
        return self.__cpp_network

    def _create_network_mappings(self) -> Tuple[
        evspnl.TimeExpandedNetwork, bidict[Node, evspnl.VertexID], bidict[Arc, evspnl.ArcID]]:
        cpp_network = evspnl.TimeExpandedNetwork()
        cpp_vertex_mapping = bidict()

        for vertex in self.network.nodes:
            next_vid = None
            if vertex.node_type == NodeType.Source:
                next_vid = cpp_network.source
            elif vertex.node_type == NodeType.Sink:
                next_vid = cpp_network.sink
            elif vertex.node_type == NodeType.Idle:
                next_vid = cpp_network.add_vertex(evspnl.Vertex(
                    evspnl.VertexType.Garage,
                    int(vertex.period.begin // evspnl.PERIOD_LENGTH),
                    vertex.period.energyPrice
                ))
            elif vertex.node_type == NodeType.Station:
                next_vid = cpp_network.add_vertex(evspnl.Vertex(int(vertex.period.begin // evspnl.PERIOD_LENGTH),
                                                                vertex.period.energyPrice,
                                                                self._cpp_charger_mapping[vertex.charger],
                                                                evspnl.create_period_profile(
                                                                    self._cpp_wdf,
                                                                    vertex.period.energyPrice
                                                                )
                                                                ))
            else:
                raise NotImplementedError

            cpp_vertex = cpp_network.get_vertex(next_vid)
            # Set required service
            if not hasattr(vertex, 'serviced_operations'):
                raise ValueError(f'Error! Vertex {vertex} doesn\'t have service_operations set! Forgot to initialize lb?')
            unserviced_ops = set(self.tours)
            for op in getattr(vertex, 'serviced_operations'):
                cpp_op = self._cpp_tour_mapping[op]
                cpp_vertex.set_service_required(cpp_op.id, True)
                unserviced_ops.remove(op)
            for op in unserviced_ops:
                cpp_vertex.add_potentially_unserviced_operation(self._cpp_tour_mapping[op])
                cpp_vertex.set_soc_cost_lower_bound(to_cpp_pwl_function(getattr(vertex, 'min_charging_cost')))
            #print(vertex, unserviced_ops)


            cpp_vertex_mapping[vertex] = next_vid

        cpp_arc_mapping = bidict()

        def create_arc(arc: Arc, *args, **kwargs) -> evspnl.ArcID:
            return cpp_network.add_arc(cpp_vertex_mapping[arc.origin], cpp_vertex_mapping[arc.target], evspnl.Arc(*args, **kwargs))

        for arc in self.network.arcs:
            arc_id = None
            if arc.arc_type in (ArcType.Source, ArcType.Charging, ArcType.Idle):
                arc_id = create_arc(arc, _arc_type_mapper[arc.arc_type], arc.cost, -arc.consumption, 1)
            elif arc.arc_type == ArcType.Service:
                operation = self._cpp_tour_mapping[arc.tour]
                arc_id = create_arc(arc, arc.cost, -arc.consumption, operation.duration, operation)
            else:
                raise NotImplementedError
            cpp_arc_mapping[arc] = arc_id

        return cpp_network, cpp_vertex_mapping, cpp_arc_mapping

    @property
    def _network_dump_enabled(self) -> bool:
        return self._cli_args.get('dump_networks', False)

    def __deepcopy__(self, memodict={}) -> 'CPPSubproblem':
        # TODO Implement
        clone = copy(self)
        # Bindings are fine - we can reuse the ones the original node uses
        # We want to recreate the network
        clone.network = deepcopy(self.network)
        # Invalidate CPP network binding
        clone.__cpp_network = None
        # Finally, reset the generated columns
        clone.generated_column = None
        return clone

    def _apply_optimizations(self):
        optimizations = [
            partial(remove_covered_periods, self.network),
            partial(remove_superfluous_sink_periods, self.network, self.tours),
            partial(remove_dead_ends, self.network),
            partial(remove_idle_chains, self.network)
        ]
        removal_stats = apply_optimizations(*optimizations)
        if sum(removal_stats.values()) > 0:
            self.__cpp_network = None
        logger.info(f'Optimizer stats: {removal_stats}, total removal: {sum(removal_stats.values())}')
        return removal_stats

    def add_constraint(self, constraint: Constraint) -> bool:
        if not constraint.applies_to_vehicle(self.vehicle):
            return False

        if isinstance(constraint, BlockChargerConstraint):
            if constraint.force_usage:
                raise NotImplementedError
                # force_station_usage works, but optimizer may remove that station from the network. This case
                # needs to be checked which is not implemented yet.
                self._force_station_usage(constraint.period, constraint.charger)
            else:
                self._forbid_station_usage(constraint.period, constraint.charger)

            # Invalidate cpp network
            self.__cpp_network = None

            # Reoptimize
            self._apply_optimizations()
        else:
            raise NotImplementedError

        return True

    def _force_station_usage(self, p: DiscretePeriod, f: Charger):
        # Remove all other vertices (station and idle) from the period.
        for forbidden_station_vertex in self.network.nodes_by_period[p]:
            if forbidden_station_vertex.charger is not f:
                self.network.remove_vertex(forbidden_station_vertex)
                assert forbidden_station_vertex not in self.network.nodes

        # Additionally, remove all service arcs which correspond to schedules where the vehicle is en-route in p.
        for departure_period, departure_node in self.network.idle_nodes.items():
            if departure_period > p:
                break
            for service_arc in (arc for arc in departure_node.outgoing_arcs if arc.arc_type == ServiceArc):
                if service_arc.target.period > p:
                    self.network.remove_arc(arc=service_arc)
                    assert service_arc not in self.network.arcs

    def _forbid_station_usage(self, p: DiscretePeriod, f: Charger):
        # Remove the corresponding vertex from the network. This will also remove all charging arcs.
        station_vertex = self.network._get_station(p, f)
        self.network.remove_vertex(station_vertex)
        assert station_vertex not in self.network.nodes

    def can_improve(self, coverage_delta: float, capacity_delta: Dict[Tuple[DiscretePeriod, Charger], float]):
        if self.generated_column is None or coverage_delta is None or coverage_delta != 0.0:
            return True
        # Positive delta -> duals increased
        for (p, f) in self.generated_column.charging_operations:
            # If the dual of a used charger decreases improvement is possible
            if capacity_delta.get((p, f), 0.0) < 0.0:
                return True

        for (p, f), delta in capacity_delta.items():
            # If the dual of an unused charger increases then it may be worthwhile to use that charger instead
            # unless we can't use the charger as we're en-route
            if not self.generated_column.charger_usage[p, f] and delta > 0.0 \
                    and self.network.has_node(p, f):
                return True
        return False

    def generate_column(self, coverage_dual: float, capacity_dual: Dict[Tuple[DiscretePeriod, Charger], float],
                        obj_threshold=0.0) \
            -> Optional[Column]:
        with Timer(factor=1000, fmt="{:.3f} ms") as network_timer:
            # TODO Cache column and check if any charger gets less expensive (charger may be covered by operation)
            self._update_fix_costs(coverage_dual=coverage_dual, capacity_dual=capacity_dual)

        if self._cpp_network is None:
            self._create_network_mappings()

        # Measure C API call time
        with Timer(factor=1000, fmt="{:.3f} ms") as c_ext_timer:

            if self._network_dump_enabled:
                next_col_id = Column._next_id
                import pathlib, shutil
                dumps_dirname = pathlib.Path(f'network_img_v{self.vehicle}_col{next_col_id}')
                if dumps_dirname.exists():
                    shutil.rmtree(str(dumps_dirname))
                dumps_dirname.mkdir()
                self._dump_cpp_network(view=False, filename=dumps_dirname / 'network_img_cpp')
                self.network.render().render(view=False, filename=dumps_dirname / 'network_img_py')

            solver = evspnl.Solver(self._cpp_network)
            min_cost_label = None

            try:
                min_cost_label = solver.solve(obj_threshold)
            except Exception as e:
                raise e


        if self._network_dump_enabled:
            self._dump_network_to_file(
                expected_solution_value=min_cost_label.minimum_cost if min_cost_label is not None else 'INFEASIBLE',
                coverage_dual=coverage_dual, obj_threshold=obj_threshold)

        if not min_cost_label:
            logger.info(
                f'Vehicle [{self.vehicle}]: Could not find feasible/improving schedule [C Call: {c_ext_timer} ms, Network: {network_timer} ms]! '
                f'Threshold: {obj_threshold}, Offset: {coverage_dual}')
            solver.reset()
            self.generated_column = None
            return None

        logger.info(
            f'Vehicle [{self.vehicle}]: Found best schedule with cost {min_cost_label.minimum_cost} [C Call: {c_ext_timer} ms, Network: {network_timer} ms]!')

        with Timer(factor=1000) as col_timer:
            column = self._construct_column(min_cost_label)
        solver.reset()

        if self.generated_column is not None and column is not None and column == self.generated_column:
            logger.debug(f'Vehicle [{self.vehicle}] generated duplicate column - returning none')
            return None

        logger.debug(f'Vehicle [{self.vehicle}] constructed col in {col_timer} ms: {repr(column)}')

        self.generated_column = column

        return column

    def _update_fix_costs(self, coverage_dual: float, capacity_dual: Dict[Tuple[DiscretePeriod, Charger], float]):
        # Always set negative costs. See subproblem objective function.
        self.network.set_duals(capacity_dual=capacity_dual, coverage_dual=coverage_dual)
        self.__cpp_network = None

    def _dump_cpp_network(self, view=True, filename=None):
        dot_viz = repr(self._cpp_network)
        import graphviz
        if filename is None:
            from tempfile import mktemp
            filename = mktemp()
        graphviz.Source(dot_viz, filename=filename).render(view=view, filename=filename)

    def _dump_network_to_file(self, file_obj=None, expected_solution_value=None, coverage_dual=0.0, capacity_dual={},
                              obj_threshold=None):
        if file_obj is None:
            file_obj = NamedTemporaryFile(mode='w+b', delete=False, dir=self._cli_args.get('output_dir', '.'),
                                          prefix='network-', suffix='.pickle')
        import pickle
        pickle.dump({
            "expected_solution_value": expected_solution_value,
            "coverage_dual": coverage_dual,
            "capacity_dual": capacity_dual,
            "obj_threshold": obj_threshold,
            "subproblem": [self._instance, self.vehicle, self._cli_args]
        }, file_obj)

    def _iterate_label_path(self, tail_label) -> Generator[Tuple[evspnl.Label, Arc], None, None]:
        labels, arcs = [], []
        next_label = tail_label
        while not next_label.is_root:
            labels.append(next_label)
            arcs.append(self._cpp_arc_mapping.inverse[next_label.arc])
            next_label = next_label.predecessor
        for label, arc in zip(reversed(labels), reversed(arcs)):
            yield label, arc


    def _construct_column(self, min_cost_label) -> Column:
        assert evspnl.approx_eq(min_cost_label.minimum_soc, 0.0)
        labels_and_arcs = list(self._iterate_label_path(min_cost_label))

        # Default values
        energy_charged: Dict[DiscretePeriod, float] = {p: 0.0 for p in self.periods}
        degradation_cost: Dict[DiscretePeriod, float] = {p: 0.0 for p in self.periods}
        charger_usage: Dict[Tuple[DiscretePeriod, Charger], bool] = {(p, f): False for p, f in
                                                                     product(self.periods, self.chargers)}
        # Set tour departures
        tour_departures: Dict[DiscreteTour, DiscretePeriod] = {arc.tour: arc.origin.period for _, arc in labels_and_arcs
                                                               if arc.arc_type == ArcType.Service}

        labels = [label for label, _ in labels_and_arcs]
        traversed_arcs = [arc for _, arc in labels_and_arcs]
        # Problem:
        #   1) We don't know exactly how much we charge at each station, only the soc and cost we reach the sink with.
        #   -> At station replacements and the sink, we have a certain target soc that we need to reach.
        #   -> Charge replenished at intermediate and replaced stations are not known.
        #
        # As we know the target soc and that we charge as much as possible on intermediate stations we can however
        # calculate how much we charge by calculating a function that gives us the soc at the end of a path, given
        # that we charge \delta q at some uncommitted station.
        # This results in the following algorithm:
        # 1) Identify subpaths which start at the last point up to which the charge has been calculated, and end at the
        #    next commitment (station replacement or reaching the sink)
        # 2) For each of these in order:
        #       1) Calculate the exit-entry-soc profile.
        #       2) Get the inverse at target_soc to calculate the amount recharged at the replaced/tracked station
        # 3) Iterate over the path, charging as much as possible at intermediate stations

        charging_paths = identify_subpaths(labels_and_arcs)
        assert sum(len(subpath) for subpath, _ in charging_paths) == len(traversed_arcs)

        logger.debug("Charging paths:")
        for charging_path, target_soc in charging_paths:
            logger.debug(f"\tTarget SoC {target_soc}: ")
            for label, arc in charging_path:
                logger.debug(f'\t\t{arc}')

        delta_soc_at_label: Dict[evspnl.Label, float] = defaultdict(float)

        charger_of = lambda x: self._cpp_charger_mapping[x.origin.charger]
        prev_soc = self.battery.initialCharge
        for charging_path, target_soc in charging_paths:
            station_label, station_arc = charging_path[0]
            path = [
                charger_of(arc).phi if arc.arc_type == ArcType.Charging else arc.consumption for label, arc in rest(charging_path)
            ]

            if station_arc.arc_type != ArcType.Source:
                delta_soc_at_label[station_label] = calculate_charge_at_tracked_station(
                    station=charger_of(station_arc).phi, path=path, target_soc=target_soc, max_charge_time=evspnl.PERIOD_LENGTH,
                    max_soc=self.battery.maximumCharge, initial_soc=prev_soc)
            else:
                delta_soc_at_label[station_label] = -1.0 * station_arc.consumption
            prev_soc = target_soc

        # Iterate over the path. With the now known values for replaced stations we can infer how much is charged at
        # each intermediate station
        current_soc = 0.0
        remaining_charge = sum(pi.consumption for pi in self.tours) - self.battery.initialCharge
        for label, arc in labels_and_arcs:
            if arc.arc_type == ArcType.Charging:
                if label.intermediate_charge:
                    assert label not in delta_soc_at_label
                    delta_soc_at_label[label] = min(charger_of(arc).phi.getCharge(current_soc, evspnl.PERIOD_LENGTH),
                                                    current_soc + remaining_charge, self.battery.maximumCharge) \
                                                - current_soc
                else:
                    # A label visiting a station without fix-cost will always dominate visiting the garage node, hence
                    # replaced stations don't nessesarily charge
                    assert delta_soc_at_label[label] >= 0.0

                assert not math.isnan(delta_soc_at_label[label]), f'Error: Delta soc at label {label}, arc: {arc} is NaN!'

                remaining_charge -= delta_soc_at_label[label]
            else:
                delta_soc_at_label[label] = -1.0 * arc.consumption
            # Avoid rounding errors
            assert not evspnl.certainly_lt(remaining_charge, 0.0, 1e-4), f'Error: Negative charge remaining! {remaining_charge}'
            current_soc += delta_soc_at_label[label]

            assert not evspnl.certainly_lt(current_soc, self.battery.minimumCharge, 1e-4), \
                f'Current SoC {current_soc:.2f} is significantly less than minimum charge {self.battery.minimumCharge}'
            assert not evspnl.certainly_gt(current_soc, self.battery.maximumCharge, 1e-4), \
                f'Current SoC {current_soc:.2f} is significantly more than maximum charge {self.battery.maximumCharge}'
            # Mitigate (minor) rounding errors. Conditions above ensure that the error is indeed minor.
            current_soc = min(max(0.0, current_soc), self.battery.maximumCharge)

        assert evspnl.approx_eq(current_soc, 0.0, 1e-4)

        # Now, finally, update energy_charged and charger_usage accordingly
        for label, arc in labels_and_arcs:
            p, f = arc.origin.period, arc.origin.charger
            if arc.arc_type != ArcType.Charging:
                continue
            delta_soc = energy_charged[p] = delta_soc_at_label[label]
            if delta_soc > 0.0 or arc.cost > 0.0:
                charger_usage[p, f] = True

        # Set degradation costs
        cur_soc = 0.0
        for label, arc in zip(labels, traversed_arcs):
            if arc.arc_type == ArcType.Charging:
                delta_soc = energy_charged[arc.origin.period]
                degradation_cost[arc.origin.period] = self._cpp_wdf.getWearCost(cur_soc, cur_soc + delta_soc)
                assert 0.0 <= degradation_cost[arc.origin.period] <= self._cpp_wdf.maximum_cost, \
                    f'Invalid deg cost in period {arc.origin.period}: {degradation_cost[arc.origin.period]} ' \
                    f'but cpp wdf has max cost {self._cpp_wdf.maximum_cost}, ' \
                    f'py wdf: {self.battery.wearCostDensityFunction.image_upper_bound}.' \
                    f'delta_soc is {delta_soc}, cur_soc is {cur_soc}'
            else:
                delta_soc = -arc.consumption
            cur_soc += delta_soc
            assert not math.isnan(cur_soc), f'Error: Found current soc of NaN! {cur_soc}. At label {label}, arc {arc},' \
                                            f' delta soc {delta_soc}'
            assert not evspnl.certainly_lt(cur_soc, 0.0)
            # Fix potential rounding errors
            cur_soc = max(0.0, cur_soc)

        col = Column(energy_charged=energy_charged,
                     degradation_cost=degradation_cost,
                     charger_usage=charger_usage,
                     tour_departures=tour_departures,
                     vehicle=self.vehicle,
                     objective=min_cost_label.minimum_cost,
                     cost=min_cost_label.minimum_cost - sum(x.cost for x in traversed_arcs))

        assert evspnl.approx_eq(col.cost, min_cost_label.minimum_cost - sum(x.cost for x in traversed_arcs))

        return col