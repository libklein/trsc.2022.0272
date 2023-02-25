# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from itertools import product, chain
from operator import attrgetter
from typing import List, Union, Set, Dict, Tuple, Iterable
from docplex.mp.linear import Var
from docplex.mp.model import Model
from evrpscp import Charger, DiscreteTour, DiscretePeriod, DiscretizedInstance, PiecewiseLinearSegment, \
    FleetChargingSchedule
from funcy import flatten, cached_readonly, group_by
from graphviz import dot
from pprint import pprint as print

from .models import PWLBreakpoint, Vehicle
from column_generation.subproblem.network import *


class DynamicTourModel:

    def __init__(self, instance: DiscretizedInstance, **cli_args):

        self.instance = instance
        self.vehicles = instance.vehicles
        self.tours = instance.tours
        self.params = instance.parameters
        self.battery = self.params.battery
        self.chargers = instance.chargers
        self.periods = instance.periods
        self._cli_args = cli_args

        self.breakpoints_per_charger = {f: [PWLBreakpoint(time=time, soc=soc) for (time, soc) in f.breakpoints]
                                        for f in self.chargers}

        self.model = Model(f'Dynamic Tour EVRP-SCP')
        self.model.MAX_TIME = self.periods[-1].end * self.periods[-1].duration

        # Variables
        self.tau: Dict[Node, Var] = None
        self.beta: Dict[Node, Var] = None
        self.x: Dict[Arc, Var] = None
        self.phi_entry: Dict[Tuple[Node, PWLBreakpoint], Var] = None
        self.phi_exit: Dict[Tuple[Node, PWLBreakpoint], Var] = None
        self.wdf_entry: Dict[Tuple[Node, PWLBreakpoint], Var] = None
        self.wdf_exit: Dict[Tuple[Node, PWLBreakpoint], Var] = None

        self.vehicle_networks: Dict[Vehicle, TimeExpandedNetwork] = \
            {k: TimeExpandedNetwork(instance=instance, vehicle=k) for k in self.vehicles}

        self._initialize_variables()
        self._create_objective()

        for g in self.vehicle_networks.values():
            self._create_vehicle_constraints(g)

        self._create_capacity_constraits()

    @property
    def cplex(self) -> Model:
        return self.model

    def render_solution(self, vehicle: int = 0) -> dot.Digraph:
        raise NotImplementedError
        return self.vehicle_networks[vehicle].render()

    def display_solution(self, *args, **kwargs):
        self.render_solution(*args, **kwargs).render(view=True)


    def _initialize_variables(self):
        def _add_(var_dict, node, bps, *args, **kwargs):
            # TODO set LB and UB
            added_vars = []
            for bp in bps:
                new_var = self.model.continuous_var(bp, *args, **kwargs)
                var_dict[node, bps] = new_var
                added_vars.append(new_var)
            self.model.add_sos2(added_vars)
            return added_vars

        for k, network in self.vehicle_networks.items():
            for node in network.nodes:
                self.tau[node] = self.model.continuous_var(lb=0.0, ub=self.model.MAX_TIME, name=f"tau^{k}_{node}")
                self.beta[node] = self.model.continuous_var(lb=0.0, ub=self.model.MAX_TIME, name=f"beta^{k}_{node}")
                if node.charger is not None:
                    _add_(self.phi_entry, node, self.model._breakpoints_per_charger[node.charger])
                    _add_(self.phi_exit, node, self.model._breakpoints_per_charger[node.charger])
                    _add_(self.wdf_entry, node, self.model._wdf_breakpoints)
                    _add_(self.wdf_exit, node, self.model._wdf_breakpoints)
            for arc in network.arcs:
                self.x[arc] = self.model.binary_var(name=f"x^{k}_{arc}")

    def _create_objective(self):
        objective = self.model.linear_expr()

        for k, network in self.vehicle_networks.items():
            if not self._cli_args.get('ignore_degradation'):
                #

                all_tour_vertices: Iterable[Node] = chain(*map(attrgetter('tour_vertices'),
                                                                     self.vehicle_networks.values()))
                deg_cost = self.model.sum(
                    self.model.sum(rho * seg.slope for seg, rho in v.rho.items())
                    for v in all_tour_vertices)
                self.model.add_kpi(deg_cost, publish_name='Battery Degradation Cost')
                objective += deg_cost
            else:
                print('Ignoring battery degradation')

            if not self._cli_args.get('ignore_energy_price'):
                period_charge_cost = self.model.sum(
                    period.energyPrice * self.model.sum(self.gamma[p, station] for station in station_nodes)
                    for period, station_nodes in network.station_nodes
                )
                objective += period_charge_cost
                self.model.add_kpi(period_charge_cost, publish_name='Energy Price')
            else:
                print('Ignoring energy price')

            if not self._cli_args.get('ignore_route_costs'):
                raise NotImplementedError("No support for route costs has been implemented so far!")
            else:
                print('Ignoring route cost')

        self.model.minimize(objective)
        self.model.add_kpi(objective, publish_name='Schedule Cost')

    def _create_capacity_constraits(self):
        # Capacity constraints
        for p in self.periods:
            charges_in_period = defaultdict(list)
            for g in self.vehicle_networks.values():
                for v in g.charger_vertices:
                    if v.period is p:
                        charges_in_period[v.charger].append(sum(e.x for e in g.incoming_arcs(v)))
            self.model.add_constraints(sum(charges_in_period[f]) <= f.capacity for f in self.chargers if not f.isBaseCharger and f in charges_in_period)

    def _create_vehicle_constraints(self, g: VehicleGraph):
        model = self.model

        def add_constraints(constrs, *args, names: str, **kwargs):
            if isinstance(constrs, Iterable):
                return model.add_constraints(constrs, *args, names=f'Vehicle {g.vehicle}: {names}', **kwargs)
            else:
                return model.add_constraint(constrs, *args, ctname=f'Vehicle {g.vehicle}: {names}', **kwargs)

        def arc_sum(arcs: Iterable[Edge]):
            return model.sum(arc.x for arc in arcs)

        add_constraints(arc_sum(g.outgoing_arcs(g.source)) == 1,
                        names='source has outgoing arc')

        add_constraints((arc_sum(g.incoming_arcs(v)) == 1 for v in chain(g.tour_vertices, (g.sink,))),
                        names='tours and sink have incoming arc')

        add_constraints((arc_sum(g.incoming_arcs(v)) == arc_sum(g.outgoing_arcs(v))
                         for v in chain(g.tour_vertices, g.charger_vertices)),
                        names='flow preservation')

        add_constraints((e.origin.tau + e.origin.duration <= e.target.tau + (1 - e.x) * len(self.periods)
                         for e in g.edges if e.origin not in g.tour_vertices),
                        names='arrival time propagation chargers')

        # Add +1 for tour vertices as origin.tau + duration gives us the arrival period. We're still blocked in that
        # period though
        add_constraints((e.origin.tau + e.origin.duration + 1 <= e.target.tau + (1 - e.x) * len(self.periods)
                         for e in g.edges if e.origin in g.tour_vertices),
                        names='arrival time propagation tours')

        add_constraints((e.origin.beta + e.origin.delta_charge + e.origin.gamma >= e.target.beta - (1 - e.x) * self.battery.maximumCharge
                        for v in g.charger_vertices for e in g.outgoing_arcs(v)),
                        names='charge propagation (charger)')

        add_constraints((e.origin.beta + e.origin.delta_charge >= e.target.beta - (1 - e.x) * self.battery.maximumCharge
                         for v in chain(g.tour_vertices, (g.source,)) for e in g.outgoing_arcs(v)),
                        names='charge propagation (charger)')

        add_constraints((v.earliest_begin_pid <= v.tau
                         for v in g.vertices),
                        names='lower time bound')

        add_constraints((v.latest_begin_pid >= v.tau
                         for v in g.vertices),
                        names='upper time bound')

        add_constraints((self.battery.minimumCharge <= v.beta
                         for v in g.vertices),
                        names='lower soc bound')

        add_constraints((self.battery.maximumCharge >= v.beta
                         for v in g.vertices),
                        names='upper soc bound')

        # add_constraints((e.origin.mu >= e.target.mu - (1 - e.x) * 2 * self.battery.maximumCharge
        #                 for i in g.charger_vertices for e in g.outgoing_arcs(i)),
        #                 names='proagate mu')
        #
        # # Sets and initialized mu
        # add_constraints((e.target.mu <= e.target.beta + (1 - e.x) * self.battery.maximumCharge
        #                  for i in chain(g.tour_vertices, (g.source,)) for e in g.outgoing_arcs(i)),
        #                 names='set mu')

        add_constraints((e.target.mu >= e.origin.mu + e.origin.gamma - (1 - e.x) * 2 * self.battery.maximumCharge
                        for i in g.charger_vertices for e in g.outgoing_arcs(i)),
                        names='proagate mu')

        # Sets and initialized mu
        add_constraints((e.target.mu <= (1 - e.x) * self.battery.maximumCharge
                         for i in chain(g.tour_vertices, (g.source,)) for e in g.outgoing_arcs(i)),
                        names='set mu')

        add_constraints(g.source.mu == 0, names='initialize mu')

        add_constraints(g.source.tau == 0, names='initialize tau')
        add_constraints(g.source.beta == self.battery.initialCharge, names='initialize beta')


        add_constraints((model.sum(v.lambda_entry[b] * b.soc for b in self.breakpoints_per_charger[v.charger])
                         == v.beta
                         for v in g.charger_vertices), names='bind entry soc')

        add_constraints((model.sum((v.lambda_exit[b] - v.lambda_entry[b]) * b.soc
                                        for b in self.breakpoints_per_charger[v.charger])
                         == v.gamma
                         for v in g.charger_vertices), names='bind exit soc')

        add_constraints((model.sum(v.lambda_exit[b] * b.time for b in self.breakpoints_per_charger[v.charger])
                         - self.model.sum(v.lambda_entry[b] * b.time for b in self.breakpoints_per_charger[v.charger])
                         <= self.params.period_length * arc_sum(g.incoming_arcs(v)) for v in g.charger_vertices),
                        names='limit charge by duration')

        # Convexity
        add_constraints(
            (model.sum(v.lambda_entry[b] for b in self.breakpoints_per_charger[v.charger]) == 1
             for v in g.charger_vertices), names='convexity soc entry')

        add_constraints(
            (model.sum(v.lambda_exit[b] for b in self.breakpoints_per_charger[v.charger]) == 1
             for v in g.charger_vertices), names='convexity soc exit')

        # Construct SOS2 Sets
        for v in g.charger_vertices:
            if len(self.breakpoints_per_charger[v.charger]) > 2:
                model.add_sos2([v.lambda_entry[b] for b in self.breakpoints_per_charger[v.charger]],
                               name="lambda_entry_sos2")
                model.add_sos2([v.lambda_exit[b] for b in self.breakpoints_per_charger[v.charger]],
                               name="lambda_exit_sos2")

        ##################################################################################################

        add_constraints((w.upperBound >= v.beta - v.mu - (1 - u) * self.battery.maximumCharge
                         for v in g.tour_vertices for w, u in v.u.items()),
                        names='only non-filled deg segments can be selected')

        add_constraints((v.rho[w] <= (w.upperBound - w.lowerBound) * v.u[w]
                         for v, w in product(g.tour_vertices, self.battery)),
                        names='limit rho by segment length')

        add_constraints((v.rho[w] <= w.upperBound - (v.beta - v.mu) + (1 - v.u[w]) * self.battery.maximumCharge
                         for v, w in product(g.tour_vertices, self.battery)),
                        names='limit rho by residual charge')

        add_constraints((model.sum(v.rho.values()) == v.mu for v in g.tour_vertices),
                              names='set rho required')

    def create_fleet_charging_schedule(self) -> FleetChargingSchedule:
        fleet_schedule = FleetChargingSchedule([g.create_charging_schedule(self.battery)
                                                for g in self.vehicle_networks.values()], isFeasible=True)
        fleet_schedule.calculate_cost(self.battery)
        return fleet_schedule

    def validate_solution(self):
        for g in self.vehicle_networks.values():
            g.validate_solution()

        visit_counter: Dict[Tuple[DiscretePeriod, Charger], int] = defaultdict(int)
        for k, g in self.vehicle_networks.items():
            for v in g.visited_vertices:
                if isinstance(v, ChargerVertex):
                    visit_counter[v.period, v.charger] += 1
        for (p, f), visit_count in visit_counter.items():
            assert visit_count <= f.capacity or f.isBaseCharger, f'Error: Charger {f} is used by {visit_count} vehicles in period {p}'
