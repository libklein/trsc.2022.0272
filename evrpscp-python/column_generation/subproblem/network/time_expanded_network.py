# coding=utf-8
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from typing import List, Iterator, Dict, Optional, Any, ClassVar, Callable, Set, Tuple, TypeVar, Generic
from itertools import product

from column_generation import Column
from column_generation.debug_mode import extensive_checks
from evrpscp.models import ChargingOperation, VehicleDeparture
from evrpscp import DiscretizedInstance, Charger, DiscreteTour, Vehicle, DiscretePeriod, Battery
from .arcs import *
from .nodes import *

from funcy import *
from graphviz import dot

def DefaultArcFactory(type: ArcType, *args, **kwargs) -> Arc:
    if type == ArcType.Source:
        return SourceArc(arc_type=type, *args, **kwargs)
    elif type == ArcType.Charging:
        return ChargingArc(arc_type=type, *args, **kwargs)
    elif type == ArcType.Idle:
        return IdleArc(arc_type=type, *args, **kwargs)
    elif type == ArcType.Service:
        return ServiceArc(arc_type=type, *args, **kwargs)
    else:
        raise ValueError

def DefaultNodeFactory(type: NodeType, *args, **kwargs) -> Node:
    if type == NodeType.Source:
        return SourceNode(*args, **kwargs)
    elif type == NodeType.Sink:
        return SinkNode(*args, **kwargs)
    elif type == NodeType.Idle:
        return IdleNode(*args, **kwargs)
    elif type == NodeType.Station:
        return StationNode(*args, **kwargs)
    else:
        raise NotImplementedError

NodeTypes = TypeVar('NodeTypes', SourceNode, SinkNode, IdleNode, StationNode)
ArcTypes = TypeVar('ArcTypes', SourceArc, ChargingArc, IdleArc, ServiceArc)

class TimeExpandedNetwork:
    DummyCharger = None

    def __init__(self, instance: DiscretizedInstance, vehicle: Vehicle, arc_factory=DefaultArcFactory, node_factory=DefaultNodeFactory):
        self.arc_factory = arc_factory
        self.node_factory = node_factory
        self.source = node_factory(NodeType.Source, period=None, charger=None)
        self.sink = node_factory(NodeType.Sink, period=None, charger=None)
        self.station_nodes: Dict[DiscretePeriod, List[StationNode]] = {p: [] for p in instance.periods}
        self.idle_nodes: Dict[DiscretePeriod, IdleNode] = {}
        self.nodes_by_period: Dict[Optional[DiscretePeriod], List[Node]] = {p: [] for p in instance.periods}

        self._build_network(instance.periods, instance.chargers, instance.tours[vehicle], instance.parameters.battery, vehicle)

    def __deepcopy__(self, memodict={}):
        network = copy(self)

        memodict.update((id(entity), copy(entity)) for entity in chain(self.nodes, self.arcs))

        # Update source, sink, station_nodes, idle_nodes and nodes_by_period
        network.source = memodict[id(self.source)]
        network.sink = memodict[id(self.sink)]
        network.station_nodes = {p: [memodict[id(f)] for f in station_list] for p, station_list in self.station_nodes.items()}
        network.idle_nodes = {p: memodict[id(idle_node)] for p, idle_node in self.idle_nodes.items()}
        network.nodes_by_period = {p: [memodict[id(node)] for node in node_list] for p, node_list in self.nodes_by_period.items()}

        for entity in chain(network.nodes, network.arcs):
            entity.update_references(memodict)

        return network

    def _get_station(self, period: DiscretePeriod, charger: Charger) -> Node:
        if charger is None:
            return self.idle_nodes[period]
        for node in self.station_nodes[period]:
            if node.charger is charger:
                return node
        raise ValueError(f'Charger {charger} does not exist in period {period}!')

    def has_node(self, period: DiscretePeriod, charger: Optional[Charger] = None):
        if charger is None:
            return period in self.idle_nodes
        else:
            return any(station_node.charger is charger for station_node in self.station_nodes[period])

    def _get_service_arc(self, from_node: Node, tour: DiscreteTour) -> Arc:
        for arc in from_node.outgoing_arcs:
            if arc.arc_type == ArcType.Service and arc.tour is tour:
                return arc

        raise ValueError(f'Node {from_node} has not service arc corresponding to {tour}!')

    def _get_charging_arc(self, charger: Charger, target: Node) -> Arc:
        for arc in target.incoming_arcs:
            if charger is None and arc.arc_type == ArcType.Idle:
                return arc
            if arc.arc_type == ArcType.Charging and arc.charger is charger:
                return arc

        raise ValueError(f'Node {target} is not connected with any node representing {charger}!')

    @property
    def nodes(self) -> Iterator[Node]:
        return chain(flatten(self.nodes_by_period.values()), (self.source, self.sink))

    @property
    def arcs(self) -> Iterator[Arc]:
        return flatten(node.outgoing_arcs for node in self.nodes)

    def arcs_by_type(self, arc_type: Generic[ArcTypes]) -> Iterator[ArcTypes]:
        return filter(lambda x: isinstance(x, arc_type), self.arcs)

    def nodes_by_type(self, node_type: Generic[NodeTypes]) -> Iterator[NodeTypes]:
        return filter(lambda x: isinstance(x, node_type), self.nodes)

    def set_duals(self, capacity_dual: Dict[Tuple[DiscretePeriod, Charger], float], coverage_dual: float):
        for source_arc in self.source.outgoing_arcs:
            source_arc.cost = -coverage_dual

        for p, stations in self.station_nodes.items():
            for f in stations:
                dual = capacity_dual.get((p, f.charger), None)
                if dual is not None:
                    for outgoing_arc in f.outgoing_arcs:
                        outgoing_arc.cost = -dual

    def clear_solution(self):
        """
        Clears all solution information from the network
        """
        raise NotImplementedError

    def set_solution(self, column: Column, reset=True):
        """
        Sets the network state according to the column object passed, optionally resetting the network
        """
        if reset:
            self.clear_solution()

        period_operations = {
            p: None for p in column.periods
        }
        initial_soc = 0
        # Map operations to the periods in which they occur
        for operation in column.iter_operations(initial_charge=initial_soc):
            period_operations[operation.begin] = operation

        prev_node = self.source
        prev_p, prev_operation = None, None
        p = first(column.periods)
        # Iterate over all periods.
        while p is not None:
            # The network is in a proper state up to period prev_p. prev_operation determines
            # the arc from prev_p to whatever period is next active (tours may skip periods)
            # operation determines the target node (in p)
            operation = period_operations[p]

            # Determine the target
            if operation is None:
                # Vehicle will idle in period p. Hence, we take the arc that leads to an idle node in p.
                target = self.idle_nodes[p]
            elif isinstance(operation, ChargingOperation):
                # Vehicle will charge in p.
                target = self._get_station(period=p, charger=operation.charger)
            elif isinstance(operation, VehicleDeparture):
                # Vehicle departs will depart in period p. Hence, we need to be on a idle node in p.
                target = self.idle_nodes[p]
            else:
                raise ValueError(f'Unsupported operation type in period {p}: {operation}')

            arc = prev_node.get_arc_to(target)
            arc.is_active = True
            # Determine what we did
            if prev_operation is None:
                prev_node.charge = prev_node.deg_cost = 0
            elif isinstance(prev_operation, ChargingOperation):
                # Charge
                prev_node.charge = prev_operation.soc_delta
                prev_node.deg_cost = prev_operation.degradation_cost
            elif isinstance(prev_operation, VehicleDeparture):
                # Take a service arc
                # Here, we need to advance to the first period before arrival, i.e., operation.end
                # before arrival is important here as we advance p at the end of the loop, so it will point
                # to a valid period in the next operation
                p = prev_operation.end
            else:
                # This actually cannot happen, previous raise would catch the error first
                raise ValueError(f'Unexpected operation type')

            if prev_operation is not None:
                target.entry_soc = prev_operation.exitSoC
            else:
                target.entry_soc = prev_node.exit_soc
            prev_node, prev_p, prev_operation = arc.target, p, operation
            p = p.succ

        entry_soc = 0
        prev_node = self.source
        for operation, prev_operation in with_prev(column.iter_operations(initial_charge=entry_soc)):
            # Operations indicate charging or departure events, but not idle time
            entry_soc = operation.entrySoC
            #arc = prev_node.get_arc_to(operation.begin)
            if isinstance(operation, VehicleDeparture):
                # Mark arc as taken
                arc = self._get_service_arc(from_node=prev_node, tour=operation.tour)
                pass
            elif isinstance(operation, ChargingOperation):
                arc = self._get_charging_arc(charger=prev_operation.charger, target=self._get_station(period=operation.begin, charger=operation.charger))
                pass
            else:
                raise NotImplementedError
            prev_node = arc.origin


    def create_column(self, chargers: List[Charger], vehicle: int, objective: float) -> Column:
        """
        Transforms the solution of the current network into a column.
        """
        energy_charged: Dict[DiscretePeriod, float] = {}
        degradation_cost: Dict[DiscretePeriod, float] = {}
        charger_usage: Dict[Tuple[DiscretePeriod, Charger], bool] = {}
        tour_departures: Dict[DiscreteTour, DiscretePeriod] = {}

        for period, nodes in self.nodes_by_period.items():
            active_arc = first(arc for node in nodes for arc in node.outgoing_arcs if arc.is_active)
            # En-route during this period
            if not active_arc:
                energy_charged[period] = 0.0
                degradation_cost[period] = 0.0
                for f in chargers:
                    charger_usage[period, f] = False
            else:
                energy_charged[period] = active_arc.origin.charge
                degradation_cost[period] = active_arc.origin.deg_cost
                for f in chargers:
                    charger_usage[period, f] = (f is active_arc.origin.charger and active_arc.arc_type == ArcType.Charging)
                if active_arc.arc_type == ArcType.Service:
                    tour_departures[active_arc.tour] = period

        return Column(energy_charged=energy_charged, degradation_cost=degradation_cost,charger_usage=charger_usage,
                      tour_departures=tour_departures, vehicle=vehicle, objective=objective)

    def _check_solution(self):
        for period, nodes in self.nodes_by_period.items():
            assert sum(1 for node in nodes for arc in node.outgoing_arcs if arc.is_active) <= 1
            for node in nodes:
                node.check_solution()
                for arc in node.outgoing_arcs:
                    arc.check_solution()

    def _add_charging_opportunities(self, periods: List[DiscretePeriod], chargers: List[Charger]):
        for p, f in product(periods, chargers):
            self.station_nodes[p].append(self.node_factory(type=NodeType.Station, period=p, charger=f))

    def _add_idle_nodes(self, periods: List[DiscretePeriod]):
        for p in periods:
            self.idle_nodes[p] = self.node_factory(type=NodeType.Idle, period=p, charger=TimeExpandedNetwork.DummyCharger)

    def _create_service_arcs(self, node: Node, tours: List[DiscreteTour]):
        last_period = last(iter(node.period))
        assert last_period is not None
        for pi in tours:
            if node.period > pi.latest_departure or node.period < pi.earliest_departure:
                continue
            arrival_period = nth(pi.duration, node.period)
            if arrival_period is None:
                continue
            arrival_idle_node = self.idle_nodes[arrival_period]
            arc = node.add_outgoing_arc(self.arc_factory(type=ArcType.Service, origin=node, target=arrival_idle_node, cost=0.0, consumption=pi.consumption, tour=pi))
            arrival_idle_node.add_incomming_arc(arc)
            #target_period: DiscretePeriod = nth(1, iter(arrival_period))
            #arrival_nodes = self.nodes_by_period[target_period] if target_period is not None else [self.sink]
            #for arrival_node in arrival_nodes:
            #    arc = node.add_outgoing_arc(self.arc_factory(type=ArcType.Service, origin=node, target=arrival_node, cost=0.0, consumption=pi.consumption, tour=pi))
            #    arrival_node.add_incomming_arc(arc)

    def add_shortcut(self, origin: Node, arc_type=ArcType.Idle, **kwargs) -> 'Arc':
        assert origin.is_idle_node
        default_args = dict(cost=0.0, consumption=0.0)
        default_args.update(**kwargs)
        arc = origin.add_outgoing_arc(self.arc_factory(type=arc_type, origin=origin, target=self.sink, **default_args))
        self.sink.add_incomming_arc(arc)
        return arc

    def _connect_node(self, node: Node, tours: List[DiscreteTour], create_arc: Callable):
        next_period: Optional[DiscretePeriod] = nth(1, iter(node.period))
        if not next_period:
            arc = node.add_outgoing_arc(create_arc(node, self.sink))
            self.sink.add_incomming_arc(arc)
            return
        # Connect to other stations
        for successor in self.station_nodes[nth(1, iter(node.period))]:
            arc = node.add_outgoing_arc(create_arc(node, successor))
            successor.add_incomming_arc(arc)
        # Connect to idle node
        target = self.idle_nodes[nth(1, iter(node.period))]
        arc = node.add_outgoing_arc(create_arc(node, target))
        target.add_incomming_arc(arc)

    def _connect_station_node(self, station: StationNode, tours: List[DiscreteTour]):
        self._connect_node(station, tours, lambda origin, target: self.arc_factory(type=ArcType.Charging, origin=origin, target=target, cost=0.0, consumption=0.0))

    def _connect_idle_node(self, node: IdleNode, tours: List[DiscreteTour]):
        self._connect_node(node, tours, lambda origin, target: self.arc_factory(type=ArcType.Idle, origin=origin, target=target, cost=0.0, consumption=0.0))
        # Create service arcs
        self._create_service_arcs(node, tours)

    def _build_network(self, periods: List[DiscretePeriod], chargers: List[Charger], tours: List[DiscreteTour], battery: Battery, vehicle: Vehicle):
        # Add vertices
        self._add_charging_opportunities(periods=periods, chargers=chargers)
        self._add_idle_nodes(periods=periods)
        for p in self.nodes_by_period:
            self.nodes_by_period[p].extend(self.station_nodes[p])
            if (node := self.idle_nodes.get(p, None)) is not None:
                self.nodes_by_period[p].append(node)

        # Add arcs
        for node in flatten(self.nodes_by_period.values()):
            if node.is_idle_node:
                self._connect_idle_node(node, tours)
            elif node.is_station_node:
                self._connect_station_node(node, tours)
            else:
                raise NotImplementedError(f'Unknown node type: {node}')

        # Connect source
        for successor in self.nodes_by_period[periods[0]]:
            arc = self.source.add_outgoing_arc(self.arc_factory(type=ArcType.Source, origin=self.source, target=successor, consumption=-battery.initialCharge, cost=0.0))
            successor.add_incomming_arc(arc)

    def remove_vertex(self, vertex: Node):
        if vertex in (self.source, self.sink):
            raise ValueError("Cannot remove source/sink nodes from time expanded network!")

        # Remove vertex from dicts
        if vertex.is_station_node:
            self.station_nodes[vertex.period].remove(vertex)
        elif vertex.is_idle_node:
            del self.idle_nodes[vertex.period]
        self.nodes_by_period[vertex.period].remove(vertex)

        # Remove vertex
        vertex._remove()

    def remove_arc(self, arc):
        arc._remove()

    def render(self, *args, **kwargs) -> dot.Digraph:
        g = dot.Digraph(f'Time expanded network')
        for arc in self.arcs:
            arc.render(g, *args, **kwargs)
        return g

    def display(self, *args, **kwargs):
        self.render(*args, **kwargs).render(view=True)
