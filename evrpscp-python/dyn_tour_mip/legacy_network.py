# coding=utf-8

from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from itertools import product, chain
from operator import attrgetter
from typing import List, Union, Set, Dict, Tuple, Iterable, Optional
from docplex.mp.linear import Var
from docplex.mp.model import Model
from evrpscp import Charger, DiscreteTour, DiscretePeriod, DiscretizedInstance, PiecewiseLinearSegment, Battery, \
    is_close, VehicleChargingSchedule, ChargingOperation, VehicleDeparture, Operation
from funcy import flatten, cached_readonly, group_by, keep, nth
from graphviz import dot
from .models import Vehicle, PWLBreakpoint


@dataclass
class Vertex:
    model: InitVar[Model]
    min_charge: InitVar[float]
    max_charge: InitVar[float]
    energy_cost: float
    delta_charge: float
    duration: int
    earliest_begin: DiscretePeriod
    latest_begin: DiscretePeriod
    fix_cost: float
    vehicle: Vehicle

    beta: Var = field(init=False, compare=False)
    gamma: Var = field(init=False, compare=False)
    tau: Var = field(init=False, compare=False)
    mu: Var = field(init=False, compare=False)

    def __post_init__(self, model: Model, min_charge: float, max_charge: float):
        self.beta = model.continuous_var(name=f'v{self.vehicle}-beta_{str(self)}', lb=min_charge, ub=max_charge)
        self.gamma = model.continuous_var(name=f'v{self.vehicle}-gamma_{str(self)}', lb=min_charge, ub=max_charge)
        self.tau = model.continuous_var(name=f'v{self.vehicle}-tau_{str(self)}', lb=self.earliest_begin_pid, ub=self.latest_begin_pid)
        self.mu = model.continuous_var(name=f'v{self.vehicle}-mu_{str(self)}', lb=min_charge, ub=max_charge)

    def format_solution(self) -> str:
        return f'{str(self)}[beta: {self.beta.solution_value:.2f}, gamma: {self.gamma.solution_value:.2f},' \
               f' tau: {int(self.tau.solution_value)}, mu: {self.mu.solution_value:.2f}]'

    def __repr__(self):
        return str(self)

    def validate_solution(self):
        pass

    @property
    def has_operation(self) -> bool:
        return True

    @property
    def entry_soc(self):
        return self.beta.solution_value

    @property
    def exit_soc(self):
        return self.entry_soc + self.gamma.solution_value + self.delta_charge

    @property
    def entry_time(self):
        return int(self.tau.solution_value)

    @property
    def exit_time(self):
        return self.entry_time + self.duration

    @property
    def replenished_charge(self):
        return self.delta_charge

    @property
    def earliest_begin_pid(self):
        return self.earliest_begin.begin // self.earliest_begin.duration

    @property
    def latest_begin_pid(self):
        return self.latest_begin.begin // self.latest_begin.duration

    def to_operation(self) -> Optional[Operation]:
        return None

    def __hash__(self):
        return id(self)

class SourceVertex(Vertex):
    def __init__(self, model: Model, vehicle: Vehicle, initial_charge: float, initial_cost: float, min_charge: float, max_charge: float):
        super().__init__(model=model, vehicle=vehicle, energy_cost=0.0, delta_charge=initial_charge, duration=-1,
                         earliest_begin=None, latest_begin=None, fix_cost=initial_cost, min_charge=min_charge, max_charge=max_charge)

    @property
    def has_operation(self) -> bool:
        return False

    @property
    def earliest_begin_pid(self):
        return 0

    @property
    def latest_begin_pid(self):
        return 0

    def __str__(self):
        return f'Source'

class SinkVertex(Vertex):
    def __init__(self, model: Model, latest_period: DiscretePeriod, vehicle: Vehicle, min_charge: float, max_charge: float):
        self._pid = (latest_period.begin // latest_period.duration) + 1
        super().__init__(model=model, energy_cost=0.0, delta_charge=0.0, duration=0,
                         earliest_begin=None, latest_begin=None, fix_cost=0.0, vehicle=vehicle, min_charge=min_charge, max_charge=max_charge)

    @property
    def has_operation(self) -> bool:
        return False

    @property
    def earliest_begin_pid(self):
        return self._pid

    @property
    def latest_begin_pid(self):
        return self._pid

    def __str__(self):
        return f'Sink'

class TourVertex(Vertex):
    tour: DiscreteTour

    rho: Dict[PiecewiseLinearSegment, Var] = field(init=False, compare=False)
    u: Dict[PiecewiseLinearSegment, Var] = field(init=False, compare=False)

    def __init__(self, model: Model, wdf_breakpoints: Iterable[PiecewiseLinearSegment], tour: DiscreteTour, vehicle: Vehicle, min_charge: float, max_charge: float):
        self.tour = tour

        self.rho = {b: model.continuous_var(name=f'v{vehicle}-rho_{b}_{str(self)}') for b in wdf_breakpoints}
        self.u = {b: model.binary_var(name=f'v{vehicle}-u_{b}_{str(self)}') for b in wdf_breakpoints}

        super().__init__(model=model, energy_cost=0.0, delta_charge=-tour.consumption, duration=tour.duration,
                         earliest_begin=tour.earliest_departure, latest_begin=tour.latest_departure, fix_cost=tour.cost, vehicle=vehicle, min_charge=min_charge, max_charge=max_charge)

    def validate_solution(self):
        assert self.exit_time == self.entry_time + self.tour.duration
        assert self.tour.earliest_departure_time <= self.entry_time * self.tour.earliest_departure.duration
        assert self.entry_time * self.tour.earliest_departure.duration <= self.tour.latest_departure_time

    def to_operation(self) -> VehicleDeparture:
        return VehicleDeparture(begin=nth(self.entry_time - self.earliest_begin_pid, iter(self.earliest_begin)),
                                end=nth(self.exit_time - self.earliest_begin_pid, iter(self.earliest_begin)),
                                entrySoC=self.entry_soc, exitSoC=self.exit_soc, isFeasible=True, tour=self.tour)

    @property
    def active_segments(self) -> Dict[PiecewiseLinearSegment, Var]:
        return {seg: var.solution_value for seg, var in self.u.items() if var.to_bool()}

    @property
    def charge_on_segments(self) -> Dict[PiecewiseLinearSegment, Var]:
        return {seg: var.solution_value for seg, var in self.rho.items() if var.solution_value > 1e-6}

    @property
    def cost(self):
        return self.tour.cost

    def validate_solution(self):
        super().validate_solution()
        assert set(self.active_segments.keys()).issuperset(set(self.charge_on_segments.keys())), \
            f'Error: Invalid vertex {str(self)}. Active segments: {self.active_segments}' \
            f' do not match charge on segments {self.charge_on_segments}' \
            f' i.e. there are rho with pos. value where u is not 1'
        assert is_close(sum(self.charge_on_segments.values()), self.mu.solution_value)

    def format_solution(self) -> str:
        deg_values = ', '.join(f'({u.solution_value:.2f}: {rho.solution_value:.2f})' for (u, rho) in zip(self.u.values(), self.rho.values()))
        return f'{super().format_solution()}[{deg_values}]'

    def __str__(self):
        return str(self.tour)

class ChargerVertex(Vertex):
    period: DiscretePeriod
    charger: Charger

    lambda_entry: Dict[PWLBreakpoint, Var] = field(init=False, compare=False)
    lambda_exit: Dict[PWLBreakpoint, Var] = field(init=False, compare=False)

    def __init__(self, model: Model, breakpoints: Iterable[PWLBreakpoint], period: DiscretePeriod, charger: Charger, vehicle: Vehicle, min_charge: float, max_charge: float):
        assert isinstance(period, DiscretePeriod) and isinstance(charger, Charger)
        self.period = period
        self.charger = charger

        self.lambda_entry = {b: model.continuous_var(name=f'v{vehicle}-lambda-entry_{b}_{str(self)}') for b in breakpoints}
        self.lambda_exit = {b: model.continuous_var(name=f'v{vehicle}-lambda-exit_{b}_{str(self)}') for b in breakpoints}

        super().__init__(model=model, energy_cost=period.energyPrice, delta_charge=0.0, duration=1, earliest_begin=period,
                         latest_begin=period, fix_cost=0.0, vehicle=vehicle, min_charge=min_charge, max_charge=max_charge)

    def to_operation(self) -> ChargingOperation:
        return ChargingOperation.FromChargerPeriod(self.charger, self.period, self.entry_soc, self.replenished_charge)

    @property
    def replenished_charge(self):
        return self.gamma.solution_value

    @property
    def has_operation(self) -> bool:
        return self.charges

    @property
    def charges(self):
        return self.replenished_charge >= 1e-6

    def validate_solution(self):
        super().validate_solution()
        assert round(self.exit_soc - self.entry_soc, 2) <= round(self.charger.charge_for(self.entry_soc, self.period.duration) - self.entry_soc, 2), \
            f'Error: Charger {str(self)} replenishes {self.exit_soc - self.entry_soc} (gamma: {self.gamma.solution_value:.2f}) but {self.charger} allows only {self.charger.charge_for(self.entry_soc, self.period.duration) - self.entry_soc} in {self.period.duration} minutes'

    @property
    def cost(self):
        return self.replenished_charge * self.period.energyPrice

    def format_solution(self) -> str:
        conv_values = f'Entry: {"-".join(str(b) for b, var in self.lambda_entry.items() if var.solution_value > 0.99)}, ' \
                      f'Exit: {"-".join(str(b) for b, var in self.lambda_exit.items() if var.solution_value > 0.99)}'
        return f'{super().format_solution()}[{conv_values}]'

    def __str__(self):
        return f'{self.period} ({self.charger})'

@dataclass
class Edge:
    model: InitVar[Model]
    origin: Vertex
    target: Vertex
    vehicle: int

    x: Var = field(init=False, compare=False)

    def __post_init__(self, model: Model):
        assert isinstance(self.origin, Vertex)
        assert isinstance(self.target, Vertex)
        self.x = model.binary_var(name=f'x_{str(self)}')

    def validate_solution(self):
        if not self.active:
            return False
        assert is_close(self.origin.entry_soc + self.origin.replenished_charge, self.target.entry_soc), \
            f'Error: Entry soc + replenished charge at {self.origin} does not equal the entry soc of {self.target}: ' \
            f'{self.origin.entry_soc} + {self.origin.replenished_charge} = ' \
            f'{self.origin.entry_soc + self.origin.replenished_charge} != {self.target.entry_soc}'
        assert self.origin.entry_time + self.origin.duration <= self.target.entry_time, f'Error: {self.origin} entry time >= {self.target} entry time: {self.origin.entry_time} <= {self.target.entry_time}'
        if isinstance(self.origin, ChargerVertex):
            assert round(self.target.mu.solution_value, 2) >= round(self.origin.mu.solution_value + self.origin.gamma.solution_value, 2)
        elif isinstance(self.origin, TourVertex):
            assert round(self.origin.entry_soc, 2) > round(self.target.entry_soc, 2)
            assert self.target.mu.solution_value == 0.0

    @property
    def active(self) -> bool:
        return self.x.to_bool(precision=1e-4)

    def __str__(self):
        return f'({self.origin}, {self.target}, {self.vehicle})'

    def __iter__(self) -> Iterable[Vertex]:
        return iter((self.origin, self.target))

    def __hash__(self):
        return hash((self.origin, self.target, self.vehicle))



@dataclass
class VehicleGraph:
    vehicle: Vehicle
    source: SourceVertex
    sink: SinkVertex
    tour_vertices: Set[TourVertex]
    charger_vertices: Set[ChargerVertex]
    edges: Set[Edge]

    def __post_init__(self):
        for v in chain(self.tour_vertices, self.charger_vertices):
            for e in self.incoming_arcs(v):
                assert e.target is v
            for e in self.outgoing_arcs(v):
                assert e.origin is v
        for e in self.edges:
            assert e.origin is not e.target

    def incoming_arcs(self, vertex: Vertex, active=False) -> Iterable[Edge]:
        if not active:
            return (e for e in self.edges if e.target is vertex)
        return (e for e in self.edges if e.target is vertex if e.active)

    def outgoing_arcs(self, vertex: Vertex, active=False) -> Iterable[Edge]:
        if not active:
            return (e for e in self.edges if e.origin is vertex)
        return (e for e in self.edges if e.origin is vertex if e.active)

    def render(self, solution=False) -> dot.Digraph:
        g = dot.Digraph(f'Network of vehicle {self.vehicle}')
        for edge in self.edges:
            g.edge(str(edge.origin), str(edge.target))
        return g

    def display(self, *args, **kwargs):
        self.render(*args, **kwargs).render(view=True)

    def __str__(self):
        return f'Network of vehicle {self.vehicle} with {len(self.tour_vertices)} tours,' \
               f' {len(self.charger_vertices)} chargers and {len(self.edges)} edges'

    def __repr__(self):
        return str(self)

    @cached_readonly
    def chargers_by_period(self) -> Dict[DiscretePeriod, Set[ChargerVertex]]:
        chargers = defaultdict(set)
        for v in self.charger_vertices:
            chargers[v.period].add(v.charger)
        return chargers

    @cached_readonly
    def visited_vertices(self) -> Iterable[Vertex]:
        visited_vertices = set()
        next_vertex = self.source
        while next_vertex is not self.sink:
            outgoing_arcs = list(self.outgoing_arcs(next_vertex, active=True))
            assert len(outgoing_arcs) == 1
            visited_vertices.add(next_vertex)
            next_vertex = outgoing_arcs[0].target
        visited_vertices.add(next_vertex)
        return visited_vertices


    def validate_solution(self):
        # Validate each vertex
        for v in self.vertices:
            v.validate_solution()
        # Validate each arc
        for e in self.edges:
            e.validate_solution()
        # Validate connectivity and propagation
        # All tours should have been visisted
        for v in self.tour_vertices:
            assert v in self.visited_vertices

    @property
    def vertices(self) -> Iterable[Vertex]:
        return chain((self.source, self.sink), self.charger_vertices, self.tour_vertices)

    def create_charging_schedule(self, battery: Battery) -> VehicleChargingSchedule:
        operations = sorted((v.to_operation() for v in self.visited_vertices if v.has_operation), key=lambda operation: operation.begin)
        cost = 0.0
        schedule = VehicleChargingSchedule(cost=cost, isFeasible=True, operations=operations, vehicleID=self.vehicle)
        schedule.calculate_cost(battery=battery)
        return schedule

def build_vehicle_graph(vehicle: Vehicle, model: Model,
                        periods: Iterable[DiscretePeriod],
                        tours: Iterable[DiscreteTour],
                        battery: Battery,
                        breakpoints_per_charger,
                        charging_opportunities: Iterable[Tuple[DiscretePeriod, Charger]]) -> VehicleGraph:

    source = SourceVertex(model=model, initial_charge=battery.initialCharge, initial_cost=0.0, vehicle=vehicle, min_charge=battery.minimumCharge, max_charge=battery.maximumCharge)
    sink = SinkVertex(model=model, latest_period=periods[-1], vehicle=vehicle, min_charge=battery.minimumCharge, max_charge=battery.maximumCharge)
    tour_vertices = {TourVertex(model=model, wdf_breakpoints=battery, tour=pi, vehicle=vehicle, min_charge=battery.minimumCharge, max_charge=battery.maximumCharge)
                     for pi in tours}
    charger_vertices = {ChargerVertex(model=model, breakpoints=breakpoints_per_charger[f],
                                      period=p, charger=f, vehicle=vehicle, min_charge=battery.minimumCharge, max_charge=battery.maximumCharge) for (p, f) in charging_opportunities}

    # Generate edges
    edges: Set[Edge] = set()

    def add_edge(origin: Vertex, target: Vertex):
        edges.add(Edge(model=model, origin=origin, target=target, vehicle=vehicle))

    chargers_by_period = group_by(attrgetter('period'), charger_vertices)

    # Link charger nodes
    prev_nodes = [source]
    for p in periods:
        if len(chargers_in_period := chargers_by_period[p]) == 0:
            continue

        # Create Edges
        for cur_node in chargers_in_period:
            # To Tours
            for tour_vertex in tour_vertices:
                # Link only those chargers that are within the TW or right before it's begin
                if p >=tour_vertex.tour.earliest_departure.pred and p < tour_vertex.tour.latest_departure:
                    add_edge(cur_node, tour_vertex)
            # To previous chargers / source
            for prev_node in prev_nodes:
                if cur_node is not prev_node:
                    add_edge(prev_node, cur_node)
        prev_nodes = chargers_in_period

    # Create Tour edges
    for tour_vertex in tour_vertices:
        tour = tour_vertex.tour
        # Link Source to tour (only if no charger in between)
        if tour.earliest_departure == periods[0]:
            add_edge(source, tour_vertex)
        # Link to sink
        add_edge(tour_vertex, sink)
        # Link to other tours
        for other_tour_vertex in tour_vertices:
            other_tour = other_tour_vertex.tour
            if other_tour.id == tour_vertex.tour.id:
                continue
            # Create link only if py_tour's arrival time windows intersects other_py_tour's departure time window
            # i.e. it is possible to travel other_py_tour directly after arriving from py_tour (i.e. without visiting any additional chargers)
            if other_tour.earliest_departure.pred <= tour.earliest_arrival < other_tour.latest_departure:
                add_edge(tour_vertex, other_tour_vertex)
        # Link to chargers
        for charger_vertex in charger_vertices:
            # Link T1 to p only if p lies in the arrival time window
        # Use > here as earliest_arrival is blocked
            if tour.latest_arrival.succ is not None and tour.earliest_arrival < charger_vertex.period <= tour.latest_arrival.succ:
                add_edge(tour_vertex, charger_vertex)

    return VehicleGraph(source=source, sink=sink, tour_vertices=tour_vertices,
                        charger_vertices=charger_vertices, edges=edges, vehicle=vehicle)


def render_graph(g: VehicleGraph, solution=False) -> dot.Digraph:
    dot_graph = dot.Digraph(f'Vehicle {g.vehicle}')
    if not solution:
        for e in g.edges:
            dot_graph.edge(str(e.origin), str(e.target))
        return dot_graph

    # Render solution
    active_edges = set(filter(lambda e: e.active, g.edges))
    active_vertices: Set[Vertex] = {e.origin for e in active_edges} | {e.target for e in active_edges}
    active_vertices.add(g.sink)
    # draw vertices
    for v in active_vertices:
        dot_graph.node(name=str(v), label=v.format_solution())
        assert any(e.origin is v or e.target is v for e in active_edges)
    # draw edges
    for e in active_edges:
        dot_graph.edge(tail_name=str(e.origin), head_name=str(e.target))
        assert e.origin in active_vertices
        assert e.target in active_vertices
    return dot_graph


def display_graph(g: VehicleGraph, solution=False):
    viz = render_graph(g, solution=solution)
    viz.render(view=True)