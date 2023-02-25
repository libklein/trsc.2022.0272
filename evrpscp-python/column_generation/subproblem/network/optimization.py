# coding=utf-8
import sys
from collections import deque

from .time_expanded_network import *
from evrpscp.models.piecewise_linear_function import PiecewiseLinearFunction, PiecewiseLinearSegment

def apply_optimizations(*optimizations, **kwargs):
    removed_nodes = {
        strategy.func.__name__: strategy(**kwargs) for strategy in optimizations
    }
    return removed_nodes


def remove_dead_ends(network: TimeExpandedNetwork):
    """
    Recursively removes all vertices without incomming or outgoing arcs.
    """
    # TODO Recurse (via event?)
    num_removals = 0
    while True:
        dead_ends = list(v for v in network.nodes if (len(v.outgoing_arcs) == 0 or len(v.incoming_arcs) == 0)
                         and not (v.is_sink_node or v.is_source_node))
        if len(dead_ends) == 0:
            break
        for v in dead_ends:
            num_removals += 1
            network.remove_vertex(v)
    return num_removals


def remove_superfluous_sink_periods(network: TimeExpandedNetwork, operations: List[DiscreteTour]):
    """
    Remove vertices occuring after the arrival of the last tour. Also add sink arcs to each idle node after the earliest
    arrival
    """
    last_tour = max(operations, key=lambda op: op.latest_arrival_time)

    # Remove stations in latest arrival period
    nodes_to_remove = list(network.station_nodes[last_tour.latest_arrival])
    for p in drop(1, iter(last_tour.latest_arrival)):
        # Remove all nodes of periods after the last arrival
        nodes_to_remove.extend(network.nodes_by_period[p])

    # Remove nodes
    for node in nodes_to_remove:
        network.remove_vertex(node)

    first_finish_opportunity = max(operations, key=lambda op: op.earliest_arrival_time)
    for p in first_finish_opportunity.earliest_arrival:
        if idle_node := network.idle_nodes.get(p):
            network.add_shortcut(idle_node)

    return len(nodes_to_remove)


def remove_covered_periods(network: TimeExpandedNetwork):
    """
    Removes all vertices assigned to periods during which the vehicle is guaranteed to be en-route
    """
    service_arcs_by_operation: Dict[DiscreteTour, List[ServiceArc]] = defaultdict(list)
    for arc in filter(lambda x: x.arc_type == ArcType.Service, network.arcs):
        service_arcs_by_operation[arc.tour].append(arc)

    nodes_to_remove = []
    for period, nodes in network.nodes_by_period.items():
        # If all arcs of any operation cover this period, the vehicle will always be en-route during period
        for operation, arcs in service_arcs_by_operation.items():

            def arc_covers_period(arc: ServiceArc, period: DiscretePeriod):
                return arc.origin.period.begin <= period.begin and period.end <= arc.target.period.end

            if all(arc_covers_period(arc, period) for arc in arcs):
                # Idle nodes (for departure/arrival) must be kept if arc.origin.period == period or arc.target.period == period
                if operation.latest_departure is period or operation.earliest_arrival is period:
                    nodes_to_remove.extend(network.station_nodes[period])
                    break
                else:
                    nodes_to_remove.extend(nodes)
                    break

    if len(nodes_to_remove) == 0:
        return 0

    for node in nodes_to_remove:
        network.remove_vertex(node)
    return len(nodes_to_remove)


def _can_reach_station(vertex: Node) -> Optional[Node]:
    """
    Returns the first station reachable from vertex or None if no station exists
    """
    visited_vertices = set()
    vertex_stack = deque((vertex,))
    while len(vertex_stack) > 0:
        next_vertex = vertex_stack.popleft()
        if next_vertex.is_station_node:
            return next_vertex
        if next_vertex in visited_vertices:
            continue
        visited_vertices.add(next_vertex)
        vertex_stack.extend(arc.target for arc in next_vertex.outgoing_arcs)
    return None


def remove_idle_chains(network: TimeExpandedNetwork):
    nodes_to_remove = []
    for p, idle_node in reversed(network.idle_nodes.items()):
        if _can_reach_station(idle_node) or len(idle_node.incoming_arcs) > 1 or any(arc.arc_type != ArcType.Idle for arc in idle_node.incoming_arcs):
            break
        nodes_to_remove.append(idle_node)

    for node in nodes_to_remove:
        # May be unconnected
        predecessor = node.incoming_arcs[0].origin if len(node.incoming_arcs) > 0 else None
        network.remove_vertex(node)
        if predecessor:
            network.add_shortcut(predecessor)
    return len(nodes_to_remove)


def compute_required_service(network: TimeExpandedNetwork, operations: List[DiscreteTour]):
    """
    Sets for each vertex the tours that must have been completed at this point for the solution to remain feasible.
    """

    def set_serviced_ops(node: Node, ops: List[DiscreteTour]):
        setattr(node, 'serviced_operations', list(ops))

    unfinished_ops = deque(sorted(operations, key=lambda op: op.latest_departure))
    finished_ops = []
    set_serviced_ops(network.source, finished_ops)
    for p, nodes in network.nodes_by_period.items():
        while len(unfinished_ops) > 0:
            if unfinished_ops[0].latest_departure < p:
                finished_ops.append(unfinished_ops.popleft())
            else:
                break
        for node in nodes:
            set_serviced_ops(node, finished_ops)

    assert len(unfinished_ops) == 0
    set_serviced_ops(network.sink, finished_ops)


def compute_lower_bound(network: TimeExpandedNetwork, wdf: PiecewiseLinearFunction):
    total_consumption = sum(tour.consumption for tour in set(arc.tour for arc in network.arcs if arc.arc_type == ArcType.Service))

    def create_and_set_profile(node: Node, slope: float):
        setattr(node, 'min_charging_cost', PiecewiseLinearFunction.CreatePWLFromSlopeAndUB([
            (total_consumption, slope)
        ]))

    create_and_set_profile(network.sink, 10_000.0)

    min_deg_cost = wdf.segments[1].slope
    min_energy_cost = sys.float_info.max
    for p, nodes in reversed(network.nodes_by_period.items()):
        min_energy_cost = min(min_energy_cost, p.energyPrice + min_deg_cost)
        # TODO Improve
        for node in nodes:
            create_and_set_profile(node, min_energy_cost)

    create_and_set_profile(network.source, min_energy_cost)
