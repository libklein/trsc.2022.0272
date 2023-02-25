# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass
from math import isinf
from typing import List, Dict, Iterable, Tuple, Any, Optional

from column_generation import Column, INTEGRALITY_TOLERANCE, INTEGRALITY_PRECISION
from evrpscp.models import DiscretePeriod, DiscreteTour, Battery, Charger
from .types import Vehicle

@dataclass
class SoCPeriod:
    entry_soc: float
    exit_soc: float

    @property
    def delta_soc(self) -> float:
        return self.exit_soc - self.entry_soc


def _get_period_socs(periods: List[DiscretePeriod], charge_per_period: Dict[DiscretePeriod, float],
                     consumption_per_period: Dict[DiscretePeriod, float],
                     initial_soc: float) -> List[SoCPeriod]:
    soc_evolution = []
    soc = initial_soc
    for p in periods:
        soc_evolution.append(SoCPeriod(soc, soc + charge_per_period.get(p, 0.0) - consumption_per_period.get(p, 0.0)))
        soc = soc_evolution[-1].exit_soc

    return soc_evolution


def get_soc_evolution(charge_per_period: Dict[DiscretePeriod, float], tours: List[DiscreteTour], initial_charge: float) \
        -> Dict[DiscretePeriod, SoCPeriod]:
    if len(charge_per_period) == 0:
        raise ValueError
    periods = list(charge_per_period.keys())
    periods.sort(key=lambda x: x.begin)
    soc_evolution = _get_period_socs(periods, charge_per_period, {pi.departure: pi.consumption for pi in tours},
                                     initial_charge)
    return {p: soc for p, soc in zip(periods, soc_evolution)}


def iter_periods(begin: DiscretePeriod, end: DiscretePeriod, include_end=True) -> Iterable[DiscretePeriod]:
    assert begin <= end
    next = begin
    while next is not end:
        yield next
        next = next.succ
    if include_end:
        yield next


def get_reachable_nodes(network: 'FRVCPNetwork') -> Dict[DiscretePeriod, List['Node']]:
    from column_generation.subproblem.frvcp_cpp.processors import NodeProcessor, ProcessingPass
    class GatherProcessor(NodeProcessor):
        direction = ProcessingPass.BACKWARD
        def __init__(self):
            self.nodes = defaultdict(list)

        def init(self, root):
            self.nodes[root.period].append(root)

        def __call__(self, current_node):
            if current_node.period is not None and current_node not in self.nodes[current_node.period]:
                self.nodes[current_node.period].append(current_node)

    # Make sure insertion order corresponds to periods
    nodes = sorted(network.process(GatherProcessor()).nodes.items(), key=lambda period_nodes: period_nodes[0].begin)
    return {k: v for k, v in nodes}


def distribute_amount_charged(charger: Charger, entry_soc: float, delta_soc: float, periods: Iterable[DiscretePeriod]) \
        -> Iterable[Tuple[DiscretePeriod, float]]:
    """
    Distribute the amount charged across the periods passed
    """
    #logger.info(f'Requires dur: {charger.duration(entry_soc, entry_soc+delta_soc)}')
    assert delta_soc >= 0.0
    next_segment_id = charger.inverseChargingFunction.get_segment_id(entry_soc)
    next_segment = charger.chargingFunction[next_segment_id]
    assert next_segment.imageLowerBound <= entry_soc < next_segment.imageUpperBound
    for period in periods:
        #period_amount = min(next_segment.imageUpperBound - entry_soc,
        #                    next_segment.imageUpperBound - next_segment.imageLowerBound,
        #                    next_segment.slope * period.duration,
        #                    delta_soc)

        period_amount = min(charger.charge_for(entry_soc, period.duration) - entry_soc, delta_soc)
        yield period, period_amount
        entry_soc += period_amount
        delta_soc -= period_amount
        if delta_soc <= 0.01:
            break
        if entry_soc >= next_segment.imageUpperBound:
            next_segment_id += 1
            next_segment = charger.chargingFunction[next_segment_id]
    assert delta_soc <= 0.01

def is_solution_integral(solution: Dict[Column, float]) -> bool:
    return all(weight <= INTEGRALITY_TOLERANCE or weight >= 1.0 - INTEGRALITY_TOLERANCE for weight in solution.values())

def solution_value(solution: Dict[Column, float]) -> float:
    return sum(col.cost * round(weight, INTEGRALITY_PRECISION) for col, weight in solution.items())

def is_solution_feasible(solution: Dict[Column, float]) -> bool:
    return len(solution) > 0 and all(not col.is_dummy for col in solution)

def calculate_gap(lb: Optional[float], ub: Optional[float]) -> float:
    if lb is None or isinf(lb) or ub is None:
        return 1.0
    return abs(lb-ub) / (1e-10 + ub)

def dump_pwl_to_txt(pwl: 'PiecewiseLinearFunction', out):
    out.write('BEGIN_PWL\n')
    out.write('// imageLB\n')
    out.write(f'{pwl.segments[0].imageLowerBound}\n')
    out.write('// breakpoints\n')
    out.write(' '.join(str(s.lowerBound) for s in pwl)+' '+str(pwl[len(pwl)-1].upperBound)+' \n')
    out.write('// slopes\n')
    out.write(' '.join(str(s.slope) for s in pwl)+'\n')
    out.write('END_PWL\n')

def dump_to_txt(network: 'FRVCPNetwork', out, expected_solution_value = None, coverage_dual = 0.0, obj_bound = 100000.0):
    raise NotImplementedError
    out.write(f'// Expected solution: {expected_solution_value}\n')
    out.write(f'// Coverage Dual: {coverage_dual}\n')
    out.write(f'// Upper Bound: {obj_bound}\n')
    out.write(f'// Latest Departure: {0}\n')
    # Dump battery
    out.write('BEGIN_BATTERY\n')
    out.write('// MinSoC MaxSoC Capacity InitCharge\n')
    out.write(f'{network.battery.minimumCharge} {network.battery.maximumCharge} {network.battery.capacity} {network.battery.initialCharge}\n')
    out.write('// WDF\n')
    dump_pwl_to_txt(network.battery.wearCostDensityFunction, out)
    out.write('END_BATTERY\n')
    # Dump chargers
    out.write('// Chargers\n')
    for charger in network.chargers:
        out.write(f'// Charger {str(charger)}\n')
        out.write('BEGIN_CHARGER\n')
        out.write(f'// baseCharger capacity\n')
        out.write(f'{int(charger.isBaseCharger)} {charger.capacity}\n')
        out.write(f'// Phi\n')
        dump_pwl_to_txt(charger.chargingFunction, out)
        out.write('END_CHARGER\n')
    # Dump Tours
    out.write('// Tours\n')
    for tour in network.tours:
        out.write(f'// Tour {str(tour)}\n')
        out.write('BEGIN_TOUR\n')
        out.write(f'// Departure Arrival Consumption Cost\n')
        out.write(f'{tour.departure_time} {tour.arrival_time} {tour.consumption} {tour.cost}\n')
        out.write('END_TOUR\n')
    out.write('// Nodes\n')
    next_node_id = 2
    tours = []
    reachable_nodes = get_reachable_nodes(network)
    for p, node_list in reachable_nodes.items():
        out.write(f'// Period {str(p)}\n')
        out.write('BEGIN_PERIOD\n')
        for node in node_list:
            if node.is_direct_path:
                continue
            out.write('BEGIN_NODE\n')
            out.write('// type charger/tour id node_id fix_costs energy_price begin_time end_time\n')
            if node.is_charger:
                node_type = 'C'
                op_id = node.charger.id
            elif node.is_direct_path:
                node_type = 'D'
                op_id = -1
            else:
                node_type = 'T'
                if not node.tour.id in tours:
                    tours.append(node.tour.id)
                op_id = tours.index(node.tour.id)
            out.write(f'{node_type} {op_id} {next_node_id} {node.fix_costs} {node.energy_price} {node.begin} {node.end}\n')
            out.write('// soc breakpoints\n')
            out.write(' '.join(str(x) for x in node.get('breakpoints', [])) + '\n')
            next_node_id += 1
            out.write('END_NODE\n')
        out.write('END_PERIOD\n')
