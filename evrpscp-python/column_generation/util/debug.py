# coding=utf-8
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

from docplex.mp.conflict_refiner import ConflictRefiner, ConflictRefinerResult
from docplex.mp.constr import LinearConstraint
from evrpscp import is_close
from evrpscp.models import Vehicle, DiscretePeriod, Charger, Battery, DiscreteTour

class CostMissmatchError(Exception):
    def __init__(self, expected_cost: float, calculated_cost: float, message: Optional[str] = None):
        self.expected_cost = expected_cost
        self.calculated_cost = calculated_cost
        super().__init__(message if message is not None else f'Cost missmatch! Expected {expected_cost}, got {calculated_cost} (delta: {expected_cost - calculated_cost})')

class ConstraintViolation(Exception):
    def __init__(self, constraint, message: Optional[str] = None):
        self.constraint = constraint
        super().__init__(message if message is not None else f'Constraint violation! Constraint {constraint} is violated.')

class TimeWindowViolation(Exception):
    pass

def check_column(column: 'Column', instance, constraints, coverage_duals, capacity_duals):
    for c in constraints:
        if c.is_violated(column):
            raise ConstraintViolation(c)
    # Asserts feasibility, i.e. throws if infeasible
    column.create_vehicle_schedule(instance)
    calculated_cost = calculate_column_objective(column, coverage_duals[column.vehicle], capacity_duals,
                                                 instance.parameters.battery,
                                                 instance.periods, instance.tours[column.vehicle])
    if not is_close(calculated_cost, column.objective):
        err = CostMissmatchError(expected_cost=column.objective, calculated_cost=calculated_cost)
        #print(f"[WARNING]: Cost missmatch (Column {column})! Ignoring error for now... ({err})")
        column.objective = calculated_cost
        raise err

    # Check for time window violations
    tours = instance.tours[column.vehicle]
    for pi in tours:
        departure_period = column.tour_departures[pi]
        if departure_period > pi.latest_departure:
            raise ConstraintViolation(f'Vehicle serves {pi} in {departure_period}, but latest departure is {pi.latest_departure}')
        if departure_period < pi.earliest_departure:
            raise ConstraintViolation(f'Vehicle serves {pi} in {departure_period}, but earliest departure is {pi.earliest_departure}')
    return calculated_cost


def calculate_column_objective(column: 'Column', coverage_dual: float,
                               capacity_dual: Dict[Tuple[DiscretePeriod, Charger], float], battery: Battery,
                        periods: List[DiscretePeriod], tours: List[DiscreteTour]) -> float:
    obj = -coverage_dual
    capacity_saving = 0.0
    energy_cost = 0.0
    for (period, charger), uses_charger in column.charger_usage.items():
        if uses_charger:
            obj -= capacity_dual[period, charger]
            charging_cost = column.energy_charged[period] * period.energyPrice
            obj += charging_cost

            energy_cost += charging_cost
            capacity_saving += -capacity_dual[period, charger]

    current_soc = battery.initialCharge
    wear_cost = 0.0
    periods_with_departures = {p: pi for pi, p in column.tour_departures.items()}
    for p in periods:
        if (next_tour := periods_with_departures.get(p)) is not None:
            current_soc -= next_tour.consumption
        else:
            entry_soc = current_soc
            current_soc += column.energy_charged[p]
            deg = battery.wearCost(entry_soc, current_soc)
            obj += deg
            wear_cost += deg

    tour_cost = sum(pi.cost for pi in tours)
    obj += tour_cost

    print(f'Schedule cost: {obj}, energy: {energy_cost}, deg: {wear_cost}, conv: {coverage_dual}, '
          f'capacity: {capacity_saving}, routing: {tour_cost}')

    return obj

def check_column_via_mip(column: 'Column', _mip_subproblem: 'MIPSubproblem') -> float:
    mip = deepcopy(_mip_subproblem)
    mip.set_solution(column, fix=True)
    sol = mip.model.solve()
    if not sol:
        conflict_refiner = ConflictRefiner()
        conflicts: ConflictRefinerResult = conflict_refiner.refine_conflict(mip.model, display=False)
        for x in conflicts:
            print(f'{x.element.as_constraint() if not isinstance(x.element, LinearConstraint) else x.element}')
        raise ValueError
    return sol.objective_value