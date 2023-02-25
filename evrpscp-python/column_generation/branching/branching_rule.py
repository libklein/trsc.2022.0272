# coding=utf-8
from typing import Set, List, Tuple, Dict, Any, Protocol
from itertools import product
from pprint import pformat as pretty_print

from evrpscp import Charger, DiscretePeriod

from column_generation.solution import Solution
from column_generation.util import setup_default_logger, plot_column_charger_usage, Vehicle
from column_generation.constraints import BlockChargerConstraint, Constraint
from column_generation import DiscretizedInstance, Column, EPS
from evrpscp import is_close
from funcy import first, none, select_keys, map


logger = setup_default_logger(__name__)

class BranchingRule(Protocol):

    def create_branches(self, solution: Solution) -> List[Set[Constraint]]:
        raise NotImplementedError


class PeriodBranchingRule:
    def __init__(self, instance: DiscretizedInstance):
        self.instance = instance

    @staticmethod
    def _filter_by_energy_price(conflicts: Dict[Tuple['DiscretePeriod', 'Charger'], Any]) -> Dict[Tuple[DiscretePeriod, Charger], Any]:
        min_energy_price = min(map(lambda x: x[0].energyPrice, conflicts))
        return select_keys(lambda x: x[0].energyPrice == min_energy_price, conflicts)

    @staticmethod
    def _filter_by_charging_speed(conflicts: Dict[Tuple['DiscretePeriod', 'Charger'], Any]) -> Dict[Tuple[DiscretePeriod, Charger], Any]:
        max_charging_speed = max(map(lambda x: x[1].max_charging_rate, conflicts))
        return select_keys(lambda x: x[1].max_charging_rate == max_charging_speed, conflicts)

    @staticmethod
    def _select_vehicles(conflicts: Dict[Tuple['DiscretePeriod', 'Charger'], List[Vehicle]]) -> Tuple[Tuple[DiscretePeriod, Charger], List[Vehicle]]:
        # Select the conflict with the most fractional vehicles
        return max(conflicts.items(), key=lambda item: len(item[1]))
        #return min(map(lambda item: (item[0], min(item[1])), conflicts.items()), key=lambda item: item[1])

    @staticmethod
    def _remove_conflicting_vehicles(active_constraints: Set[Constraint], conflicts: Dict[Tuple['DiscretePeriod', 'Charger'], List[Vehicle]]) -> Dict[Tuple[DiscretePeriod, Charger], Vehicle]:
        constraints = {(c.period, c.charger, c.vehicle) for c in active_constraints if isinstance(c, BlockChargerConstraint)}
        conflicts_without_constraints: Dict[Tuple[DiscretePeriod, Charger], List[Vehicle]] = {}
        for (p,f), vehicles in conflicts.items():
            for v in vehicles:
                if (p,f,v) in constraints:
                    continue
                conflicts_without_constraints.setdefault((p, f), list()).append(v)
        return conflicts_without_constraints


    def create_branches(self, solution: Solution) -> List[Set[BlockChargerConstraint]]:
        logger.info('-----------------------------------------')
        active_columns = list(solution)
        non_integral_columns: Dict[Column, float] = {col: x for col, x in solution.items() if EPS <= x <= 1.0-EPS}

        # Conflicts is a list of all vehicles with non integral columns competing for charger f in period p.
        conflicts: Dict[Tuple['DiscretePeriod', 'Charger'], List['Vehicle']] = {}
        for period, charger in product(self.instance.periods, self.instance.chargers):
            columns_using_charger = [col for col in active_columns if col.charger_usage[period, charger]]
            involved_vehicles = set(col.vehicle for col in columns_using_charger)
            if len(involved_vehicles) > charger.capacity:
                # Charger usage is binding and at least one column is non-integral. So this is a potential branching candidate
                conflicts[period, charger] = list(involved_vehicles)

        if len(conflicts) == 0:
            logger.critical(f'Could not branch on node with solution:\n{pretty_print(solution)}')
            raise RuntimeError('Could not create branches')

        logger.debug(f'Branching on node with solution:\n{pretty_print(solution)}.')
        logger.debug(f'Found conflicts:\n{pretty_print(conflicts)}')

        # Select by price, charger speed and number of conflicting vehicles
        conflicts = PeriodBranchingRule._filter_by_energy_price(conflicts)
        conflicts = PeriodBranchingRule._filter_by_charging_speed(conflicts)
        (period, charger), conflicting_vehicles = PeriodBranchingRule._select_vehicles(conflicts)
        logger.info(f'Selected vehicles {conflicting_vehicles} for branching on charger {charger} in {period}')

        return [
            {BlockChargerConstraint(vehicle, charger=charger, period=period, allocate=False)}
            for vehicle in conflicting_vehicles
        ]