# coding=utf-8
from collections import defaultdict
from itertools import chain
from typing import List, Dict, Optional, Iterable, Tuple

from column_generation import Column, INTEGRALITY_TOLERANCE
from docplex.mp.dvar import Var
from docplex.mp.model import Model
from evrpscp import DiscretizedInstance, Charger, Vehicle, DiscretePeriod
from funcy import compact
from column_generation.util import is_solution_integral, setup_default_logger
from column_generation.solution import Solution

logger = setup_default_logger(__name__)

class CandidateEvaluator:
    """
    Evaluates candidates.
    """

    def __init__(self, columns: Iterable[Column], vehicles: Iterable[Vehicle], charger_capacity: Dict[Charger, int], periods: List[DiscretePeriod]):
        self.vehicles = list(vehicles)
        self.charger_capacity = charger_capacity
        self.periods = periods
        self._model, self._col_var_mapper = self._build_initial_model(columns=columns, vehicles=vehicles,
                                                                      charger_capacity=charger_capacity, periods=periods)


    @staticmethod
    def FromInstance(columns: Iterable[Column], instance: DiscretizedInstance) -> 'CandidateEvaluator':
        """
        Creates a CanididateEvalator from a set of columns and an instance.
        @param columns: An iterable sequence of columns that should be evaluated, i.e., occur in the mip.
        @param instance: A discrete instance.
        @return: A CandidateEvaluator instance.
        """
        return CandidateEvaluator(columns=columns, vehicles=instance.vehicles, charger_capacity={
            f: f.capacity for f in instance.chargers
        }, periods=instance.periods)

    def solve(self, columns: Dict[Column, bool]) -> Optional[Dict[Column, float]]:
        # Fix columns
        added_constraints = self._model.add_constraints(self._col_var_mapper[col] >= 1.0 for col in compact(columns))

        self._model.solve()

        solution = {col: var.solution_value for col, var in self._col_var_mapper.items()} \
            if self._model.solution else None

        self._model.remove_constraints(added_constraints)

        return solution

    def _build_initial_model(self, columns: Iterable[Column], vehicles: Iterable[Vehicle],
                             charger_capacity: Dict[Charger, int], periods: List[DiscretePeriod]) \
            -> Tuple[Model, Dict[Column, Var]]:
        model = Model('Diving Heuristic')
        model.parameters.lpmethod = 1
        model.parameters.parallel = 1 # Force deterministic mode
        model.parameters.threads = 1 # Force sequential solve
        model.time_limit = 60 # Max 1 minutes

        x = model.continuous_var_dict(columns, lb=0.0, ub=1.0, name="x")
        # Objective
        model.minimize(model.sum(var * col.cost for col, var in x.items()))

        # Coverage/Convexity
        model.add_constraints((model.sum(x for col, x in x.items() if col.vehicle == k) >= 1 for k in vehicles),
                              names=(f'Coverage_v{k}' for k in vehicles))

        # Station capacity
        model.add_constraints((model.sum(x * int(col.charger_usage[p, f]) for col, x in x.items()) <= capacity
                               for f, capacity in charger_capacity.items() for p in periods),
                              names=(f'Capacity_{f}_{p}' for f, capacity in charger_capacity.items() for p in periods))
        return model, x

class DivingHeuristic:
    def __init__(self, instance: DiscretizedInstance):
        # Stack of fixed columns
        self._fixed_columns: List[Column] = []
        # Current solution
        self._current_solution: Dict[Column, int] = {}
        self._current_objective: Optional[float] = None
        # Charger capacities
        self._charger_capacities: Dict[Charger, int] = {f: f.capacity for f in instance.chargers}
        self._periods = instance.periods
        self._vehicles = instance.vehicles
        # Solver
        self.__solver: Optional[CandidateEvaluator] = None

    def _get_proper_columns(self, columns: List[Column], fixed_columns: List[Column]) -> List[Column]:
        joint_charger_usage = defaultdict(int)
        for column in fixed_columns:
            for (p, f), uses_charger in column.charger_usage.items():
                if uses_charger:
                    joint_charger_usage[p, f] += 1

        # Remove all
        improper_columns = set()
        # Already fixed a column for the vehicle
        vehicles_with_fixed_column = set(col.vehicle for col in fixed_columns)
        for col in columns:
            if col.vehicle in vehicles_with_fixed_column:
                improper_columns.add(col)
        # Charges in period that is already at capacity
        for (p, f), usage_count in joint_charger_usage.items():
            if usage_count < self._charger_capacities[f]:
                continue
            for col in columns:
                if col.charger_usage[p, f]:
                    improper_columns.add(col)

        proper_columns = list(set(columns) - improper_columns)

        logger.debug(f'Identified proper columns:')
        for col in proper_columns:
            logger.debug(col)

        return proper_columns

    def _create_solver(self, columns: List[Column]) -> CandidateEvaluator:
        veh_set = {col.vehicle for col in chain(columns, self._fixed_columns)}
        assert len(veh_set) == len(self._vehicles)

        return CandidateEvaluator(columns=chain(columns, self._fixed_columns), vehicles=self._vehicles,
                                  charger_capacity=self._charger_capacities, periods=self._periods)

    def _compute_current_solution(self):
        columns_to_fix = {column: True for column in self._fixed_columns}

        self._current_solution = self.__solver.solve(columns=columns_to_fix)
        if self._current_solution is not None:
            self._current_objective = sum(col.cost * x for col, x in self._current_solution.items())
        else:
            self._current_objective = None
        logger.debug(f'Current solution ({self._current_objective}): {self._current_solution}')

    def _fix_column(self, column: Column):
        self._fixed_columns.append(column)
        self._compute_current_solution()

    def _is_integral(self) -> bool:
        return is_solution_integral(self._current_solution)

    def _is_infeasible(self) -> bool:
        return self._current_solution is None

    def _find_diving_candidate(self, proper_columns: List[Column]) -> Optional[Column]:
        return max(proper_columns, key=lambda col: self._current_solution[col])

    def solve(self, columns: List[Column], best_ub: Optional[Solution] = None) -> Optional[Solution]:
        assert len(self._fixed_columns) == 0
        sol = self._solve(columns=columns)
        if sol is None:
            return None
        return Solution({col: 1.0 for col in sol})

    def _solve(self, columns: List[Column]) -> Optional[List[Column]]:
        assert all((col in columns) for col in self._fixed_columns)
        # Initialization
        self.__solver = self._create_solver(self._get_proper_columns(columns, self._fixed_columns))
        self._compute_current_solution()
        while not self._is_integral():
            logger.info(f'Diving from initial solution with obj value {self._current_objective}')
            if (column_to_fix := self._find_diving_candidate(proper_columns=self._get_proper_columns(
                    columns=columns, fixed_columns=self._fixed_columns))) is None:
                return None
            logger.debug(f'Selected column {column_to_fix}')
            # Fix column and update the current solution
            self._fix_column(column_to_fix)
            # This should never happen as evaluating the weights should already fail
            if self._is_infeasible():
                logger.info(f'Fixing column {column_to_fix} yields infeasible LP relaxation.')
                return None

        logger.info(f'Found final integral solution with objective {self._current_objective}')
        logger.debug(f'Solution: {self._current_solution}')

        return [col for col, val in self._current_solution.items() if val > 1.0 - INTEGRALITY_TOLERANCE]