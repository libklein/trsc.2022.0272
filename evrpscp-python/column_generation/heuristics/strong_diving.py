# coding=utf-8

from . import DivingHeuristic
import numpy as np
from contexttimer import Timer
from numpy.random import choice

from typing import Dict, List, Optional

from column_generation.util import setup_default_logger, solution_value
from column_generation.parameters import INTEGRALITY_TOLERANCE
from column_generation.column import Column
from itertools import chain

from funcy import group_by

logger = setup_default_logger(__name__)

class StrongDivingHeuristic(DivingHeuristic):
    def _evaluate_columns(self, proper_columns: List[Column]) -> Dict[Column, Optional[float]]:
        """
        Evaluates the lower bound when fixing a certain column.
        @param proper_columns: A list of columns that can potentially be added to the current solution
        @type proper_columns:
        @return A mapping of columns to objective values. Infeasible solutions get an objective of None
        """
        # Initialize solver
        solver = self._create_solver(proper_columns)
        # Initialize fixed columns
        columns = {column: True for column in self._fixed_columns}

        # Try fixing column candidates
        evaluated_columns: Dict[Column, Optional[float]] = {}
        for next_col in proper_columns:
            columns[next_col] = True

            # Assert initial conditions
            assert next_col.vehicle not in [x.vehicle for x in self._fixed_columns]
            # Check capacity
            for p in self._periods:
                for f, capacity in self._charger_capacities.items():
                    assert sum(col.charger_usage[p, f] for col in columns) <= capacity

            with Timer(factor=1000) as fix_solve_timer:
                candidate_solution = solver.solve(columns=columns)
            logger.debug(f'MIP Evaluation took a total of {fix_solve_timer}ms')
            if candidate_solution is None:
                candidate_objective = None
            else:
                candidate_objective = solution_value(candidate_solution)
            logger.debug(f'Evaluated column {next_col}: {candidate_objective}')
            del columns[next_col]
            evaluated_columns[next_col] = candidate_objective

        return evaluated_columns

    def _assign_weights(self, columns_with_objectives: Dict[Column, Optional[float]]) -> Optional[Dict[Column, float]]:
        # Normalize to [0.0, 1.0].
        weighted_columns = columns_with_objectives.copy()
        total_objective = 0.0
        for col, obj_val in weighted_columns.items():
            weighted_columns[col] = 1.0/obj_val if obj_val is not None else 0.0
            total_objective += weighted_columns[col]

        # No feasible solutions
        if total_objective == 0.0:
            return None

        for col in weighted_columns:
            weighted_columns[col] /= total_objective

        return weighted_columns

    def _select_column(self, columns_with_weights: Dict[Column, float]) -> Column:
        return max(columns_with_weights.items(), key=lambda x: x[1])[0]

    def _select_column_subset(self, columns: List[Column], column_weights: Optional[Dict[Column, float]] = None, max_size: int = 10, min_size: int = 1) -> List[Column]:
        subset_size = min(max_size, len(columns), max(round(0.1 * len(columns)), min_size))
        if subset_size >= len(columns):
            return columns

        if column_weights is None:
            column_weights = {col: 1.0 for col in columns}

        non_zero_weighted_cols = []
        zero_weighted_cols = []
        for col in columns:
            if column_weights[col] >= INTEGRALITY_TOLERANCE:
                non_zero_weighted_cols.append(col)
            else:
                zero_weighted_cols.append(col)

        weights = np.fromiter((column_weights[col] for col in non_zero_weighted_cols), dtype=np.float,
                              count=len(non_zero_weighted_cols))
        weights /= np.sum(weights)

        promising_columns = list(choice(non_zero_weighted_cols, p=weights, replace=False,
                                        size=min(subset_size, len(non_zero_weighted_cols))))

        promising_columns.extend(choice(zero_weighted_cols, replace=False, size=subset_size - len(promising_columns)))

        return promising_columns

    def _find_diving_candidate(self, proper_columns: List[Column]) -> Optional[Column]:
        if len(proper_columns) == 0:
            logger.info(f'Could not find any fixable column.')
            return None

        # Select subset of interesting columns
        candidates_per_vehicle = group_by(lambda col: col.vehicle, proper_columns)
        max_cols_per_vehicle = min(100, len(proper_columns), round(0.1 * len(proper_columns))) // (
                len(self._vehicles) - len(self._fixed_columns))
        if max_cols_per_vehicle == 0:
            return None
        interesting_candidates: List[Column] = list(chain(*(
            self._select_column_subset(columns=cols, column_weights=self._current_solution, min_size=1,
                                       max_size=max_cols_per_vehicle)
            for cols in candidates_per_vehicle.values()
        )))
        assert set(self._vehicles) == (
                {col.vehicle for col in interesting_candidates} | set(col.vehicle for col in self._fixed_columns))
        logger.info(f'Considering {len(interesting_candidates)} of {len(proper_columns)} (proper) columns to fix')
        # Evaluate candidates
        weighted_column_candidates = self._evaluate_columns(proper_columns=proper_columns)
        # Assign weights according to objective value
        weighted_column_candidates = self._assign_weights(columns_with_objectives=weighted_column_candidates)

        if weighted_column_candidates is None:
            logger.info(f'None of the fixable columns are feasible.')
            return None

        # Select column to fix according to weights
        return self._select_column(columns_with_weights=weighted_column_candidates)