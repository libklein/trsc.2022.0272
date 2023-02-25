# coding=utf-8
import sys

from docplex.mp.linear import LinearExpr, Var
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from typing import *

from column_generation.util import setup_default_logger
from column_generation.parameters import EPS
from column_generation.discretization import DiscretizedInstance
from column_generation.column import Column
from column_generation.solution import Solution
from itertools import product


logger = setup_default_logger(__name__)


class PrimalMPHeuristic:
    def __init__(self, instance: DiscretizedInstance):
        self.instance: DiscretizedInstance = instance
        self.x: Dict[Column, Var] = {}
        self.model: Optional[Model] = None

    @property
    def columns(self) -> Iterable[Column]:
        return self.x.keys()

    @property
    def objective_value(self) -> Optional[float]:
        if self.model.is_optimized() and self.model.solution.has_objective():
            return self.model.objective_value
        else:
            return None

    @property
    def solution(self) -> Dict[Column, float]:
        assert self.model.is_optimized()
        return {col: variable.solution_value for col, variable in self.x.items() if variable.solution_value >= EPS}

    def _set_starting_solution(self, columns: Iterable[Column]):
        start_sol = SolveSolution(self.model)
        for col in columns:
            if (var := self.x.get(col)) is not None:
                start_sol.add_var_value(var, 1.0)
        self.model.add_mip_start(start_sol)

    def solve(self, columns: List[Column], best_ub: Optional[Solution] = None) -> Optional[Solution]:
        self.model = self._build_initial_model(columns)
        if best_ub is not None:
            self._set_starting_solution(best_ub)
        logger.info(f'Built model with {self.model.number_of_integer_variables} integer variables '
                    f'({len(columns)} columns were passed)')
        self.model.parameters.parallel = 1 # Force deterministic mode
        self.model.parameters.threads = 1 # Force sequential solve
        self.model.time_limit = 300 # Max 5 minutes
        self.model.parameters.tune.display = 3
        self.model.context.log_output = True
        self.model.get_cplex().set_results_stream(sys.stdout)
        self.model.parameters.mip.display = 3
        for i in ('covers', 'bqp', 'cliques', 'disjunctive', 'flowcovers', 'pathcut', 'gomory', 'gubcovers', 'implied', 'localimplied', 'liftproj', 'mircut', 'mcfcut', 'rlt', 'zerohalfcut'):
            setattr(self.model.parameters.mip.cuts, i, 2)
        sol = self.model.solve()
        if not sol:
            logger.info(f'Primal heuristic failed to find a feasible solution in {self.model.solve_details.time}! Status: {self.model.solve_details.status}')
            return None
        return Solution(self.solution)

    def _add_var(self, column: Column) -> Var:
        # Modify the model
        # Add variable
        x_i = self.x[column] = self.model.binary_var(name=f'x_{column.id}')
        # Modify coverage constraint
        coverage_constraint = self.coverage[column.vehicle]
        coverage_constraint.lhs += x_i
        # Modify charger usage constraint
        for (p, f) in product(self.instance.periods, self.instance.chargers):
            if column.charger_usage[p, f]:
                self.capacity[p, f].lhs += x_i * 1
        # Modify objective
        objective: LinearExpr = self.model.objective_expr
        objective += x_i * column.cost
        self.model.minimize(objective)
        return x_i

    def _build_initial_model(self, columns: Iterable[Column]) -> Model:
        model = Model('Primal Heuristic')
        self.x = {col: model.binary_var(name=f'x^d_{i}') for i, col in enumerate(columns)}
        # Objective
        model.minimize(model.sum(var * col.cost for col, var in self.x.items()))

        # Add coverage constraint
        self.coverage = {k: model.add_constraint(model.sum(x for col, x in self.x.items() if col.vehicle == k) == 1, ctname=f'coverage_{k}')
                              for k in self.instance.vehicles}
        # Station capacity
        self.capacity = {(p, f): model.add_constraint(model.sum(x * (1 if col.charger_usage[p, f] else 0)
                                                                     for col, x in self.x.items())
                                                           <= f.capacity, ctname=f'capacity^{p}_{f}')
                              for f in self.instance.chargers for p in self.instance.periods}

        return model
