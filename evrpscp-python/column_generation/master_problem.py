# coding=utf-8
from contextlib import ExitStack

from docplex.mp.linear import LinearExpr, Var
from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint
from docplex.mp.solution import SolveSolution
from typing import *

from funcy import group_by

from column_generation.util import setup_default_logger
from column_generation import EPS, BIG_M_COEF, DiscretizedInstance, Column, Constraint
from evrpscp import Charger, DiscretePeriod, Vehicle
from itertools import product
from copy import copy
import knapsack

logger = setup_default_logger(__name__)

CoverageDual = Dict[Vehicle, float]
CapacityDual = Dict[Tuple[DiscretePeriod, Charger], float]


def solve_knapsack(weights: List[float], profits: List[float], capacity: float) -> List[int]:
    assert len(weights) == len(profits)
    profit, items = knapsack.knapsack(weights, profits).solve(capacity)
    return items


def separate_cover_cg_cut(fractional_solution: Dict[Column, float]) -> Dict[Column, float]:
    m = Model("CG cuts")

    cols = list(fractional_solution.keys())
    vehicles = [col.vehicle for col in cols]

    gamma = m.integer_var_dict(fractional_solution, lb=0)
    gamma_0 = m.integer_var(name='gamma_0', lb=0)

    u = m.continuous_var_dict(vehicles, lb=0)
    delta = 0.01
    f = m.continuous_var_dict(fractional_solution, lb=0, ub=1-delta)
    f_0 = m.continuous_var(lb=0, ub=1-delta)
    ub = m.integer_var(lb=0)

    m.maximize(sum(g*l for g,l in zip(gamma.values(), fractional_solution.values())) - gamma_0 - sum(1e-6*u_i for u_i in u.values()))

    m.add_constraints((f[col] == sum((u_i if veh == col.vehicle else 0) for veh, u_i in u.items()) - gamma[col]) for col in fractional_solution)
    m.add_constraint(f_0 == (ub - gamma_0))
    m.add_constraint(ub == sum(u.values()))

    sol = m.solve()
    assert sol is not None
    if any(u_i.solution_value > 0 for u_i in u.values()):
        print({veh: u_i.solution_value for veh, u_i in u.items() if u_i.solution_value > 0.00001})
        print(f'UB: {sum(u_i.solution_value for u_i in u.values())}')
    return {veh: u_i.solution_value for veh, u_i in u.items()}

def separate_capacity_cg_cut(fractional_solution: Dict[Column, float]) -> Dict[Column, float]:
    m = Model("CG cuts")

    cols = list(fractional_solution.keys())

    gamma = m.integer_var_dict(fractional_solution, lb=0)
    gamma_0 = m.integer_var(name='gamma_0', lb=0)

    u = m.continuous_var_dict(cols[0].charger_usage, lb=0)
    delta = 0.01
    f = m.continuous_var_dict(fractional_solution, lb=0, ub=1-delta)
    f_0 = m.continuous_var(lb=0, ub=1-delta)
    ub = m.integer_var(lb=0)

    m.maximize(sum(g*l for g,l in zip(gamma.values(), fractional_solution.values())) - gamma_0 - sum(1e-6*u_i for u_i in u.values()))

    m.add_constraints((f[col] == sum(u[p, f] * (1 if uses else 0) for (p, f), uses in col.charger_usage.items()) - gamma[col]) for col in fractional_solution)
    m.add_constraint(f_0 == (ub - gamma_0))
    m.add_constraint(ub == sum(u_i*f.capacity for (p, f), u_i in u.items()))

    sol = m.solve()
    if sol is None:
        print("No CB cut separated")
        return {}

    print({(p, f): u_i.solution_value for (p, f), u_i in u.items() if u_i.solution_value > 0.00001})
    print(f'UB: {sum(u_i.solution_value for u_i in u.values())}')
    return {(p, f): u_i.solution_value for (p, f), u_i in u.items()}

def separate_cg_cut(solution: Dict[Column, float]):
    return separate_cover_cg_cut(solution)

class MasterProblem:
    def __init__(self, instance: DiscretizedInstance, **cli_args):
        self.instance: DiscretizedInstance = instance
        self.dummy_columns: Dict[int, Column] = {
            k: Column.DummyColumn(instance.periods, instance.chargers, instance.tours[k], k)
            for k in instance.vehicles}
        self.coverage: Dict[int, LinearConstraint] = {}
        self.capacity: Dict[[Charger, DiscretePeriod], LinearConstraint] = {}
        self.x: Dict[Column, Var] = {}

        self._x_idx: Dict[Column, int] = {}
        self._coverage_idx: Dict[int, int] = {}
        self._capacity_idx: Dict[[Charger, DiscretePeriod], int] = {}
        self._robust_cuts = []

        self._has_been_optimized: bool = False

        self.model = self._build_initial_model()
        self.model.parameters.parallel = 1  # Force deterministic mode
        if cli_args.get('use_barrier'):
            self.model.parameters.lpmethod = 4  # Force barrier algorithm
        else:
            self.model.parameters.lpmethod = 2
        self.model.parameters.threads = 1  # Sequential mode
        self.model.set_log_output(None)

    def __deepcopy__(self, memodict={}) -> 'MasterProblem':
        # Does not deepcopy columns.
        clone = copy(self)
        clone._x_idx = self._x_idx.copy()
        clone._coverage_idx = self._coverage_idx.copy()
        clone._capacity_idx = self._capacity_idx.copy()
        # Clone model and update constraints
        clone.model = self.model.clone()
        clone._build_from_index()
        # Cloning requires re-solving
        clone._has_been_optimized = False

        return clone

    @property
    def has_been_solved(self) -> bool:
        return self._has_been_optimized

    @property
    def has_integral_solution(self) -> bool:
        assert self.has_been_solved
        return self.has_feasible_solution and all(
            (var.solution_value > (1 - EPS) or var.solution_value < EPS for var in self.x.values()))

    @property
    def has_feasible_solution(self) -> bool:
        assert self.has_been_solved
        for next_dummy_col in self.dummy_columns.values():
            # Infeasible if a dummy variable is part of the schedule
            if self.x[next_dummy_col].solution_value > 1 - EPS:
                return False
        return True

    @property
    def columns(self) -> Iterable[Column]:
        return self.x.keys()

    @property
    def objective_value(self) -> Optional[float]:
        if self.has_been_solved and self.model.solve_details is not None:
            return self.model.objective_value
        else:
            return None

    @property
    def solution(self) -> Dict[Column, float]:
        assert self.has_been_solved
        return {col: variable.solution_value for col, variable in self.x.items() if variable.solution_value >= EPS}

    def _solve_separation_problem(self, period: DiscretePeriod, charger: Charger, solution: Dict[Column, float]):
        return None
        # Only include colums that charge in this period at this charger
        columns = {x: val for x, val in solution.items() if x.charger_usage[period, charger]}
        if len(columns) == 0:
            return None
        # Construct MIP
        separation_problem = Model("Separation Problem")
        z: Dict[Column, Var] = separation_problem.binary_var_dict(columns)

        separation_problem.minimize(sum((1 - x_i) * z_i for z_i, x_i in zip(z.values(), columns.values())))
        separation_problem.add_constraint(sum(z.values()) >= charger.capacity + 1e-3)

        k = separation_problem.solve()
        if k is None or k.objective_value >= 1.0 - 1e-3:
            return None
        else:
            return [z_i for z_i, val in z.items() if val.solution_value > 1e-3]

    def _separate_cuts(self, solution: Dict[Column, float]) -> List:
        return []
        separate_cg_cut({x: val for x, val in solution.items() if val > 0.0001})
        covers = []
        for p, f in product(self.instance.periods, self.instance.chargers):
            cover = self._solve_separation_problem(p, f, solution)
            if cover is not None:
                covers.append(cover)
                logger.debug(f"Separated cover cut in {p} at {f}: {cover}")
        return []

    def solve(self) -> Optional[SolveSolution]:
        while not self._has_been_optimized:
            sol = self.model.solve()
            if len(cuts := self._separate_cuts({col: variable.solution_value for col, variable in self.x.items() if
                                                variable.solution_value >= EPS})) > 0:
                # Add cuts
                self._robust_cuts.extend(cuts)
            else:
                self._has_been_optimized = True

        if not sol:
            logger.info(f'Master problem failed to find solution ({self.model.solve_details.time / 1000.0:.2f}ms)!')
            return None
        logger.info(
            f'Master problem found solution with obj value {sol.objective_value} ({self.model.solve_details.time / 1000.0:.2f}ms)!')

        # Output solution grouped by vehicle
        schedules_by_vehicle = group_by((lambda col_var_pair: col_var_pair[0].vehicle),
                                        filter(lambda col_var_pair: col_var_pair[1].solution_value > EPS,
                                               self.x.items()))
        for vehicle, schedules in sorted(schedules_by_vehicle.items(), key=lambda col_var_pair: col_var_pair[0]):
            logger.debug(f'Vehicle {vehicle}: ')
            for col, x in schedules:
                logger.debug(f'{x.name}: {x.solution_value} -> {col}')

        return sol

    def get_coverage_duals(self) -> CoverageDual:
        assert self.has_been_solved
        return {vehicle: dual for vehicle, dual in
                zip(self.coverage.keys(), self.model.dual_values(self.coverage.values()))}

    def get_capacity_dual(self) -> CapacityDual:
        assert self.has_been_solved
        duals = self.model.dual_values(self.capacity.values())
        return {key: dual for key, dual in zip(self.capacity.keys(), duals)}

    def add_column(self, column: Column) -> Var:
        assert column is not None
        logger.debug(f'Adding column {column}.')
        return self._add_var(column)

    def remove_var(self, column: Column):
        if not column.is_dummy:
            logger.debug(f'Removing column {column}')
            self._remove_var(column)

    def add_constraint(self, constraint: Constraint):
        columns_to_remove = [col for col in self.columns if constraint.is_violated(col)]
        for col in columns_to_remove:
            self.remove_var(col)

    def _add_var(self, column: Column) -> Var:
        # Modify the model
        # Add variable
        x_i = self.x[column] = self.model.continuous_var(lb=0.0, ub=1.0, name=f'x_{column.id}')
        assert x_i is not None
        self._x_idx[column] = x_i.index
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

        self._has_been_optimized = False

        return x_i

    def _remove_var(self, column: Column):
        assert not column.is_dummy
        x_i = self.x.pop(column)
        del self._x_idx[column]

        self.model.add_constraint(x_i == 0, f'del_{column.id}')

        self._has_been_optimized = False

    def _build_from_index(self):
        self.x = {col: self.model.get_var_by_index(idx) for col, idx in self._x_idx.items()}
        self.coverage = {col: self.model.get_constraint_by_index(idx) for col, idx in self._coverage_idx.items()}
        self.capacity = {col: self.model.get_constraint_by_index(idx) for col, idx in self._capacity_idx.items()}

    def _build_initial_model(self) -> Model:
        model = Model('Master Problem')
        model.init_numpy()
        self.x = {col: model.continuous_var(lb=0.0, ub=1.0, name=f'x^d_{i}') for i, col in
                  enumerate(self.dummy_columns.values())}
        self._x_idx = {key: var.index for key, var in self.x.items()}
        # Objective
        model.minimize(model.scal_prod(iter(self.x.values()), BIG_M_COEF))

        # Add coverage constraint
        self.coverage = {k: model.add_constraint(self.x[self.dummy_columns[k]] >= 1, ctname=f'coverage_{k}')
                         for k in self.instance.vehicles}
        self._coverage_idx = {key: var._index for key, var in self.coverage.items()}
        # Station capacity
        self.capacity = {(p, f): model.add_constraint(model.sum(x * int(col.charger_usage[p, f])
                                                                for col, x in self.x.items())
                                                      <= f.capacity, ctname=f'capacity^{p}_{f}')
                         for f in self.instance.chargers for p in self.instance.periods}
        self._capacity_idx = {key: var._index for key, var in self.capacity.items()}

        return model
