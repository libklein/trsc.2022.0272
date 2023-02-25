# coding=utf-8
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Generator, Iterable, Set, Protocol, Callable, Type, TypeVar
from copy import deepcopy, copy
from sys import maxsize
from collections import deque

from funcy import without
from math import inf

import docplex.mp.utils
from contexttimer import Timer

from column_generation.solution import Solution
from column_generation.util import setup_default_logger, is_solution_feasible, is_solution_integral, calculate_gap, solution_value
from column_generation.subproblem import SubProblemInfeasible, CPPSubproblem

from column_generation.constraints import Constraint
from column_generation import Column
from column_generation.master_problem import CoverageDual, CapacityDual, MasterProblem
from evrpscp import *
from itertools import chain

logger = setup_default_logger(__name__)

Subproblem = TypeVar('Subproblem')

# Can't this be a function? Does it need state?
class ColumnSelectionStrategy:
    def select_columns(self, columns: Dict[Vehicle, Column], current_solution: Dict[Column, float]) -> Dict[Vehicle, Column]:
        return columns


class ColumnGenerator:
    def __init__(self, subproblems: Dict[Vehicle, Subproblem]):
        self.subproblems = subproblems

    def add_constraint(self, constraint: Constraint):
        for s in self.subproblems.values():
            s.add_constraint(constraint)

    def _generate_column(self, subproblem: Subproblem, coverage_dual: float, capacity_duals: CapacityDual) -> Optional[Column]:
        """
        Generates a column using the specified subproblem according to the specified dual variables.
        @param subproblem:
        @type subproblem:
        @param coverage_dual:
        @type coverage_dual:
        @param capacity_duals:
        @type capacity_duals:
        @return: The generated column or None if no column with negative reduced cost could be generated.
        @rtype:
        """
        return subproblem.generate_column(coverage_dual=coverage_dual, capacity_dual=capacity_duals, obj_threshold=1e-2)

    def generate_columns(self, coverage_duals: Dict[Vehicle, float], capacity_duals: CapacityDual) -> Dict[Vehicle, Column]:
        """
        Generate columns with negative reduced costs according to the specified dual variables.
        @param coverage_duals:
        @type coverage_duals:
        @param capacity_duals:
        @type capacity_duals:
        @return: A dict of columns with negative reduced cost keyed by vehicle. An empty dict indicates that no columns
        could be generated.
        @rtype:
        """
        logger.debug(f'Generating columns according to coverage duals: {coverage_duals} and capacity duals: {capacity_duals}')
        columns = {}
        try:
            for vehicle, subproblem in self.subproblems.items():
                with Timer(factor=1000) as generation_timer:
                    column = self._generate_column(subproblem=subproblem, coverage_dual=coverage_duals[vehicle],
                                                   capacity_duals=capacity_duals)
                if column is not None:
                    columns[vehicle] = column
                    logger.info(f'[Vehicle {vehicle}] generated column {column} in {generation_timer}ms')
                else:
                    logger.info(f'[Vehicle {vehicle}] did not find improving column in {generation_timer}ms')
        except SubProblemInfeasible:
            return {}
        return columns


class VehicleOrdering(Protocol):
    def order(self, vehicles: List[Vehicle]) -> Iterable[Vehicle]:
        ...


class RandomizedVehicleOrdering:
    def __init__(self):
        self._vehicles_without_improvement: Set[Vehicle] = set()
        self._random_engine = random.Random(0) # Force determinism

    def order(self, vehicles: List[Vehicle]) -> Iterable[Vehicle]:
        vehicle_order = list(without(vehicles, *self._vehicles_without_improvement))
        self._random_engine.shuffle(vehicle_order)
        vehicle_order.extend(self._vehicles_without_improvement)
        return vehicle_order


class ScoreProvider(Protocol):
    def compute_score(self, vehicles: List[Vehicle]) -> Dict[Vehicle, float]:
        ...


class DualScoreProvider:
    def __init__(self, master_problem: MasterProblem):
        self._master_problem = master_problem
        self._prev_coverage_dual = None
        self._prev_capacity_dual = None

    def _get_current_iteration_duals(self) -> Tuple[CoverageDual, CapacityDual]:
        return self._master_problem.get_coverage_duals(), self._master_problem.get_capacity_dual()

    def _compute_reduced_cost(self, schedule: Column, coverage_dual: float, capacity_dual: CapacityDual) -> float:
        return schedule.cost - coverage_dual - sum(x * capacity_dual[p, f] for (p, f), x in schedule.charger_usage.items())

    def _compute_iteration_score(self, schedule: Column, coverage_dual: float, capacity_dual: CapacityDual) -> float:
        new_reduced_cost = self._compute_reduced_cost(schedule, coverage_dual=coverage_dual, capacity_dual=capacity_dual)
        if self._prev_capacity_dual is not None and self._prev_capacity_dual is not None:
            last_iteration_reduced_cost = self._compute_reduced_cost(schedule=schedule, coverage_dual=self._prev_coverage_dual[schedule.vehicle], capacity_dual=self._prev_capacity_dual)
        else:
            last_iteration_reduced_cost = 1
        return (new_reduced_cost - last_iteration_reduced_cost) / (last_iteration_reduced_cost if last_iteration_reduced_cost != 0.0 else 1.0)

    def compute_score(self, vehicles: List[Vehicle]) -> Dict[Vehicle, float]:
        coverage_duals, capacity_duals = self._get_current_iteration_duals()
        scores: Dict[Vehicle, float] = {}
        for vehicle in vehicles:
            score = sum(x * self._compute_iteration_score(schedule, coverage_dual=coverage_duals[vehicle], capacity_dual=capacity_duals) for schedule, x in self._master_problem.solution.items() if schedule.vehicle == vehicle)
            scores[vehicle] = score
        self._prev_coverage_dual = coverage_duals
        self._prev_capacity_dual = capacity_duals
        return scores


class AdaptiveVehicleOrdering:
    def __init__(self, weight: float, score_provider: ScoreProvider):
        self._score_provider = score_provider
        self._weight = weight
        self._prev_scores: Dict[Vehicle, float] = {}

    def _compute_score(self, vehicles: List[Vehicle]) -> Dict[Vehicle, float]:
        vehicle_scores = self._score_provider.compute_score(vehicles)
        self._prev_scores = {
            vehicle: self._weight * score + (1 - self._weight) * self._prev_scores.get(vehicle, 0.0) for vehicle, score in vehicle_scores.items()
        }
        return self._prev_scores

    def order(self, vehicles: List[Vehicle]) -> Iterable[Vehicle]:
        vehicles_and_scores = self._compute_score(vehicles)
        sorted_vehicles_and_scores = sorted(vehicles_and_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f'Vehicle scores: {vehicles_and_scores}\n\t -> Vehicle {sorted_vehicles_and_scores[0][0]}')
        return [x for x, _ in sorted_vehicles_and_scores]


class CyclicVehicleOrdering:
    def __init__(self, vehicles: List[Vehicle]):
        self._vehicles = deque(vehicles)

    def order(self, *args, **kwargs) -> Iterable[Vehicle]:
        self._vehicles.rotate(1)
        return iter(self._vehicles)


class PartialColumnGenerator(ColumnGenerator):
    def __init__(self, columns_per_iteration: int, vehicle_ordering: VehicleOrdering, periodic_full_iterations: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vehicle_ordering = vehicle_ordering
        self._columns_per_iteration = columns_per_iteration
        self._periodicity = periodic_full_iterations if periodic_full_iterations is not None else maxsize
        self._iters_to_next_full_iteration = 0

    def generate_columns(self, coverage_duals: Dict[Vehicle, float], capacity_duals: CapacityDual) -> Tuple[Dict[Vehicle, Column], bool]:
        """
        Generate columns with negative reduced costs according to the specified dual variables.
        @param coverage_duals:
        @type coverage_duals:
        @param capacity_duals:
        @type capacity_duals:
        @return: A dict of columns with negative reduced cost keyed by vehicle. An empty dict indicates that no columns
        could be generated.
        @rtype:
        """
        logger.debug(f'Generating columns according to coverage duals: {coverage_duals} and capacity duals: {capacity_duals}')
        generated_columns = {}
        try:
            for vehicle in self._vehicle_ordering.order(list(self.subproblems.keys())):
                subproblem = self.subproblems[vehicle]
                with Timer(factor=1000) as generation_timer:
                    column = self._generate_column(subproblem=subproblem, coverage_dual=coverage_duals[vehicle],
                                                   capacity_duals=capacity_duals)
                if column is not None:
                    logger.info(f'[Vehicle {vehicle}] generated column {column:t} in {generation_timer}ms')
                    generated_columns[vehicle] = column
                else:
                    logger.info(f'[Vehicle {vehicle}] did not find improving column in {generation_timer}ms')

                if len(generated_columns) >= self._columns_per_iteration and self._iters_to_next_full_iteration > 0:
                    break
            self._iters_to_next_full_iteration -= 1
            if self._iters_to_next_full_iteration < 0:
                self._iters_to_next_full_iteration = self._periodicity - 1
        except SubProblemInfeasible:
            return {}, True
        return {key: col for key, col in generated_columns.items() if col is not None}, len(generated_columns) == len(self.subproblems)


class ColumnGenerationSolver:
    def __init__(self, column_generator: ColumnGenerator, master_problem: MasterProblem, column_selection_strategy: ColumnSelectionStrategy):
        self._master_problem: MasterProblem = master_problem
        self._column_generator: ColumnGenerator = column_generator
        self._column_selection_strategy = column_selection_strategy

    def add_constraint(self, constraint: Constraint):
        self._master_problem.add_constraint(constraint)
        self._column_generator.add_constraint(constraint)

    @property
    def _generated_columns(self) -> Iterable[Column]:
        """
        Get the list of columns generated. This never includes dummy columns
        @return:
        @rtype:
        """
        return filter(lambda col: not col.is_dummy, self._master_problem.columns)

    def _get_duals(self):
        # Solve master problem to obtain duals
        return self._master_problem.get_coverage_duals(), self._master_problem.get_capacity_dual()

    def _generate_schedules(self) -> Dict[Vehicle, Column]:
        # Get duals
        coverage_duals, capacity_duals = self._get_duals()

        # Generate new schedules
        return self._column_generator.generate_columns(coverage_duals=coverage_duals, capacity_duals=capacity_duals)

    def solve(self) -> Generator[Tuple[float, Solution], None, Optional[Tuple[Optional[float], Dict[Column, float]]]]:
        """
        Generates (lb, Solution) pairs. Returns None when no columns with reduced cost can be generated
        @return:
        @rtype:
        """
        # Initialize MP lazily
        if not self._master_problem.has_been_solved:
            self._master_problem.solve()

        while True:
            # Generate/add new columns
            with Timer(factor=1000) as col_gen_timer:
                generated_columns, can_compute_lb = self._generate_schedules()

            selected_columns = list(self._column_selection_strategy.select_columns(
                    columns=generated_columns, current_solution=self._master_problem.solution).values())

            logger.info(f'Generated {len(generated_columns)} columns in {col_gen_timer}ms. Selected {len(selected_columns)}')
            if len(selected_columns) == 0:
                break

            for next_column in selected_columns:
                self._master_problem.add_column(column=next_column)

            # Solve mp
            self._master_problem.solve()

            # Calculate lower bound
            if can_compute_lb:
                lb = self._master_problem.objective_value + sum(col.objective for col in generated_columns.values())
            else:
                lb = -inf
            yield lb, self._master_problem.solution

        if self._master_problem.has_feasible_solution:
            return self._master_problem.objective_value, self._master_problem.solution
        return None


@dataclass
class SolveStats:
    integral_solution: Optional[Solution]
    solution: Optional[Solution]
    iterations: int

    @property
    def has_integral_solution(self) -> bool:
        return self.integral_solution is not None


class NodeSolver(Protocol):
    def solve(self, **kwargs) -> SolveStats:
        ...

    @staticmethod
    def FromParent(parentSolver: 'NodeSolver', new_constraints: Set[Constraint]) -> 'NodeSolver':
        ...


class ColumnGenerationNodeSolver:
    """
    Handles solving a node.
    Responsibilities:
        i) Propagating constraints
        ii) Managing lower and upper bounds
        iii)

    Most basic behavior:
        Construct and solve something given a set of constraints
    """

    def __init__(self, instance: DiscretizedInstance, constraints: Set[Constraint], column_generator_factory: Callable[[Dict[Vehicle, Subproblem]], PartialColumnGenerator], SubProblem: Type[Subproblem], **cli_args):
        self._instance = instance
        self._optimality_gap = cli_args.get('gap')

        master_problem = MasterProblem(self._instance, **cli_args)

        self._solver = ColumnGenerationSolver(
            column_generator=column_generator_factory({k: self._construct_with_constraints(SubProblem, constraints, self._instance, k, **cli_args) for k in self._instance.vehicles}),
            master_problem=master_problem,
            column_selection_strategy=ColumnSelectionStrategy())

    def _construct_with_constraints(self, cls, constraints: Iterable[Constraint], *args, **kwargs):
        constructed_class = cls(*args, **kwargs)
        for constraint in constraints:
            constructed_class.add_constraint(constraint)
        return constructed_class

    def solve(self, upper_bound: float = inf, max_iterations: int = maxsize, **kwargs) -> SolveStats:
        generated_lower_bound, improved_solution = None, None
        solve_stats = SolveStats(integral_solution=None, solution=None, iterations=0)
        # Generate solutions using the column generation solver until no improvement is possible
        best_lower_bound = -inf
        improved_solution = None
        for next_solve_state, _ in zip(self._solver.solve(), range(max_iterations)):
            if next_solve_state is None:
                break
            generated_lower_bound, improved_solution = next_solve_state
            solve_stats.iterations += 1

            if generated_lower_bound > best_lower_bound and is_solution_feasible(improved_solution):
                logger.info(f'Generated new lb: {generated_lower_bound} |'
                            f'Current lb: {best_lower_bound}, ub: {upper_bound} | '
                            f'Solution (integral: {is_solution_integral(improved_solution)}, '
                            f'feas: {is_solution_feasible(improved_solution)}): {solution_value(improved_solution)}')
                best_lower_bound = generated_lower_bound

            if is_solution_integral(improved_solution) and is_solution_feasible(improved_solution):
                solve_stats.integral_solution = Solution(improved_solution)
                upper_bound = solve_stats.integral_solution.obj_val
                logger.info(f'Solution is integral! New upper bound: {upper_bound}')
                assert upper_bound >= best_lower_bound

            # Early abort
            if calculate_gap(lb=best_lower_bound, ub=upper_bound) <= self._optimality_gap:
                logger.info(f'Aborting column generation due to gap.')
                break

        if improved_solution is None:
            solve_stats.solution = Solution.InfeasibleSolution()
        else:
            solve_stats.solution = Solution(improved_solution)

        logger.info(f'No further improvement possible. Aborting with best ub: {upper_bound} best lb: {best_lower_bound}'
                    f' after {solve_stats.iterations} iterations')

        vehicles = sorted({x.vehicle for x in solve_stats.solution.columns})
        for veh in vehicles:
            cols = [(x, val) for x, val in solve_stats.solution.items() if x.vehicle == veh]
            logger.debug(f"------------- Vehicle {veh} has {len(cols)} columns ------------")
            for (col, weight) in cols:
                logger.debug(f'{weight:.4f}: {col:t}')
        return solve_stats

    @property
    def _columns(self) -> Iterable[Column]:
        """
        Get the columns generated while solving the node. This does not include dummy columns.
        @return:
        @rtype:
        """
        return self._solver._generated_columns

    @staticmethod
    def FromParent(parentSolver: NodeSolver, new_constraints: Set[Constraint]) -> 'NodeSolver':
        """
        Constructs a new node solver from a parent solver.
        @param node:
        @type node:
        @param parentSolver:
        @type parentSolver:
        @return:
        @rtype:
        """
        if not isinstance(parentSolver, ColumnGenerationNodeSolver):
            raise NotImplementedError

        concrete_solver_copy = deepcopy(parentSolver._solver)
        for constraint in new_constraints:
            concrete_solver_copy.add_constraint(constraint)
        # Clone the parent solver. This should leave all found columns etc. intact
        solver = copy(parentSolver)
        solver._solver = concrete_solver_copy

        return solver
