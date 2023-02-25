# coding=utf-8
import logging
from itertools import product
from typing import Optional, Dict, List, Set, Tuple, Iterable, Protocol, Type, Callable
from heapq import *
from time import time as now
from copy import deepcopy
from logging import StreamHandler
from sys import stdout, stderr
from math import inf, isinf

from events import Events
from funcy import partial

from contexttimer import Timer

from column_generation.util import setup_default_logger, signal_based_timeout
from column_generation.debug_mode import *
from column_generation.branching import PeriodBranchingRule
from column_generation.constraints import Constraint, BlockChargerConstraint
from column_generation.util import TimedOut, calculate_gap
from column_generation.node_queue import NodeQueue, TwoStageQueue
from column_generation.heuristics import PrimalMPHeuristic, StrongDivingHeuristic, DivingHeuristic, Heuristic
from .output import SolveDetails

from .solution import Solution
from .node_solver import NodeSolver, SolveStats, ColumnGenerationNodeSolver
from .node import Node
from .discretization import DiscretizedInstance
from .column import Column
from .parameters import EPS
from evrpscp import *

logger = setup_default_logger(__name__)


class TerminationCriteriaMet(Exception):
    pass


class HeuristicStrategy(Protocol):
    def should_run_heuristic(self, solution: Solution, lower_bound: float, upper_bound: Optional[Solution]) -> Optional[
        Heuristic]:
        ...


class DisableHeuristicStrategy:
    def should_run_heuristic(self, solution: Solution, lower_bound: float, upper_bound: Optional[Solution]) -> Optional[
        Heuristic]:
        return None


class UntilIntegralHeuristicStrategy:
    def __init__(self, solver_type: Type[Heuristic], *args, **kwargs):
        self._solver_type = solver_type
        self._solver_construction_args = args
        self._solver_construction_kwargs = kwargs

    def should_run_heuristic(self, solution: Solution, lower_bound: float,
                             upper_bound: Optional[Solution]) -> Heuristic:
        if upper_bound is None:
            return self._solver_type(*self._solver_construction_args, **self._solver_construction_kwargs)
        return None


class PeriodicHeuristicStrategy:
    def __init__(self, solver_type: Type[Heuristic], interval: int = 10, *args, **kwargs):
        self._solver_type = solver_type
        self._solver_construction_args = args
        self._solver_construction_kwargs = kwargs
        self._interval = interval
        self._iters_to_next_solve = self._interval

    def should_run_heuristic(self, solution: Solution, lower_bound: float, upper_bound: Optional[Solution]) -> Optional[
        Heuristic]:
        if self._iters_to_next_solve <= 1:
            self._iters_to_next_solve = self._interval
            return self._solver_type(*self._solver_construction_args, **self._solver_construction_kwargs)
        self._iters_to_next_solve -= 1
        return None


class SequentialHeuristicStrategy:
    def __init__(self, *strategies: HeuristicStrategy):
        self._strategies = strategies

    def should_run_heuristic(self, *args, **kwargs) -> Optional[Heuristic]:
        for s in self._strategies:
            if (solver := s.should_run_heuristic(*args, **kwargs)) is not None:
                return solver
        return None


class Solver:

    def __init__(self, instance: DiscretizedInstance, heuristic_strategy: HeuristicStrategy,
                 node_queue_factory: Callable[[Events], NodeQueue],
                 node_solver_factory: Callable[[DiscretizedInstance, Node, Callable[[Node], NodeSolver]], NodeSolver],
                 **cli_args):
        self.instance = instance
        self._heuristic_strategy = heuristic_strategy
        self.cli_args = cli_args
        self._min_gap = self.cli_args.get('gap', 1e-04)

        if extensive_checks():
            logger.warning('Performing extensive checks!')

        self.solver_events = Events(
            ('on_integral_solution_found', 'on_lower_bound_improved', 'on_node_solved',
             'on_node_pruned', 'on_node_created', 'on_node_infeasible'))

        self._root_node = Node(None, set())

        self._solvers: Dict[Node, NodeSolver] = dict()
        self._branching_rule = PeriodBranchingRule(self.instance)
        self.upper_bound: Optional[Solution] = None
        self.lower_bound: float = -inf
        self.num_nodes_solved: int = 0
        self._node_queue: NodeQueue = node_queue_factory(self.solver_events)
        self._node_queue.enqueue(self.root_node)
        self._node_solver_factory = node_solver_factory

    @property
    def upper_bound_obj(self) -> Optional[float]:
        return self.upper_bound.obj_val if self.upper_bound is not None else None

    def _solve_heuristically(self, heuristic_solver: Heuristic, column_list: List[Column]) -> Optional[Solution]:
        assert heuristic_solver is not None and len(column_list) > 0
        sol: Optional[Solution] = heuristic_solver.solve(columns=column_list)
        return sol

    @property
    def nodes(self):
        return self._solvers.keys()

    def _create_node_solver(self, node: Node) -> NodeSolver:
        return self._node_solver_factory(self.instance, node, lambda node: self._get_node_solver(node))

    def _get_node_solver(self, node: Node) -> NodeSolver:
        if node not in self._solvers:
            self._solvers[node] = solver = self._create_node_solver(node)
        else:
            solver = self._solvers[node]
        return solver

    def _solve_node(self, node: Node) -> SolveStats:
        """
        Solves and updates the node accordingly.
        :param node: The node to solve
        :return:
        """
        stats = self._get_node_solver(node).solve(upper_bound=self.upper_bound_obj)
        if stats.solution is not None and stats.solution.feasible:
            node.lower_bound = stats.solution.obj_val
        return stats

    def _add_node(self, node: Node) -> Node:
        self._node_queue.enqueue(node)
        # Fire node created event
        self.solver_events.on_node_created(node)
        return node

    @property
    def gap(self) -> float:
        return calculate_gap(self.lower_bound, self.upper_bound_obj)

    @property
    def num_open_nodes(self) -> int:
        return len(self._node_queue)

    @property
    def open_nodes(self) -> Iterable[Node]:
        return self._node_queue

    @property
    def root_node(self) -> Node:
        return self._root_node

    @property
    def has_integral_solution(self) -> bool:
        return self.upper_bound is not None

    @property
    def should_terminate(self) -> bool:
        return self.gap <= self._min_gap

    def on_integral_solution_found(self, solution: Solution):
        assert solution.feasible and solution.integral
        if self._update_ub(solution):
            logger.info(f'Found new UB {self.upper_bound}. Gap: {self.gap}')
            logger.info(self.upper_bound)

        self.solver_events.on_integral_solution_found(solution)

        if self.should_terminate:
            raise TerminationCriteriaMet

    def _update_lower_bound(self):
        if self.num_open_nodes == 0:
            self.lower_bound = max(node.lower_bound for node in self.nodes if node.is_leaf_node)
            # if self.upper_bound_obj is not None:
            #    self.lower_bound = self.upper_bound_obj
            return
        # Search tree
        lb = self.lower_bound
        self.lower_bound = min(node.lower_bound for node in self.open_nodes)

        if self.upper_bound_obj is not None:
            self.lower_bound = min(self.upper_bound_obj, self.lower_bound)

        if lb < self.lower_bound:
            logger.info(f'Improved LB from {lb} to {self.lower_bound}')

        if self.should_terminate:
            raise TerminationCriteriaMet

    def _update_ub(self, integral_solution: Solution) -> bool:
        if self.upper_bound is None or self.upper_bound.obj_val > integral_solution.obj_val:
            self.upper_bound = integral_solution
            return True
        return False

    def on_node_pruned(self, node: Node):
        if node is not None:
            logger.info(f'Pruning node {node}.')

    def on_node_infeasible(self, node: Node):
        pass

    def _extract_node(self) -> Optional[Node]:
        while self._node_queue.has_next_node():
            # Remove the node and return if valid
            node = self._node_queue.pop()
            if self.has_integral_solution and node.lower_bound + EPS >= self.upper_bound_obj:
                logger.info(
                    f'Skipping node {node} as it\'s lb  {node.lower_bound} is >= the current ub {self.upper_bound_obj}')
            else:
                return node
        return None

    def _can_prune(self, node: Node) -> bool:
        return self.upper_bound_obj is not None and node.lower_bound >= self.upper_bound_obj

    def _process_solved_node(self, node: Node, solve_stats: SolveStats):
        # Process it
        if solve_stats.has_integral_solution:
            self.on_integral_solution_found(solve_stats.integral_solution)
        self.num_nodes_solved += 1
        logger.info(
            f'Found solution: {node.lower_bound} (Feasible: {solve_stats.solution.feasible}, Integral: {solve_stats.solution.integral})')

        if solve_stats.solution is None:
            raise ValueError("Solver returned no solution!")

        if solve_stats.solution.infeasible:
            self.solver_events.on_node_infeasible(node)
            return
        elif solve_stats.solution == solve_stats.integral_solution:
            # Node solved to optimality
            return

        # Node is feasible
        if self._can_prune(node):
            self.on_node_pruned(node)
            return

        # Try to solve the node heuristically.
        if (heuristic_solver := self._heuristic_strategy.should_run_heuristic(solution=solve_stats.solution,
                                                                              lower_bound=self.lower_bound,
                                                                              upper_bound=self.upper_bound)) is not None:
            with Timer(factor=1000) as primal_timer:
                logger.info(f'Trying to solve node {node} heuristically')
                heuristic_solution = self._solve_heuristically(heuristic_solver=heuristic_solver,
                                                               column_list=list(self._get_node_solver(node)._columns))
            logger.info(f'Heuristic solve took {primal_timer} ms')

            if heuristic_solution is not None:
                self.on_integral_solution_found(heuristic_solution)
                # No need to branch when we can consider the heuristic solution optimal for the given node
                if calculate_gap(lb=node.lower_bound, ub=heuristic_solution.obj_val) <= self._min_gap:
                    logger.debug(
                        f'Not branching on node {node} since it\'s lower bound {node.lower_bound} is already cose to the heuristic solution {heuristic_solution.obj_val}')
                    return

        # Create child nodes. We do not add the parent constraints here as they are already integrated
        constraint_sets: List[Set[Constraint]] = self._derive_cuts(solution=solve_stats.solution)
        for constraints in constraint_sets:
            next_child = Node(node, constraints)
            node.add_child(next_child)
            self._add_node(next_child)

    def _solve(self) -> Optional[Solution]:
        assert self._root_node.is_leaf_node
        begin_time = now()
        # Process node queue until no node remains
        while (node := self._extract_node()) is not None:
            logger.info(
                f'Solving node: {node}. Open nodes: {self.num_open_nodes + 1}/{self.num_open_nodes + 1 + self.num_nodes_solved}')
            assert node.is_leaf_node
            # Solve the node
            solve_stats = self._solve_node(node)
            self._process_solved_node(node, solve_stats)
            # Finally update the lower bound
            self._update_lower_bound()
            self.solver_events.on_node_solved(node, solve_stats.solution)
            logger.info('-----------------------------------------')
            if now() - begin_time > self.cli_args.get('time_limit', 3600):
                raise TimedOut

        return self.upper_bound

    def solve(self, **_cli_args) -> Tuple[Optional[FleetChargingSchedule], SolveDetails]:
        cli_args = dict(**self.cli_args)
        cli_args.update(**_cli_args)

        self.num_nodes_solved = 0
        with Timer() as solver_time:
            time_limit_secs = cli_args.get('time_limit')
            try:
                signal_based_timeout(self._solve, (), {}, time_limit_secs)
            except TimedOut:
                logger.warning(f'Timeout after {solver_time}s!')
            except TerminationCriteriaMet:
                pass
        logger.info('----------------------------------------------')
        logger.info('----------------------------------------------')
        logger.info(f'Done solving! Nodes solved: {self.num_nodes_solved} in {solver_time}s')

        details = SolveDetails(**{
            'Runtime': solver_time.elapsed,
            'ObjVal': 'infeasible',
            'ObjBound': 'infeasible',
            'MIPGap': 'infeasible',
            'RootLB': self.root_node.lower_bound,
            'IterCount': self.num_nodes_solved,
            'NodeCount': self.num_nodes_solved + len(self._node_queue)
        })

        if not self.upper_bound:
            logger.info(f'Failed to find a feasible, integral solution! Best LB: {self.lower_bound}')
            return None, details

        logger.info(f'Best Lower Bound: {self.lower_bound}, '
                    f'upper bound: {self.upper_bound_obj}, '
                    f'Gap: {self.gap * 100:.6f}%\n'
                    f'Found feasible & integral solution: \n'
                    f'{self.upper_bound}')

        solution = self._construct_schedule(self.upper_bound)

        details.ObjVal = self.upper_bound_obj
        details.ObjBound = self.lower_bound
        details.MIPGap = self.gap

        return solution, details

    def _derive_cuts(self, solution: Solution) -> List[Set[Constraint]]:
        return self._branching_rule.create_branches(solution=solution)

    def _construct_schedule(self, remote_columns: Iterable[Column]) -> FleetChargingSchedule:
        assert len(list(remote_columns)) == len(self.instance.vehicles), \
            f'Solution has {len(list(remote_columns))} active columns but instance has {len(self.instance.vehicles)} vehicles'

        # Columns could come from another process, i.e. hashes may not correspond to instance.
        # Fix this, i.e. create new column with periods of self.instance
        # columns = []
        # for remote_col in remote_columns:
        #    columns.append(remote_column_to_local_column(remote_col, self.instance))
        columns = remote_columns

        schedules = [col.create_vehicle_schedule(self.instance) for col in columns]
        schedules.sort(key=lambda x: x.vehicleID)
        fleet_schedule = FleetChargingSchedule(vehicleSchedules=schedules)
        fleet_schedule.calculate_cost(self.instance.parameters.battery)
        return fleet_schedule
