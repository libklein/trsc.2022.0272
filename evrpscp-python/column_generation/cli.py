# coding=utf-8
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Type, Callable

from events import Events

from evrpscp.models.instance import plot_fleet_tours
from terminaltables import AsciiTable
from evrpscp import PathParser, Dump, SchedulingInstance, FleetChargingSchedule, DiscretizedInstance, Vehicle
from column_generation.debug_mode import set_extensive_checks

from column_generation.util import BranchAndBoundTreePlotter
from column_generation.heuristics import PrimalMPHeuristic, StrongDivingHeuristic, DivingHeuristic
from column_generation.node_queue import TwoStageQueue, FIFONodeQueue, LIFONodeQueue, BestBoundQueue, NodeQueue
from . import Node, CPPSubproblem
from .node_solver import NodeSolver, ColumnGenerationNodeSolver, Subproblem, PartialColumnGenerator, \
    CyclicVehicleOrdering

from .output import write_solutions
from .solution import Solution
from .solver import Solver, DisableHeuristicStrategy, UntilIntegralHeuristicStrategy, SequentialHeuristicStrategy, \
    PeriodicHeuristicStrategy
from .util.column_conflict_graph import draw_column_conflict_graph
from .util.log import configure as configure_logging

PRIMAL_HEURISTIC_FACTORIES = {
    'submip': PrimalMPHeuristic,
    'diving': DivingHeuristic,
    'strong-diving': StrongDivingHeuristic,
    'none': DisableHeuristicStrategy
}

NODE_QUEUE_FACTORIES = {
    'FIFO': FIFONodeQueue,
    'LIFO': LIFONodeQueue,
    'best-bound': BestBoundQueue,
    'two-stage': TwoStageQueue
}

SUBPROBLEM_MAP = {
    'frvcp': CPPSubproblem
}


def resolve_strategy(options: dict, selection: str):
    if (factory := options.get(selection, None)) is None:
        raise ValueError(f'Unknown choice {selection}')
    else:
        return factory

def node_solver_factory_factory(partial_col_gen_config: str, SubProblemType: Type, pp_vehicle_count: int, **cli_args):
    def _column_generator_factory(subproblems: Dict[Vehicle, Subproblem]):
        return PartialColumnGenerator(
            subproblems=subproblems,
            columns_per_iteration=len(subproblems) if partial_col_gen_config == 'full' else pp_vehicle_count,
            vehicle_ordering=CyclicVehicleOrdering(vehicles=list(subproblems.keys())))
    def _create_node_solver(instance: DiscretizedInstance, node: Node, get_node_solver: Callable[[Node], NodeSolver]) -> NodeSolver:
        if node.is_root_node:
            return ColumnGenerationNodeSolver(instance=instance, constraints=node.global_constraints,
                                              column_generator_factory=_column_generator_factory, SubProblem=SubProblemType, **cli_args)

        return ColumnGenerationNodeSolver.FromParent(parentSolver=get_node_solver(node.parent),
                                                     new_constraints=node.local_constraints)

    return _create_node_solver

def run(arguments: argparse.Namespace) -> Tuple[Optional[FleetChargingSchedule], Dict]:
    dump = Dump.ParseDump(arguments.dump_directory)
    instance = DiscretizedInstance.DiscretizeInstance(dump.instance, period_duration=30.0)
    instance_name = dump.name
    return run_instance(instance=instance, instance_name=instance_name, **arguments.__dict__)

def run_instance(instance: DiscretizedInstance, instance_name:str, **args) -> Tuple[Optional[FleetChargingSchedule], Dict]:
    output_directory: Path = args['output_dir']
    args['output_dir'] = output_directory
    if not args['dont_create_subdirectory']:
        output_directory: Path = output_directory / instance_name
        args['output_dir'] = output_directory

    args['instance_name'] = instance_name
    # Set up logger
    configure_logging(args['logger_config_file'])

    print(f'Solving instance {instance_name}')

    # Print tours
    if args['print_tours']:
        print("--------------------------Tours-----------------------------------")
        print(plot_fleet_tours(instance.tours))
        print("---------------------------------------------------------------------")

    if not output_directory.exists():
        output_directory.mkdir()

    if args['print_instance']:
        instance.dump_to_console()

    if not args['primal_heuristic'] == 'none':
        primal_heuristic_type = resolve_strategy(PRIMAL_HEURISTIC_FACTORIES, args['primal_heuristic'])
        heuristic_strategy = SequentialHeuristicStrategy(UntilIntegralHeuristicStrategy(solver_type=primal_heuristic_type, instance=instance), PeriodicHeuristicStrategy(solver_type=primal_heuristic_type, interval=len(instance.vehicles), instance=instance))
    else:
        heuristic_strategy = DisableHeuristicStrategy()

    solver = Solver(instance,
                    heuristic_strategy=heuristic_strategy,
                    node_queue_factory=resolve_strategy(NODE_QUEUE_FACTORIES, args['node_selection_strategy']),
                    node_solver_factory=node_solver_factory_factory(args['partial_pricing_strategy'], SUBPROBLEM_MAP[args['subproblem_type']], pp_vehicle_count=min(x.capacity for x in instance.chargers), **args),
                    **args)

    # Register callback
    if args['draw_conflict_graph']:
        def render_graph(node: Node, sol: Solution):
            if (g := draw_column_conflict_graph(sol, vehicles=instance.vehicles, periods=instance.periods, chargers=instance.chargers)) is not None:
                g.render('/tmp/conflicts')
        solver.solver_events.on_node_solved += render_graph

    solution, details = solver.solve(**args)

    for col in sorted(solver.upper_bound, key=lambda col: col.vehicle):
        print(f'{col.vehicle}: {col:t}')

    outfile = (output_directory / 'sol.txt').open('w')
    for f in instance.chargers:
        print(f"Charger {f} ({f.capacity})", file=outfile)
        rows = [["Vehicle", *(f'{p.begin}-{p.end}' for p in instance.periods)]]
        for col in sorted(solver.upper_bound, key=lambda col: col.vehicle):
            rows.append([str(col.vehicle)]+[('x' if col.charger_usage[p, f] else '') for p in instance.periods])
        rows.append(["Sum", *(sum(int(col.charger_usage[p, f]) for col in solver.upper_bound) for p in instance.periods)])
        print(AsciiTable(rows).table, file=outfile)

    write_solutions(solution, details, instance_name, output_directory, root_node=solver.root_node)

    if args['dump_bnb_tree']:
        with open(output_directory / 'bnb-tree.pickle', 'wb') as bnb_dump_output:
            pickle.dump(solver.root_node, bnb_dump_output)

    if args['plot_bnb_tree']:
        tree_plotter = BranchAndBoundTreePlotter()
        tree_plotter(solver.root_node)
        tree_plotter.plot(view=False, filename="bnb-tree.gv", directory=output_directory)

    return solution, details


def register_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(run=run)
    output_grp = parser.add_argument_group('output options', 'output options')
    output_grp.add_argument('--plot-bnb-tree', dest='plot_bnb_tree', action='store_true')
    output_grp.add_argument('--dump-bnb-tree', dest='dump_bnb_tree', action='store_true')

    solver_params = parser.add_argument_group('solving options', 'solving options')

    solver_params.add_argument('--dump-networks', dest='dump_networks', help='Create dump for each network solved',
                               action='store_true')
    solver_params.add_argument('--log-duals', dest='log_duals', action='store_true')
    solver_params.add_argument('--use-barrier', dest='use_barrier', action='store_true')
    solver_params.add_argument('--primal-heuristic', dest='primal_heuristic',
                               choices=('submip', 'strong-diving', 'diving', 'none'), default='strong-diving')
    solver_params.add_argument('--node-selection-strategy', dest='node_selection_strategy',
                               choices=('FIFO', 'LIFO', 'best-bound', 'two-stage'), default='two-stage')
    solver_params.add_argument('--partial-pricing-strategy', dest='partial_pricing_strategy',
                               choices=('full', 'single'), default='single')
    solver_params.add_argument('--subproblem-type', dest='subproblem_type',
                               choices=('frvcp'), default='frvcp')
    solver_params.add_argument('--draw-conflict-graph', dest='draw_conflict_graph', action='store_true')

    return parser


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('EVRP-SCP MIP')
    parser.add_argument('--extensive-checks', help='enabled additional checks', dest='extensive_checks',
                        action='store_true')
    parser.add_argument('--profile-filename', help='create a profile and write it to disk', dest='profile_file',
                        type=PathParser(mode='w', type='f'))
    output_group = parser.add_argument_group('output', 'output related options')
    output_group.add_argument('-o', '--output-directory', type=PathParser(mode='w', type='d'), dest='output_dir',
                              default='.')
    output_group.add_argument('-l', '--logger-config', type=PathParser(mode='r', type='f'), dest='logger_config_file',
                              default='./logging.yaml')
    output_group.add_argument('--dont-create-subdirectory', action='store_true',
                              help='Dont create instance name subdirectories in logs and output dirs')

    input_group = parser.add_argument_group('input', 'input file types')
    input_group.add_argument('-d', '--dump-directory', type=PathParser(mode='r', type='d', regex=r'\.dump\.d$'),
                             dest='dump_directory', help='directory containing a solution dump', required=True)

    mip_params = parser.add_argument_group('parameters', 'solver parameters')
    mip_params.add_argument('--time-limit', type=int, dest='time_limit', help='MIP time limit')
    mip_params.add_argument('--gap', type=float, dest='gap', help='Relative Gap', default=1e-04)
    mip_params.add_argument('--ignore-route-costs', dest='ignore_route_costs', action='store_true',
                            help='Do not add route cost to objective')
    mip_params.add_argument('--ignore-degradation', dest='ignore_degradation', action='store_true',
                            help='Do not consider degradation costs in the objective function')
    mip_params.add_argument('--ignore-energy-price', dest='ignore_energy_price', action='store_true',
                            help='Do not consider energy price in the objective function')
    mip_params.add_argument('--ignore-station-capacity', dest='ignore_station_capacity', action='store_true',
                            help='Do not consider station capacity')

    verbosity_params = parser.add_argument_group('verbosity options', 'how much information to print')
    verbosity_params.add_argument('--print-instance', dest='print_instance', action='store_true',
                                  help='Print chargers, battery wear cost etc.')
    verbosity_params.add_argument('--print-tours', dest='print_tours', action='store_true',
                                  help='Print tours selected for routing pool')
    verbosity_params.add_argument('--print-solution', dest='print_sol', action='store_true', help='Print solution')
    verbosity_params.add_argument('--verbose', '-v', action='count', default=0,
                                  help='Sets verbosity level.')

    subparsers = parser.add_subparsers()

    col_gen_parser = register_parser(subparsers.add_parser('col-gen'))
    return parser


def run_from_command_line(args: Optional[List[str]] = None) -> Tuple[Optional[FleetChargingSchedule], Dict]:
    parser: argparse.ArgumentParser = create_parser()
    arguments = parser.parse_args(args=args)
    set_extensive_checks(arguments.extensive_checks)
    if arguments.profile_file is None:
        return arguments.run(arguments)
    else:
        print('Profiling application!')
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        retval = arguments.run(arguments)
        profiler.create_stats()
        print(f'Dumping profile to {arguments.profile_file.resolve()}!')
        profiler.dump_stats(str(arguments.profile_file.resolve()))
        return retval
