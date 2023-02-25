# coding=utf-8
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from terminaltables import AsciiTable
from evrpscp import PathParser, Dump, SchedulingInstance, FleetChargingSchedule

from .output import write_solution
from .solver import Solver

def run(arguments: argparse.Namespace) -> Optional[FleetChargingSchedule]:
    dump = Dump.ParseDump(arguments.dump_directory)
    instance = dump.instance
    instance_name = dump.name

    output_directory: Path = arguments.output_dir / instance_name
    args = dict(arguments.__dict__)
    args['output_dir'] = output_directory
    args['log_dir'] = arguments.log_dir / instance_name
    args['instance_name'] = instance_name

    tours = instance.tourPlans
    print(f'Solving instance {instance_name} ({arguments.dump_directory})')

    # Print tours
    if arguments.print_tours:
        print("--------------------------Tours-----------------------------------")
        max_tours_assigned = max(map(lambda x: len(x), instance.tourPlans))
        tour_table_rows = [['Vehicle']]
        tour_table_rows[0].extend(['Tour', 'Earliest Departure', 'Latest Departure', 'Duration', 'Consumption', 'Cost'] * max_tours_assigned)
        for vehicle_id, vehicle_tours in enumerate(tours):
            tour_table_rows.append([vehicle_id])
            for tour in vehicle_tours:
                tour_table_rows[-1].extend(map(str, [tour, round(tour.earliest_departure_time, 2), round(tour.latest_departure_time, 2), round(tour.duration_time, 2), round(tour.consumption, 2), round(tour.cost, 2)]))
        print(AsciiTable(tour_table_rows).table)
        print("---------------------------------------------------------------------")

    if not output_directory.exists():
        output_directory.mkdir()

    if arguments.print_instance:
        instance.dump_to_console()

    solver = Solver(instance, **args)
    solution, model = solver.solve()

    if arguments.print_sol:
        print("--------------------------Solution--------------------------------")
        print(solution)
        print("------------------------------------------------------------------")

    write_solution(model=model, solution=solution, instance_name=instance_name, output_directory=output_directory)

    return solution

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('EVRP-SCP Dynamic Tour MIP')
    output_group = parser.add_argument_group('output', 'output related options')
    output_group.add_argument('-o', '--output-directory', type=PathParser(mode='w', type='d'), dest='output_dir', default='.')
    output_group.add_argument('-l', '--log-directory', type=PathParser(mode='w', type='d'), dest='log_dir', default='.')

    input_group = parser.add_argument_group('input', 'input file types')
    input_group.add_argument('-d', '--dump-directory', type=PathParser(mode='r', type='d', regex=r'\.dump\.d$'), dest='dump_directory', help='directory containing a solution dump', required=True)

    mip_params = parser.add_argument_group('parameters', 'solver parameters')
    mip_params.add_argument('--time-limit', type=int, dest='time_limit', help='MIP time limit')
    mip_params.add_argument('--ignore-route-costs', dest='ignore_route_costs', action='store_true', help='Do not add route cost to objective')
    mip_params.add_argument('--ignore-degradation', dest='ignore_degradation', action='store_true', help='Do not consider degradation costs in the objective function')
    mip_params.add_argument('--ignore-energy-price', dest='ignore_energy_price', action='store_true', help='Do not consider energy price in the objective function')
    mip_params.add_argument('--ignore-station-capacity', dest='ignore_station_capacity', action='store_true', help='Do not consider station capacity')

    verbosity_params = parser.add_argument_group('verbosity options', 'how much information to print')
    verbosity_params.add_argument('--print-instance', dest='print_instance', action='store_true', help='Print chargers, battery wear cost etc.')
    verbosity_params.add_argument('--print-tours', dest='print_tours', action='store_true', help='Print tours selected for routing pool')
    verbosity_params.add_argument('--print-solution', dest='print_sol', action='store_true', help='Print solution')
    verbosity_params.add_argument('--verbose', '-v', action='count', default=0,
                                  help='Sets verbosity level.')

    parser.add_argument('--threads', type=int, default=1, help='Number of threads to utilize. Pass 0 to use as many threads as are available.')

    return parser


def run_from_command_line(args: Optional[List[str]] = None) -> Tuple[Optional[FleetChargingSchedule], Dict]:
    parser: argparse.ArgumentParser = create_parser()
    arguments = parser.parse_args(args=args)
    return run(arguments)
