# coding=utf-8
from pathlib import Path
import json
from typing import Dict, Optional
from pickle import dump as write_pickle

from .legacy_model_builder import DynamicTourModel
from evrpscp import FleetChargingSchedule


def write_solution(model: DynamicTourModel, solution: Optional[FleetChargingSchedule], instance_name: str, output_directory: Path):
    model.model.export_as_mps(str(output_directory / f'solution-{instance_name}.mps'))
    model.model.export_as_sav(str(output_directory / f'solution-{instance_name}.sav'))
    if model.model.solution:
        model.model.solution.export(str(output_directory / f'cplex-solution-{instance_name}.json'))

    details = model.model.get_solve_details()

    infeasible = solution is None

    with open(str(output_directory / f'solution-{instance_name}.json'), 'w') as solution_file:
        def format_val(property):
            if infeasible:
                return 'infeasible'
            return getattr(solution, property)
        solution_repr = {
            'SolutionInfo': {
                'Runtime': details.time,
                'ObjVal': solution.cost if not infeasible else 'infeasible',
                'ObjBound': details.best_bound if not infeasible else 'infeasible',
                'MIPGap': details.mip_relative_gap if not infeasible else 'infeasible',
                'IterCount': details.nb_iterations,
                'NodeCount': details.nb_nodes_processed
            },
            'ScheduleDetails': {
                'Cost': format_val('cost'),
                'TourCost': format_val('tour_cost'),
                'EnergyCost': format_val('energy_cost'),
                'DegradationCost': format_val('degradation_cost'),
                'TotalCharge': format_val('total_charge'),
            }
        }

        solution_repr['IterPerNode'] = solution_repr['SolutionInfo']['IterCount'] / \
                                       max(solution_repr['SolutionInfo']['NodeCount'], 1)
        json.dump(solution_repr, solution_file)

    with open(str(output_directory / f'solution-{instance_name}.pickle'), 'wb') as pickle_file:
        write_pickle({
            'result': solution_repr,
            'parsed_solution': solution
        }, pickle_file)
