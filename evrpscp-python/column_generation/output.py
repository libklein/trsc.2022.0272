# coding=utf-8
from dataclasses import dataclass, fields
from pathlib import Path
import json
from typing import Optional
from pickle import dump as write_pickle

from evrpscp import FleetChargingSchedule

@dataclass
class SolveDetails:
    Runtime: float
    ObjVal: float
    ObjBound: float
    RootLB: float
    MIPGap: float
    IterCount: int
    NodeCount: int

def write_solutions(solution: Optional[FleetChargingSchedule], details: SolveDetails,
                    instance_name: str, output_directory: Path, root_node = None):
    with open(str(output_directory / f'solution-{instance_name}.json'), 'w') as solution_file:
        def format_val(property):
            if solution is None:
                return 'infeasible'
            return getattr(solution, property)
        solution_repr = {
            'SolutionInfo': {field.name: getattr(details, field.name) for field in fields(details)},
            'IterPerNode': details.IterCount / max(details.NodeCount, 1),
            'ScheduleDetails': {
                'Cost': format_val('cost'),
                'TourCost': format_val('tour_cost'),
                'EnergyCost': format_val('energy_cost'),
                'DegradationCost': format_val('degradation_cost'),
                'TotalCharge': format_val('total_charge'),
            }
        }
        json.dump(solution_repr, solution_file)

    with open(str(output_directory / f'solution-{instance_name}.pickle'), 'wb') as pickle_file:
        write_pickle({
            'result': solution_repr,
            'parsed_solution': solution,
            'bnb_tree': root_node
        }, pickle_file)
