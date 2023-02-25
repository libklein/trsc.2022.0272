# coding=utf-8
from typing import Optional, Dict, List, Set, Tuple, Iterable
from time import time as now
from copy import deepcopy
from logging import StreamHandler
from sys import stdout, stderr

import docplex.mp.solution
from funcy import flatten

from .network_based_model_builder import DynamicTourModel
from docplex.mp.conflict_refiner import ConflictRefiner

from contexttimer import Timer

from evrpscp import *


class Solver:

    def __init__(self, instance: SchedulingInstance, **cli_args):
        self.instance: DiscretizedInstance = DiscretizedInstance.DiscretizeInstance(instance)
        self.cli_args = cli_args
        if self.instance.parameters.max_charges_between_tours < len(self.instance.periods):
            print(f"[WARNING]: Max charge between tours is set by instance ({self.instance.parameters.max_charges_between_tours}) but is currently not implemented!", file=stderr, flush=True)
        self.model = DynamicTourModel(self.instance, **cli_args)
        self.reset_parameters()

    @property
    def best_lb(self) -> Optional[float]:
        return None

    @property
    def upper_bound(self) -> Optional[float]:
        return None

    @property
    def gap(self) -> Optional[float]:
        return None

    def reset_parameters(self):
        self.model.set_parameters(**self.cli_args)

    def solve(self) -> Tuple[Optional[FleetChargingSchedule], DynamicTourModel]:
        #solution = self.model.model.read_mip_starts('/tmp/network-sol.mst')
        solution = self.model.solve()

        if solution is not None:
            print(f'Found a solution with obj value {self.model.cplex.objective_value}!\n'
                  f'Gap: {self.model.cplex.solve_details.mip_relative_gap}\n'
                  f'Status: {self.model.cplex.solve_details.status}')
            try:
                self.model.validate_solution()
            except AssertionError as e:
                print(f'Error: Could not validate solution!')
                print(f'-------------------------------------')
                from traceback import print_tb
                print_tb(e.__traceback__)
                print(f'-------------------------------------')
        else:
            print(f'Failed to find a feasible solution! Status: {self.model.cplex.solve_details.status}')
            return None, self.model

        # Construct schedule from solution
        return self._construct_schedule(), self.model

    def _construct_schedule(self) -> FleetChargingSchedule:
        return self.model.create_fleet_charging_schedule()
