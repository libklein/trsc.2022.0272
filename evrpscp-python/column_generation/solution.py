from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from funcy import group_by

from column_generation import Column
from column_generation.util import is_solution_feasible, is_solution_integral, solution_value


@dataclass
class Solution:
    columns: Dict[Column, float]

    def __iter__(self) -> Iterable[Column]:
        return iter(self.columns.keys())

    def items(self) -> Iterable[Tuple[Column, float]]:
        return self.columns.items()

    @property
    def infeasible(self) -> bool:
        return not self.feasible

    @property
    def feasible(self) -> bool:
        return is_solution_feasible(self.columns)

    @property
    def integral(self) -> bool:
        return is_solution_integral(self.columns)

    @property
    def obj_val(self) -> float:
        return solution_value(self.columns)

    def __str__(self) -> str:
        return f'Solution(obj={self.obj_val}, feasible={self.feasible}, integral:{self.integral})'

    def __repr__(self) -> str:
        resp = f'Solution (obj={self.obj_val}, feasible={self.feasible}, integral:{self.integral}) with columns:\n'
        for vehicle, columns in group_by(lambda x: x[0].vehicle, self.columns.items()).items():
            resp += f'Vehicle {vehicle}:'
            for col, weight in columns:
                resp += f'{weight}: {repr(col)}'
        return resp

    @staticmethod
    def InfeasibleSolution() -> 'Solution':
        return Solution(dict())