from dataclasses import dataclass
from typing import List, Generator

from . import PeriodSolutionPool

@dataclass
class SolutionPool:
    """
    Holds routing solutions for individual periods, defining compatibility relationships between individual solutions.
        1) Efficient selection of compatible candidate solutions
        2) Reduction to efficient candidates?
        3) Sampling?
    """
    period_pools: List[PeriodSolutionPool]

    def __iter__(self):
        return iter(self.period_pools)