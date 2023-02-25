# coding=utf-8
from typing import Protocol, List, Optional

from column_generation.solution import Solution
from column_generation.column import Column

from .submiping import PrimalMPHeuristic
from .diving import DivingHeuristic
from .strong_diving import StrongDivingHeuristic


class Heuristic(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def solve(self, columns: List[Column], best_ub: Optional[Solution] = None) -> Optional[Solution]:
        ...
