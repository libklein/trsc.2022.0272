from . import RankedPeriodSolution
from typing import List, Generator, Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config


@dataclass_json
@dataclass
class PeriodSolutionPool:
    """

    """
    max_size: int
    size: int
    solutions: List[RankedPeriodSolution]
    period: Optional[int] = None

    def __post_init__(self):
        # TODO Duplicate removal?
        assert len(self.solutions) == self.size
        assert self.size <= self.max_size

    @property
    def routes(self) -> Generator:
        return (route for sol in self.solutions for route in sol)

    def __iter__(self):
        return iter(self.solutions)