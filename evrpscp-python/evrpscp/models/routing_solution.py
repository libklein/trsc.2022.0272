from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json
from . import PeriodVertex

@dataclass_json
@dataclass
class Route:
    consumption: float
    cost: float
    duration: float
    vertices: List[PeriodVertex]
    id: Optional[int] = None

    def __iter__(self):
        return iter(self.vertices)

    def __hash__(self):
        if self.id is not None:
            return hash(self.id)
        else:
            return id(self)

    def __eq__(self, other):
        if self.id is not None or other.id is not None:
            return self.id == other.id
        else:
            return id(self) == id(other)

    @property
    def length(self):
        return len(self.vertices)

    # TODO Compatible

@dataclass_json
@dataclass
class PeriodRoutingSolution:
    cost: float
    routes: List[Route]

    def __iter__(self):
        return iter(self.routes)

    def discard_empty_routes(self):
        self.routes = list(filter(lambda x: x.length > 2, self.routes))


@dataclass_json
@dataclass
class RoutingSolution:
    solutions: List[PeriodRoutingSolution]

    def __iter__(self):
        return iter(self.solutions)