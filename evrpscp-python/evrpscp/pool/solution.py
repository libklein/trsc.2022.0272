from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config, CatchAll, Undefined
from typing import Optional, List

from .. import Route

@dataclass_json
@dataclass
class RankedPeriodSolution:
    cost: float
    cost_rank: int
    diversity: float
    diversity_rank: float
    fitness: float
    hash: int
    routes: List[Route] = field(metadata=config(
        decoder=lambda x: [Route.from_dict(r) for r in x['routes']],
        field_name='solution'
    ))
    #__routes: CatchAll
    #routes: Optional[List[Route]] = None

    def __post_init__(self):
        #self.routes = [Route.from_dict(x) for x in self.__routes['solution']['routes']]
        #del self.__routes
        pass

    def __iter__(self):
        return iter(self.routes)

    def discard_empty_routes(self):
        self.routes = list(filter(lambda x: x.length > 2, self.routes))
