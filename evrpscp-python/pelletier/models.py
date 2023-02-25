# coding=utf-8
from dataclasses import InitVar
from typing import List, Iterable, Dict, Union, ClassVar, Tuple

from evrpscp import DiscretePeriod
from evrpscp.models import PiecewiseLinearFunction
from pydantic.dataclasses import dataclass
from pydantic import root_validator, validator

def check_positive(value: Union[float, int]) -> Union[float, int]:
    assert value >= 0.0
    return value

def check_between_factory(lb: float, ub: float):
    def _validator(value):
        assert lb <= value <= ub
        return value

    return _validator

check_soc = check_between_factory(0.0, 1.0)

@dataclass(frozen=True)
class Segment:
    time_lb: float
    time_ub: float
    soc_lb: float
    soc_ub: float
    current: float

    _check_positive: ClassVar = validator('time_lb', 'time_ub', 'soc_lb', 'soc_ub', 'current', allow_reuse=True)(check_positive)

    @root_validator
    def check_time_interval_valid(cls, values):
        assert values['time_lb'] <= values['time_ub']
        return values

    @root_validator
    def check_soc_interval_valid(cls, values):
        assert values['soc_lb'] <= values['soc_ub']
        return values

    @property
    def time_range(self) -> Tuple[float, float]:
        return self.time_lb, self.time_ub

    @property
    def soc_range(self) -> Tuple[float, float]:
        return self.soc_lb, self.soc_ub

@dataclass(frozen=True)
class WDFSegment:
    soc_lb: float
    soc_ub: float
    cost_lb: float
    cost_ub: float
    cost_per_soc: float

    _check_positive: ClassVar = validator('soc_lb', 'soc_ub', 'cost_lb', 'cost_ub', 'cost_per_soc', allow_reuse=True)(check_positive)

    @root_validator
    def check_cost_interval_valid(cls, values):
        assert values['cost_lb'] <= values['cost_ub']
        return values

    @root_validator
    def check_soc_interval_valid(cls, values):
        assert values['soc_lb'] <= values['soc_ub']
        return values

    @property
    def cost_range(self) -> float:
        return self.cost_ub - self.cost_lb

    @property
    def soc_range(self) -> float:
        return self.soc_ub - self.soc_lb

@dataclass(frozen=True)
class Charger:
    id: int
    phi: PiecewiseLinearFunction
    capacity: int
    battery_capacity_amps: InitVar[float]

    _check_positive: ClassVar = validator('capacity', allow_reuse=True)(check_positive)

    @validator('phi')
    def check_phi_bounds(cls, phi: PiecewiseLinearFunction):
        assert phi.image_lower_bound == phi.lower_bound == 0
        assert phi.image_upper_bound == 1.0
        return phi

    @validator('phi')
    def check_phi_concave(cls, phi: PiecewiseLinearFunction):
        assert phi.is_concave()
        return phi


    def __post_init__(self, battery_capacity_amps: float):
        object.__setattr__(self, '_segments', [
            Segment(bp.lowerBound, bp.upperBound, bp.imageLowerBound, bp.imageUpperBound,
                    current=bp.slope*battery_capacity_amps) for bp in self.phi.segments[1:-1]
        ])

    def __hash__(self):
        return hash(self.id)

    @property
    def segments(self) -> List[Segment]:
        return self._segments

    def __str__(self):
        return f'C{self.id}'
@dataclass(frozen=True)
class Battery:
    wdf: PiecewiseLinearFunction
    battery_capacity_ah: float
    battery_capacity_kwh: float
    min_soc: float
    max_soc: float
    initial_soc: float

    _check_positive: ClassVar = validator('battery_capacity_ah', 'battery_capacity_kwh', allow_reuse=True)(check_positive)
    _check_soc: ClassVar = validator('min_soc', 'max_soc', 'initial_soc', allow_reuse=True)(check_soc)

    def __post_init__(self):
        object.__setattr__(self, '_wdf_segments', [
            WDFSegment(bp.lowerBound, bp.upperBound, bp.imageLowerBound, bp.imageUpperBound, bp.slope) for bp in self.wdf.segments[1:-1]
        ])

    @property
    def wdf_segments(self) -> List[WDFSegment]:
        return self._wdf_segments

    @validator('wdf')
    def check_wdf(cls, wdf: PiecewiseLinearFunction):
        assert wdf.lower_bound == wdf.image_lower_bound == 0.0, "WDF does not start at (0.0, 0.0)"
        assert wdf.upper_bound == 1.0, "WDF does not reach battery capacity"
        # Is convex
        assert wdf.is_convex(), "WDF is not convex"
        return wdf
@dataclass(frozen=True)
class Route:
    id: int
    arrival_period: DiscretePeriod
    departure_period: DiscretePeriod
    soc_consumption: float
    vehicle: int

    _check_soc: ClassVar = validator('soc_consumption', allow_reuse=True)(check_soc)

    @root_validator
    def check_times(cls, values):
        assert values['arrival_period'] > values['departure_period']
        return values

    @property
    def covered_periods(self) -> Iterable[DiscretePeriod]:
        next_en_route_period = self.departure_period
        while next_en_route_period is not self.arrival_period:
            yield next_en_route_period
            next_en_route_period = next_en_route_period.succ
        yield self.arrival_period

    def __str__(self):
        return f'R{self.id}'

@dataclass
class Instance:
    periods: List[DiscretePeriod]
    routes: Dict[int, List[Route]]
    chargers: List[Charger]
    battery: Battery
    max_number_of_charges: int

    # Sorted periods, non-overlapping
    @validator('periods')
    def check_periods_sorted(cls, periods: List[DiscretePeriod]):
        assert all(prev_p.end == next_p.begin and prev_p.succ is next_p and next_p.pred is prev_p
                   for prev_p, next_p in zip(periods, periods[1:])), "Periods are not sorted/connected"
        assert periods[0].pred is None, "Predecessor of first period ís not None"
        assert periods[-1].succ is None, "Successor of last period is not None"
        return periods

    # Sorted routes, non-overlapping
    @validator('routes')
    def check_sorted_routes(cls, routes: Dict[int, List[Route]]):
        assert all((prev_r.arrival_period.begin < next_r.departure_period.begin for prev_r, next_r in zip(veh_routes, veh_routes[1:])) for k, veh_routes in routes.items()), "Routes are not sorted"
        return routes

    # Route periods set correctly
    @root_validator(skip_on_failure=True)
    def check_routes_use_existing_periods(cls, values):
        periods: List[DiscretePeriod] = values['periods']
        routes: Dict[int, List[Route]] = values['routes']
        for veh, veh_routes in routes.items():
            for route in veh_routes:
                assert route.arrival_period in periods and route.departure_period in periods and route.vehicle == veh
        return values

    @property
    def vehicles(self) -> List[int]:
        return sorted(self.routes.keys())

    @property
    def vehicle_routes(self) -> Dict[int, List[Route]]:
        return self.routes

    def periods_between(self, start: DiscretePeriod, end: DiscretePeriod, include_end=True) -> Iterable[DiscretePeriod]:
        while start is not end:
            yield start
            start = start.succ
        if include_end:
            yield end