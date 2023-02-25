# coding=utf-8
from dataclasses import dataclass
from random import Random
from typing import Union

from dataclasses_json import dataclass_json

def inclusive_range(start: Union[int, float], end: Union[int, float], step: Union[int, float] = 1):
    _next = start
    while _next <= end:
        yield _next
        _next += step

class InstanceParameter:
    def __init__(self, default, min, max, step, name='-', nice_name=''):
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.name = name
        if nice_name == '':
            self.nice_name = name
        else:
            self.nice_name = nice_name

    def __repr__(self):
        return f'{self.name}: {self.min}-{self.max}, {self.step} ({self.default})'

    def __str__(self):
        return self.nice_name

    def __iter__(self):
        return inclusive_range(self.min, self.max, self.step)

@dataclass_json
@dataclass(unsafe_hash=True)
class InstanceParameters:
    seed: str
    fleet_size: int
    number_of_days: int
    time_window_length: int
    # Average tour duration
    average_tour_length: int
    # Average time at the depot before servicing a tour
    #average_time_at_depot_before_tour: int
    number_of_charger_types: int
    charger_complexity: int
    base_charger_capacity: int
    wdf_complexity: int
    consumption: float = 0.65
    nice_name: str = ''
    infix: str = ''
    scale_charger_capacity: bool = False

    def __post_init__(self):
        self.BATTERY_RAND_GEN = Random(self.seed)
        self.ENERGY_PRICE_RAND_GEN = Random(self.seed)
        self.INFRASTRUCTURE_RAND_GEN = Random(self.seed)
        self.TOUR_RAND_GEN = Random(self.seed)
        _vehicle_seed_generator = Random(self.seed)
        self.VEHICLE_RAND_GEN = [Random(_vehicle_seed_generator.random()) for _ in range(self.fleet_size)]

    @property
    def instancename(self) -> str:
        if self.infix != '':
            return f'exp1-{self.infix}-{self.seed}-{self.fleet_size}v-{self.number_of_days}d-{self.time_window_length}tw-{self.number_of_charger_types}c-{int(self.scale_charger_capacity)}sc-{self.charger_complexity}co-{self.base_charger_capacity}cc-{self.wdf_complexity}wc'
        else:
            return f'exp1-{self.seed}-{self.fleet_size}v-{self.number_of_days}d-{self.time_window_length}tw-{self.number_of_charger_types}c-{int(self.scale_charger_capacity)}sc-{self.charger_complexity}co-{self.base_charger_capacity}cc-{self.wdf_complexity}wc'

# Common

PERIOD_LENGTH_MINUTES = 30
PERIODS_PER_HOUR = 2
PERIODS_PER_DAY = 24 * PERIODS_PER_HOUR

# Large instances
RUNS = 50
NUM_DAYS = InstanceParameter(default=2, min=1, max=5, step=1, name='number_of_days', nice_name='Number of days')
FLEET_SIZE = InstanceParameter(default=12, min=12, max=68, step=8, name='fleet_size', nice_name='Fleet size')
TIME_WINDOW_LENGTH = InstanceParameter(default=4, min=0, max=8, step=1, name='time_window_length', nice_name='Length of departure time window size (Periods)')
CHARGER_CAPACITY = InstanceParameter(default=6, min=6, max=FLEET_SIZE.default, step=2, name='base_charger_capacity', nice_name='Charger capacity')
NUM_CHARGER_TYPES_CONSTANT_TOTAL = InstanceParameter(default=2, min=1, max=CHARGER_CAPACITY.default, step=1, name='number_of_charger_types', nice_name='Number of chargers (constant total)')
NUM_CHARGER_TYPES_VARYING_TOTAL = InstanceParameter(default=NUM_CHARGER_TYPES_CONSTANT_TOTAL.default,
                                                    min=NUM_CHARGER_TYPES_CONSTANT_TOTAL.min,
                                                    max=NUM_CHARGER_TYPES_CONSTANT_TOTAL.max,
                                                    step=1, name='number_of_charger_types',
                                                    nice_name='Number of chargers (varying total)')
CHARGER_COMPLEXITY = InstanceParameter(default=3, min=2, max=8, step=1, name='charger_complexity', nice_name='No. of segments of the piece-wise linear approximation')
WDF_COMPLEXITY = InstanceParameter(default=4, min=2, max=8, step=1, name='wdf_complexity', nice_name='No. of segments of the piece-wise linear approximation')
PARAMETERS = (FLEET_SIZE, NUM_DAYS, NUM_CHARGER_TYPES_CONSTANT_TOTAL, NUM_CHARGER_TYPES_VARYING_TOTAL, CHARGER_CAPACITY, CHARGER_COMPLEXITY, TIME_WINDOW_LENGTH, WDF_COMPLEXITY)

MIP_RUNS = 50