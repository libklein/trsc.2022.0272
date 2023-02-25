from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement
from dataclasses_json import dataclass_json
from typing import List

from . import DiscretePeriod
from . import Battery, find_intersecting_period

@dataclass_json
@dataclass
class Parameters:
    fleetSize: int
    battery: Battery
    max_charges_between_tours: int = 2

@dataclass_json
@dataclass
class DiscreteParameters:
    fleetSize: int
    battery: Battery
    period_length: float
    max_charges_between_tours: int = 2

    @staticmethod
    def FromContinuous(param: Parameters, periods: List[DiscretePeriod]) -> 'DiscreteParameters':
        # Assert homogeneous periods
        for prev,next in zip(periods, periods[1:]):
            assert prev.duration == next.duration

        return DiscreteParameters(fleetSize=param.fleetSize, battery=param.battery, period_length=periods[0].duration)
