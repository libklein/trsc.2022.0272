# coding=utf-8
from __future__ import annotations
from evrpscp.models import Period, DiscretePeriod, Tour, DiscreteTour, Parameters, DiscreteParameters, FleetTourPlan, Charger, SchedulingInstance
from evrpscp.util import skip
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from funcy import cached_readonly, with_prev


def create_discrete_periods(cont_periods: List[Period], period_duration: float) \
        -> List[DiscretePeriod]:
    periods = [DiscretePeriod(
        int(p.begin + p_id * period_duration),
        int(p.begin + (p_id + 1) * period_duration),
        p.energyPrice
    ) for p in cont_periods for p_id in range(int(p.duration)//int(period_duration))]

    for p_prev, p_next in zip(periods, periods[1:]):
        p_prev.succ = p_next
        p_next.pred = p_prev

    return periods


def create_discrete_tours(tours: FleetTourPlan, periods: List[DiscretePeriod]) -> List[List[DiscreteTour]]:
    return [[DiscreteTour.FromTour(pi, periods) for pi in vehicle_tours] for vehicle_tours in tours]

def decode_periods_from_json(decoded_list: List[Dict]) -> List[DiscretePeriod]:
    periods = [DiscretePeriod(**x) for x in decoded_list]
    for cur, prev in with_prev(periods):
        cur.pred = prev
        if prev is not None:
            prev.succ = cur
    return periods

# Declare vehicle type
Vehicle = int

@dataclass_json
@dataclass
class DiscretizedInstance:
    periods: List[DiscretePeriod] = field(metadata=config(
        decoder=decode_periods_from_json
    ))
    tours: List[List[DiscreteTour]]
    chargers: List[Charger]
    parameters: DiscreteParameters

    def __post_init__(self):
        # Homogeneous period length
        durations = {p.duration for p in self.periods}
        if len(durations) != 1:
            raise ValueError("Inhomogeneous discretization!")
        # Sort tours
        for vehicle_tours in self.tours:
            vehicle_tours.sort(key=lambda pi: pi.latest_departure_time)
        # Setup period linked list
        self.periods[0].pred = None
        for p_prev, p in zip(self.periods, skip(self.periods, 1)):
            p_prev.succ = p
            p.pred = p_prev
        self.periods[-1].succ = None

    @cached_readonly
    def latest_arrival_time(self) -> float:
        return max(pi.latest_arrival_time for schedule in self.tours for pi in schedule)

    @property
    def vehicles(self):
        return range(len(self.tours))

    @staticmethod
    def DiscretizeInstance(instance: SchedulingInstance, period_duration=30) -> DiscretizedInstance:
        periods = create_discrete_periods(instance.periods, period_duration=period_duration)
        tours = create_discrete_tours(instance.tourPlans, periods)
        parameters = DiscreteParameters.FromContinuous(instance.param, periods)

        return DiscretizedInstance(periods=periods, tours=tours, chargers=instance.chargers, parameters=parameters)