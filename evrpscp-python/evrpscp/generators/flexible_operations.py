# coding=utf-8
from dataclasses import dataclass
from itertools import cycle
from random import Random
from typing import List, Optional

from evrpscp import DiscretePeriod, DiscreteTour, Charger
from .models import Parameter, parameterized


@dataclass
class SingleTourSchedule:
    pause: int
    duration: int
    time_window_length: int
    consumption: float


def create_tours_from_schedules(periods: List[DiscretePeriod], schedules: List[SingleTourSchedule]) -> List[DiscreteTour]:
    # Construct tours
    tours = []
    scheduled_period_index = 0
    for tour_id, schedule in enumerate(schedules):
        scheduled_period_index += schedule.pause

        # Add time window
        tw_size = schedule.time_window_length
        before_scheduled_departure = int(round(tw_size/2))
        after_scheduled_departure = tw_size - before_scheduled_departure

        earliest_departure_period_id = scheduled_period_index - before_scheduled_departure
        if earliest_departure_period_id < 0:
            after_scheduled_departure += abs(earliest_departure_period_id)
            earliest_departure_period_id = 0

        latest_departure_period_id = scheduled_period_index + after_scheduled_departure
        if (overhead := latest_departure_period_id + schedule.duration + 1 - (len(periods) - 1)) > 0:
            earliest_departure_period_id = max(0, earliest_departure_period_id - overhead)
            latest_departure_period_id = latest_departure_period_id - overhead
        earliest_departure_period_id = min(earliest_departure_period_id, latest_departure_period_id)

        tours.append(
            DiscreteTour(
                id=tour_id, duration=schedule.duration, earliest_departure=periods[earliest_departure_period_id],
                latest_departure=periods[latest_departure_period_id],
                consumption=schedule.consumption, cost=0.0
            )
        )

        scheduled_period_index += schedule.duration
    return tours

# min_consumption/max_consumption are given as SoC values
@parameterized
def generate_flexible_tours(periods: List[DiscretePeriod], num_tours: int, consumption: Parameter, duration: Parameter,
                            time_window_length: Parameter, free_periods_before_first_tour: Parameter, randomize_breaks: bool = False, min_pause: int = 0,
                            generator: Optional[Random] = Random()) -> List[DiscreteTour]:
    # First, distribute tours without considering the time window just according to
    free_periods_before_first_tour: int = free_periods_before_first_tour.generate(generator)

    tour_schedules = [SingleTourSchedule(min_pause, duration.generate(generator), time_window_length.generate(generator),
                                         consumption.generate(generator)) for _ in range(num_tours)]
    tour_schedules[0].pause = max(min_pause, free_periods_before_first_tour)
    # Add dummy tour schedule to avoid having all plans end with the same period
    tour_schedules.append(SingleTourSchedule(0, 0, 0, 0))
    total_time_req = sum(x.pause + x.duration for x in tour_schedules)
    total_time_avail = len(periods) - 1 - free_periods_before_first_tour
    slack_available = total_time_avail - total_time_req
    assert slack_available >= 0, f'Not enough slack available to schedule tours: {slack_available}, time req: {total_time_req}, time avail: {total_time_avail}'
    # Distribute available slack evenly
    # TODO This will probably fail to yield feasible instances
    if not randomize_breaks:
        for i, _ in zip(cycle(tour_schedules), range(slack_available)):
            i.pause += 1
    else:
        for _ in range(slack_available):
            generator.choice(tour_schedules).pause += 1
    # Remove dummy
    dummy_tour = tour_schedules.pop()

    return create_tours_from_schedules(periods=periods, schedules=tour_schedules)

def calculate_periods_required_for_charging(num_vehicles: int, chargers: List[Charger], target_soc: float, initial_soc: float, period_length: float) -> int:
    """
    Calculates the number of periods required to recharge num_vehicles with the given initial soc such that they reach the target_soc using chargers.
    Is not optimal.
    """
    # Calculate the number of periods that we need to recharge all vehicles such that they can travel their first tours
    periods_required = 0
    vehicle_socs = [initial_soc for _ in range(num_vehicles)]
    while len(remaining_vehicles := {x: vehicle_socs[x] for x in range(num_vehicles) if vehicle_socs[x] < target_soc}) > 0:
        charger_capacities = {charger: charger.capacity for charger in chargers}
        for vehicle_id, soc in remaining_vehicles.items():
            available_chargers: List[Charger] = [charger for charger, capa in charger_capacities.items() if capa > 0]
            if len(available_chargers) == 0:
                break
            # Pick fastest charger
            fastest_charger: Charger = max(available_chargers, key=lambda f: f.charge_for(soc, period_length))
            charger_capacities[fastest_charger] -= 1

            vehicle_socs[vehicle_id] += fastest_charger.charge_for(soc, period_length)

        periods_required += 1

    return periods_required
