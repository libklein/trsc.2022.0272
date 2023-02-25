# coding=utf-8
import random
from copy import copy, deepcopy
from typing import List, Optional, Tuple
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.models import *
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper
from funcy import nth, flatten


def construct_battery() -> Battery:
    return generate_battery()

def construct_chargers(capacity: Parameter, duration: Parameter, intervals: Parameter, count=1) -> List[Charger]:
    return [generate_charger(battery_capacity=80.0, charger_capacity=capacity.generate(), duration=duration.generate(),
                             intervals=intervals.generate(), id=c) for c in range(count)]

def generate_tou_rates(end_of_horizon: float, period_length: float = 30, cost: Parameter = Parameter(0.5, 1.0)):
    return generate_tou_rates_discrete(periods_in_horizon=int(end_of_horizon//period_length), period_length=period_length, cost=cost)

# min_consumption/max_consumption are given as SoC values
def generate_tours(first_period: DiscretePeriod, num_tours: int, consumption: Parameter, duration: Parameter, time_window: Parameter, free_periods: Parameter) -> List[DiscreteTour]:
    prev_arrival = first_period
    tours = []
    for tour_id in range(num_tours):
        tour_time_before = free_periods.generate()

        _earliest_begin = nth(tour_time_before, iter(prev_arrival))
        _dur = duration.generate()

        tours.append(
            DiscreteTour(
                id=tour_id, duration=_dur, earliest_departure=_earliest_begin,
                latest_departure=nth(time_window.generate(), iter(_earliest_begin)),
                consumption=consumption.generate(), cost=0.0
            )
        )

        prev_arrival = tours[-1].latest_arrival

    return tours

def generate_small_instance(num_tours: int, seed: Optional[str], num_chargers: int, energy_price: Parameter, charger_capacity: Parameter, consumption: Parameter, num_vehicles: int = 1) -> SchedulingInstance:
    period_length = 30.0
    full_charge_dur = Parameter(300)
    tour_dur = Parameter(4, 6)
    free_time_before_tour = Parameter(int(full_charge_dur.max//period_length))
    time_window_length = Parameter(0, 2)
    latest_end_of_horizon = num_tours * (free_time_before_tour.max + tour_dur.max + 1 + time_window_length.max) * period_length

    periods = generate_tou_rates(end_of_horizon=latest_end_of_horizon, period_length=period_length, cost=energy_price)

    chargers, battery = construct_chargers(capacity=charger_capacity, duration=full_charge_dur, count=num_chargers, intervals=Parameter(3, 5)), construct_battery()
    tour_plans = []
    next_tour_id = 0
    latest_arrival = periods[0]
    for veh in range(num_vehicles):
        tours = generate_tours(num_tours=num_tours,
                               consumption=Parameter(consumption.min * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge,
                                                     consumption.max * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge),
                               duration=tour_dur, time_window=time_window_length,
                               free_periods=free_time_before_tour, first_period=periods[0])
        for pi in tours:
            pi.id = next_tour_id
            next_tour_id += 1
        latest_arrival = max(latest_arrival, tours[-1].latest_arrival)
        tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
    # Remove superfluous periods
    periods = periods[:periods.index(latest_arrival)+1]
    periods[-1].succ = None

    # Start with an empty battery
    battery.initialCharge = battery.minimumCharge

    param = Parameters(fleetSize=num_vehicles, battery=battery, max_charges_between_tours=100)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=param, tourPlans=FleetTourPlan(tour_plans, fleetSize=num_vehicles))
    return instance

@click.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--suffix', default='')
@click.option('--num-instances', default=500, type=int)
@click.option('--num-tours', default=list([2]), type=int, multiple=True)
@click.option('--num-chargers', default=list([1]), type=int, multiple=True)
@click.option('--num-vehicles', default=list([1]), type=int, multiple=True)
@click.option('--min-energy-price', type=float, default=0.5)
@click.option('--max-energy-price', type=float, default=1.0)
@click.option('--min-soc-consumption', type=float, default=0.25)
@click.option('--max-soc-consumption', type=float, default=0.75)
@click.option('--min-charger-capacity', type=int, default=1)
@click.option('--max-charger-capacity', type=int, default=1)
@click.option('--seed')
def generate_cli(output_directory: Path, suffix: str, num_tours: List[int], num_instances: int, seed: Optional[str],
                 min_energy_price: float, max_energy_price: float, num_chargers: List[int], num_vehicles: List[int],
                 min_charger_capacity: int, max_charger_capacity: int, min_soc_consumption: float,
                 max_soc_consumption: float):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    if any(c not in (1, 2) for c in num_chargers):
        raise NotImplementedError('Only 1/2 chargers are supported at the moment')

    seed_generator = random.Random(seed)

    for instance_id in range(num_instances):
        for veh_count in num_vehicles:
            for tour_count in num_tours:
                for charger_count in num_chargers:
                    instance = generate_small_instance(num_tours=tour_count, seed=str(seed_generator.random()),
                                                       num_chargers=charger_count,
                                                       energy_price=Parameter(min_energy_price, max_energy_price),
                                                       charger_capacity=Parameter(min_charger_capacity, max_charger_capacity),
                                                       consumption=Parameter(min_soc_consumption, max_soc_consumption),
                                                       num_vehicles=veh_count)

                    for p in instance.periods:
                        delattr(p, 'pred')
                        delattr(p, 'succ')

                    # Write to file
                    name = f'tiny_{instance.param.fleetSize}v_{tour_count}t_{charger_count}c_{instance_id}_{suffix}'
                    Dump.DumpSchedulingInstance((output_directory / name).with_suffix('.dump.d'), name, instance, is_discretized=True)

if __name__ == '__main__':
    generate_cli()
