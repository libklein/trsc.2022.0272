# coding=utf-8

"""
    Benchmark instance generator.
        - Battery with 80 kWh
        - Random WDF
        - Random chargers
            * First randomly choose the number of intervals
            * Then randomly choose what percentage of Image should be covered by each interval
            * Then randomly choose slopes, and sort these
            * Then construct segments from that and image ub
    Tours
"""

import random
from typing import List, Optional, Tuple, TypeVar
from json import dump as dump_json
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
from evrpscp import *
from generators.models import Range
from generators.models.battery import generate_battery
from generators.models.charger import generate_charger
from generators.models.periods import generate_tou_rates

from .models import Parameter

PWL_RAND_GEN = random.Random()
BATTERY_RAND_GEN = random.Random()
CHARGER_RAND_GEN = random.Random()
PERIOD_RAND_GEN = random.Random()
TOUR_RAND_GEN = random.Random()

def generate_chargers(battery_capacity: float, count: int = 1, capacity: Parameter = Parameter(1, 3),
                      duration: Parameter = Parameter(300, 480)) -> List[Charger]:
    # Base charger
    chargers = [
        generate_charger(battery_capacity=battery_capacity, capacity=100, isBaseCharger=True, duration=660)
    ]
    for charger_id in range(count):
        chargers.append(generate_charger(battery_capacity=battery_capacity,
                                          duration=duration.generate(CHARGER_RAND_GEN),
                                          capacity=round(capacity.generate(CHARGER_RAND_GEN)),
                                          isBaseCharger=False))
    # Sort by duration
    chargers.sort(key=lambda f: f.chargingFunction.upper_bound)
    for next_id, f in enumerate(chargers):
        f.id = next_id
    return chargers

def generate_parameters(battery: Battery, latest_arrival: float, fleet_size: int) -> Parameters:
    return Parameters(fleetSize=fleet_size, battery=battery, max_charges_between_tours=100)

# min_consumption/max_consumption are given as SoC values
def generate_tours_for_vehicle(battery, num_tours=2, consumption: Parameter = Parameter(.25, .75),
                               free_time_before: Parameter = Parameter(60.0, 300.0),
                               duration: Parameter = Parameter(60.0, 60.0),
                               time_window_size: Parameter = Parameter(30.0, 60.0),
                               cost: Parameter = Parameter(10.0, 50.0)) -> List[Tour]:
    prev_arrival = 0.0
    tours = []
    for tour_id in range(num_tours):
        tour_consumption = round(consumption.generate(TOUR_RAND_GEN) * (battery.maximumCharge - battery.minimumCharge),
                                 2)
        time_before = round(free_time_before.generate(TOUR_RAND_GEN), 2)
        tours.append(Tour(earliest_departure_time=round(time_before + prev_arrival, 2),
                          latest_departure_time=round(
                              time_before + prev_arrival + time_window_size.generate(TOUR_RAND_GEN), 2),
                          consumption=tour_consumption, cost=round(cost.generate(TOUR_RAND_GEN), 2),
                          duration_time=duration.generate(TOUR_RAND_GEN), id=tour_id))
        prev_arrival = tours[-1].latest_arrival_time

    return tours


def generate_tour_plan(battery: Battery, chargers: List[Charger], fleet_size: int, tours_per_vehicle: int,
                       consumption: Parameter = Parameter(0.25, 0.95),
                       free_time_before: Parameter = Parameter(60.0, 150.0),
                       duration: Parameter = Parameter(60., 60.), cost: Parameter = Parameter(10.0, 50.0),
                       time_window_size: Parameter = Parameter(30., 60.)):
    tours = []
    for vehicle in range(fleet_size):
        tours.append(generate_tours_for_vehicle(battery=battery, num_tours=tours_per_vehicle,
                                                consumption=consumption, duration=duration,
                                                free_time_before=free_time_before, cost=cost,
                                                time_window_size=time_window_size))
    next_id = 0
    for veh_tours in tours:
        for pi in veh_tours:
            pi.id = next_id
            next_id += 1
    return tours


def generate_small_instance(fleet_size: int, tours_per_vehicle: int, seed: Optional[str], num_chargers: int,
                            charger_dur: Range[float] = (300.0, 480.0), charger_capacity: Range[int] = (1, 2),
                            tour_free_time_before: Range[float] = (60.0, 300.0),
                            tour_duration: Range[float] = (60.0, 60.0),
                            tour_cost: Range[float] = (10.0, 50.0), tour_consumption: Range[float] = (0.25, 1.0),
                            tour_time_window_size: Range[float] = (30.0, 60.0), tou_cost: Range[float] = (0.05, 1.0)) \
        -> SchedulingInstance:
    period_length = 30.0
    battery = generate_battery()
    chargers = generate_chargers(count=num_chargers, battery_capacity=battery.capacity,
                                 duration=Parameter(*charger_dur),
                                 capacity=Parameter(*charger_capacity))
    generated_tours = generate_tour_plan(battery=battery, chargers=chargers, tours_per_vehicle=tours_per_vehicle,
                                         fleet_size=fleet_size,
                                         consumption=Parameter(*tour_consumption),
                                         free_time_before=Parameter(*tour_free_time_before),
                                         duration=Parameter(*tour_duration), cost=Parameter(*tour_cost),
                                         time_window_size=Parameter(*tour_time_window_size))
    latest_arrival = ((max(vt[-1].latest_arrival_time for vt in generated_tours) // period_length) + 1) * period_length
    param = generate_parameters(battery, latest_arrival=latest_arrival, fleet_size=fleet_size)
    # Start with an empty battery
    battery.initialCharge = battery.minimumCharge
    # Summer plan
    periods = generate_tou_rates(end_of_horizon=latest_arrival, period_length=period_length, cost=Parameter(*tou_cost))

    tour_plan = FleetTourPlan([TourPlan(tours, vehicleID=v_id) for v_id, tours in enumerate(generated_tours)],
                              tours_per_vehicle)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=param, tourPlans=tour_plan)
    return instance


def validate_instance_feasibility(instance: SchedulingInstance):
    return True


@click.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--suffix', default=None, type=str)
@click.option('--num-instances', default=500, type=int)
@click.option('--num-vehicles', default=9, type=int)
@click.option('--num-tours', default=4, type=int)
@click.option('--num-chargers', default=3, type=int)
@click.option('--charger-capacity', default=(1, 2), type=(int, int))
@click.option('--full-charge-duration', default=(300, 480), type=(float, float))
@click.option('--tour-discharge', default=(.25, .75), type=(float, float))
@click.option('--tour-cost', default=(5.0, 50.0), type=(float, float))
@click.option('--free-time-before-tour', default=(60.0, 300.0), type=(float, float))
@click.option('--tour-time-window-size', default=(30.0, 60.0), type=(float, float))
@click.option('--tour-duration', default=(60.0, 60.0), type=(float, float))
@click.option('--energy-price', default=(0.05, 1.0), type=(float, float))
@click.option('--check-feasibility', default=False, is_flag=True)
@click.option('--seed')
def generate_cli(output_directory: Path, suffix: Optional[str], num_tours: int, num_instances: int, seed: Optional[str],
                 num_chargers: int, num_vehicles: int, charger_capacity: Range[int], full_charge_duration: Range[float],
                 tour_discharge: Range[float], free_time_before_tour: Range[float], tour_time_window_size: Range[float],
                 energy_price: Range[float],
                 tour_cost: Range[float], tour_duration: Range[float], check_feasibility: bool):
    instance_generator_parameters = locals()
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    seed_generator = random.Random(seed)

    instance_count = 0
    while instance_count < num_instances:
        run_seed = str(seed_generator.random())
        instance = generate_small_instance(tours_per_vehicle=num_tours, num_chargers=num_chargers,
                                           seed=run_seed, fleet_size=num_vehicles, charger_dur=full_charge_duration,
                                           charger_capacity=charger_capacity,
                                           tour_free_time_before=free_time_before_tour,
                                           tour_duration=tour_duration, tour_cost=tour_cost,
                                           tour_consumption=tour_discharge, tour_time_window_size=tour_time_window_size,
                                           tou_cost=energy_price)

        if check_feasibility:
            is_feasible = validate_instance_feasibility(instance)
            if not is_feasible:
                print(f"\rGenerated infeasible instance (retrying). ({instance_count}/{num_instances})", end='',
                      flush=True)
                continue

        instance_count += 1
        print(f'\rGenerated {instance_count}/{num_instances} instances!', end='', flush=True)

        # Write to file
        name = f'benchmark_{num_vehicles}v_{num_tours}t_{num_chargers}c_{instance_count}{("_" + suffix) if suffix is not None else ""}'
        instance_dir_path = (output_directory / name).with_suffix('.dump.d')
        Dump.DumpSchedulingInstance(instance_dir_path, name, instance, is_discretized=True)
        with open(str(instance_dir_path / 'info.json'), 'w') as param_file:
            dump_json(dict({x: val for x, val in instance_generator_parameters.items() if '__' != x[:2]}, run_seed=run_seed),
                 param_file, indent=2)


if __name__ == '__main__':
    generate_cli()
