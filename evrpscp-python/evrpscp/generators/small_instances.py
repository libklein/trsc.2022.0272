# coding=utf-8
# std lib
import random
from copy import copy, deepcopy
from typing import List, Optional
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper

def generate_small_instance(num_tours: int, seed: Optional[str]) -> SchedulingInstance:
    battery = deepcopy(Pelletier.Battery)
    # Start with an empty battery
    battery.initialCharge = battery.minimumCharge
    # Summer plan
    tou_plan = TOURatePlan.SUMMER

    day_length = 48 * 30
    max_consumption = 1.0 / num_tours

    instance = generate_instance(
        tours_per_day=num_tours,
        tour_length=((2 * 30, ((48 // num_tours) / 2) * 30),),
        tour_arrival=(None,),
        tour_departure=((0, day_length),),
        tour_consumption=((0.25 * max_consumption, 0.75 * max_consumption),),
        fleet_size=3, fast_charger_count=1, number_of_days=1, period_length=30, tou_rates=tou_plan, seed=seed,
        battery=battery
    )

    return instance

@click.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--suffix', default='')
@click.option('--num-instances', default=500)
@click.option('--num-tours', default=list([2]))
@click.option('--seed')
def generate_cli(output_directory: Path, suffix: str, num_tours: List[int], num_instances: int, seed: Optional[str]):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    seed_generator = random.Random(seed)

    for instance_id in range(num_instances):
        for tour_count in num_tours:
            instance = generate_small_instance(tour_count, seed=str(seed_generator.random()))

            # Write to file
            name = f'small_{instance.param.fleetSize}v_{tour_count}t_{instance_id}_{suffix}'
            Dump.DumpSchedulingInstance((output_directory / name).with_suffix('.dump.d'), name, instance, is_discretized=True)

if __name__ == '__main__':
    generate_cli()