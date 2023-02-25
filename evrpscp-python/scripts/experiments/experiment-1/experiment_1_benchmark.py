# coding=utf-8
from copy import deepcopy
from itertools import cycle
from pathlib import Path
from string import ascii_letters
from typing import List, Union, Optional, Dict

import click
import matplotlib.pyplot as plt

from parameters import *

try:
    from evrpscp.generators.models import *
    from evrpscp.generators.flexible_operations import generate_flexible_tours, calculate_periods_required_for_charging
    from evrpscp import *
except:
    pass
from random import Random

from funcy import nth

PERIOD_LENGTH = 30.0

def generate_battery_exp1(capacity: float, num_intervals: int, generator: Random = Random()) -> Battery:
    return generate_battery(capacity=Parameter(capacity), intervals=Parameter(num_intervals), initial_charge=Parameter(0.0), generator=generator)

def generate_charging_infrastructure_exp1(battery_capacity: float, fleet_size: int, base_charger_capacity: int, number_of_charger_types: int, num_intervals=3, base_duration: float=2.5*60, generator: Random = Random(), total_charger_capacity: Optional[int] = None) -> List[Charger]:
    # Base charger requires 6h for a full load
    charger_generator = Random(generator.random())
    chargers = [generate_charger(battery_capacity=battery_capacity, duration=base_duration,
                                 charger_capacity=base_charger_capacity, generator=charger_generator,
                                 isBaseCharger=False, intervals=num_intervals)]

    for i in range(number_of_charger_types-1):
        charge_duration = charger_generator.uniform(base_duration * 0.95, base_duration * 1.05)
        chargers.append(generate_charger(battery_capacity, charger_capacity=base_charger_capacity,
                                         duration=charge_duration, isBaseCharger=False, id=i+1,
                                         generator=charger_generator, intervals=num_intervals))

    if total_charger_capacity is not None:
        assert total_charger_capacity >= len(chargers)
        for charger in chargers:
            charger.capacity = 0
        for _, charger in zip(range(total_charger_capacity), cycle(chargers)):
            charger.capacity += 1

    assert all(f.capacity > 0 for f in chargers)
    return chargers

def generate_periods_exp1(number_of_days: int, generator: Random = Random()) -> List[DiscretePeriod]:
    return generate_tou_rates_discrete(periods_in_horizon=PERIODS_PER_DAY*number_of_days, period_length=PERIOD_LENGTH, cost=Parameter(0.5, 1.0), generator=generator)

def generate_instance(generation_param: InstanceParameters, tours_per_day = 3, **overwrite) -> SchedulingInstance:
    def get_param_value(name):
        return overwrite.get(name, getattr(generation_param, name))
    fleet_size = get_param_value('fleet_size')
    number_of_days = get_param_value('number_of_days')
    average_tour_length = get_param_value('average_tour_length')
    time_window_periods = get_param_value('time_window_length')
    number_of_charger_types = get_param_value('number_of_charger_types')
    base_capacity = get_param_value('base_charger_capacity')
    num_intervals = get_param_value('charger_complexity')
    wdf_complexity = get_param_value('wdf_complexity')
    scale_charger_capacity = get_param_value('scale_charger_capacity')

    battery = generate_battery_exp1(capacity=80.0, num_intervals=wdf_complexity, generator=generation_param.BATTERY_RAND_GEN)
    periods = generate_periods_exp1(number_of_days=number_of_days, generator=generation_param.ENERGY_PRICE_RAND_GEN)
    chargers = generate_charging_infrastructure_exp1(battery_capacity=battery.capacity, fleet_size=fleet_size,
                                                     num_intervals=num_intervals,
                                                     base_charger_capacity=base_capacity,
                                                     number_of_charger_types=number_of_charger_types,
                                                     total_charger_capacity=base_capacity if not scale_charger_capacity else None,
                                                     generator=generation_param.INFRASTRUCTURE_RAND_GEN)
    assert len(chargers) == number_of_charger_types

    periods_before_first_tour = calculate_periods_required_for_charging(num_vehicles=fleet_size,
                                                                        chargers=chargers,
                                                                        target_soc=generation_param.consumption
                                                                                   - battery.initialCharge,
                                                                        initial_soc = battery.initialCharge,
                                                                        period_length=PERIOD_LENGTH)

    tour_plans = []
    tour_id_offset = 0
    for v_id, generator in enumerate(generation_param.VEHICLE_RAND_GEN):
        tours = generate_flexible_tours(periods=periods, num_tours=tours_per_day*number_of_days,
                                        consumption=Parameter(generation_param.consumption
                                                              * (battery.maximumCharge - battery.minimumCharge)
                                                              + battery.minimumCharge),
                                        duration=Parameter(average_tour_length),
                                        time_window_length=Parameter(time_window_periods),
                                        randomize_breaks=True, min_pause=3,
                                        free_periods_before_first_tour=periods_before_first_tour, generator=generator)
        for pi in tours:
            pi.id += tour_id_offset
        tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=v_id))
        tour_id_offset += len(tours)
    vehicle_tours = FleetTourPlan(tour_plans)

    parameters = Parameters(fleetSize=fleet_size, battery=battery, max_charges_between_tours=100)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=parameters,
                                  tourPlans=vehicle_tours)
    return instance

@click.group()
def cli():
    pass

@cli.command('scalability')
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--plot', is_flag=True)
def generate_experiment_1_scalability(output_directory: Path, plot: bool):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    def from_defaults(seed, **kwargs):
        _def = {p.name: p.default for p in PARAMETERS}
        _def.update(**kwargs)

        return InstanceParameters(seed=seed,
                                  consumption=0.55,
                                  average_tour_length=(5 * PERIODS_PER_HOUR),
                                  **_def)

    def create_tour_plans_from_reference(reference_plans: List[TourPlan], num_plans: int) -> List[TourPlan]:
        if len(reference_plans) > num_plans:
            return deepcopy(reference_plans[:num_plans])
        assert num_plans % len(reference_plans) == 0
        plan_copies = []
        for i in range(num_plans // len(reference_plans)):
            plan_copies.extend(deepcopy(reference_plans))
        for i, plan in enumerate(plan_copies):
            plan.vehicleID = i
        return plan_copies

    def generate_run(param: InstanceParameter, overwrites=None) -> Dict:
        if overwrites is None:
            overwrites = dict()

        generated_instances = {}
        for val in param:
            instance_param = from_defaults(seed=seed, infix=param.name, **{param.name: val})

            if param == FLEET_SIZE:
                overwrites['base_charger_capacity'] = int(val / 2.0)
            elif param == NUM_CHARGER_TYPES_VARYING_TOTAL:
                instance_param.scale_charger_capacity = True
            #if param in (NUM_CHARGER_TYPES_VARYING_TOTAL, NUM_CHARGER_TYPES_CONSTANT_TOTAL):
                # Experiment does not give good results for a charger capacity of 3
            #    overwrites['base_charger_capacity'] = 4

            inst = generate_instance(generation_param=instance_param, tours_per_day=3, **overwrites)

            generated_instances[instance_param] = inst

        if param == FLEET_SIZE:
            reference_plans = next(inst.tourPlans.schedules for inst_param, inst in generated_instances.items() if inst_param.fleet_size == param.max)
            # Copy vehicle schedules to ensure comparability
            for inst_param, instance in generated_instances.items():
                fleet_size = getattr(inst_param, param.name)
                instance.tourPlans = FleetTourPlan(schedules=create_tour_plans_from_reference(
                    reference_plans=reference_plans, num_plans=fleet_size), fleetSize=fleet_size)
        elif param in (CHARGER_CAPACITY, NUM_CHARGER_TYPES_VARYING_TOTAL, NUM_CHARGER_TYPES_CONSTANT_TOTAL):
            if param is NUM_CHARGER_TYPES_VARYING_TOTAL:
                reference_plans = generated_instances[
                    from_defaults(seed=seed, infix=param.name, **{param.name: param.min, 'scale_charger_capacity': True})].tourPlans.schedules
            else:
                reference_plans = generated_instances[
                    from_defaults(seed=seed, infix=param.name, **{param.name: param.min, 'scale_charger_capacity': False})].tourPlans.schedules

            for inst in generated_instances.values():
                assert len(inst.tourPlans.schedules) == len(reference_plans)
                inst.tourPlans.schedules = create_tour_plans_from_reference(reference_plans, len(reference_plans))

        return generated_instances

    seed_generator = Random()
    for _ in range(RUNS):
        seed = "".join(seed_generator.choices(ascii_letters, k=4))
        for param in PARAMETERS:
            # Generate
            generated_instances = generate_run(param=param)

            # Write
            for instance_param, instance in generated_instances.items():
                write_instance(instance_param, output_directory=output_directory, instance=instance)

@cli.command('mip')
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
def generate_experiment_1_mip(output_directory: Path):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    for run in range(MIP_RUNS):
        seed = str(run)
        # Parameters
        generation_param = InstanceParameters(
            seed=seed, run=run, infix='MIP',
            fleet_size=3, number_of_days=1, time_window_length=6,
            average_tour_length=(5*PERIODS_PER_HOUR)+1,
            number_of_charger_types=1, charger_complexity=3, base_charger_capacity=2,
            wdf_complexity=4, consumption=0.55
        )
        # Generate
        instance = generate_instance(generation_param=generation_param, tours_per_day=3)

        # Write
        write_instance(param=generation_param, output_directory=output_directory, instance=instance)

def write_instance(param: InstanceParameters, output_directory: Path, instance: SchedulingInstance):
    inst_output_dir = (output_directory / (param.instancename + '.dump.d'))

    if inst_output_dir.exists():
        raise IOError(f"Target directory {inst_output_dir} exists! Refusing to overwrite.")

    Dump.DumpSchedulingInstance(directory=inst_output_dir, instance_name=param.instancename, instance=instance,
                                is_discretized=True)
    with open(str(inst_output_dir / 'info.json'), 'w') as param_file:
        param_file.write(param.to_json())
    return inst_output_dir

if __name__ == '__main__':
    cli()