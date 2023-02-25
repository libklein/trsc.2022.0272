# coding=utf-8
import random
from random import Random
from copy import copy, deepcopy
from operator import attrgetter
from typing import List, Optional, Tuple, Union, Any
from json import dump as dump_json
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.models import *
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper
from funcy import nth, flatten, with_prev, cycle, drop, chain
from scipy.interpolate import interp1d

PERIODS_PER_DAY=48
PERIOD_LENGTH=24*60/PERIODS_PER_DAY
# Hourly rates. Taken from
# https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=GRAPH&areaType=BZN&atch=false&dateTime.dateTime=15.01.2021+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A83F!BZN|10Y1001A1001A82H&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
# and https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=GRAPH&areaType=BZN&atch=false&dateTime.dateTime=16.01.2021+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A83F!BZN|10Y1001A1001A82H&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
HOURLY_DEFAULT_ENERGY_RATES = [58.41,55.65,52.16,51.18,49.51,48.52,48.74,51.43,61.16,64.75,63.77,62.13,55.01,50.66,49.48,50.65,57.7,65.01,66.4,63.76,57.47,52.98,53.2,50.56,48.98,48.66,47.7,46.6,48,51.52,62.97,77.94,94.07,97.76,90.69,86.97,79.25,76.64,77.14,76.98,82,99.5,98.02,95.01,77.2,67.18,63.19,55.42]
# Interpolate
DEFAULT_ENERGY_RATES = []
for rate, prev_rate in with_prev(HOURLY_DEFAULT_ENERGY_RATES, fill=HOURLY_DEFAULT_ENERGY_RATES[0]):
    DEFAULT_ENERGY_RATES.append(rate)
    DEFAULT_ENERGY_RATES.append((rate+prev_rate) / 2)
assert len(DEFAULT_ENERGY_RATES) == PERIODS_PER_DAY*2

def generate_periods(number_of_periods: int, period_dur: float, prices: List[float], reference_prices: List[float] = None, reference_average_price: float = None, scale_length: bool = False) -> List[DiscretePeriod]:
    if scale_length:
        # Scale prices such that it's length matches the number_of_periods
        # Create a piecewise linear function from the prices
        price_at_time = interp1d(x=chain([0], [(timestep*period_dur + period_dur/2) for timestep in range(len(prices))], [len(prices)*period_dur]),
                                 y=chain(prices[0], prices, prices[-1]), assume_sorted=True)

    # Generate periods
    periods = []
    for p_id in range(number_of_periods):
        begin, center, end = p_id * period_dur, (p_id + 0.5) * period_dur, (p_id+1) * period_dur
        periods.append(DiscretePeriod(begin=begin, end=end, energyPrice=price_at_time(center) if scale_length else prices[p_id]))


    # Determine scaling factor - used to ensure that the average price corresponds to the average price of the reference prices
    reference_average_price = sum(reference_prices)/len(reference_prices) if reference_average_price is None else reference_average_price
    scaling_factor = reference_average_price / (sum(map(lambda x: x.energyPrice, periods))/len(periods))
    for p in periods:
        p.energyPrice *= scaling_factor

    assert len(periods) == number_of_periods
    assert abs(sum(map(lambda x: x.energyPrice, periods))/len(periods) - reference_average_price) < 0.01

    for cur, prev in drop(1, with_prev(periods)):
        prev.succ = cur
        cur.prev = prev

    return periods

def construct_battery(*args, **kwargs) -> Battery:
    return generate_battery(*args, **kwargs)

def construct_charger(capacity: int, duration: float) -> Charger:
    """
    Constructs a charger based off the Pelletier Fast Charger
    """

    base_charger: Charger = Pelletier.FastCharger
    base_charger.capacity = capacity
    # Scale to duration
    scaled_charging_function = base_charger.chargingFunction.scale_domain(new_max=duration)
    return Charger(capacity=capacity, id=0, chargingFunction=scaled_charging_function,
                   inverseChargingFunction=scaled_charging_function.inverse(), isBaseCharger=False)

def generate_tou_rates(end_of_horizon: float, period_length: float = 30, avg_energy_price: Optional[Parameter] = None, energy_price_scale: float = 1.0, generator: Optional[Random] = None):
    periods = generate_periods(number_of_periods=int(end_of_horizon//period_length), period_dur=period_length, prices=DEFAULT_ENERGY_RATES,
                            reference_prices=list(map(attrgetter('energyPrice'), Pelletier.WinterRates)),
                            reference_average_price=avg_energy_price.generate(rand_gen=generator) if avg_energy_price else None, scale_length=True)
    for p in periods:
        p.energyPrice *= energy_price_scale
    return periods

# min_consumption/max_consumption are given as SoC values
def generate_tours(first_period: DiscretePeriod, num_tours: int, consumption: Parameter, duration: Parameter, time_window: Parameter, free_periods: Parameter, generator: Optional[Random] = None) -> List[DiscreteTour]:
    prev_arrival = first_period
    tours = []
    for tour_id in range(num_tours):
        tour_time_before = free_periods.generate(rand_gen=generator)

        _earliest_begin = nth(tour_time_before, iter(prev_arrival))
        _dur = duration.generate(rand_gen=generator)

        tours.append(
            DiscreteTour(
                id=tour_id, duration=_dur, earliest_departure=_earliest_begin,
                latest_departure=nth(time_window.generate(rand_gen=generator), iter(_earliest_begin)),
                consumption=consumption.generate(rand_gen=generator), cost=0.0
            )
        )

        prev_arrival = tours[-1].latest_arrival

    return tours

def generate_instance(num_tours: int, seed: Optional[str], num_chargers: int, avg_energy_price: Optional[Parameter],
                      charger_capacity: Parameter, consumption: Parameter, time_window_length_periods: Parameter,
                      full_charge_dur: Parameter, tour_dur_periods: Parameter, free_time_before_tour: Parameter,
                      energy_price_scale: float, num_vehicles: int = 1) -> SchedulingInstance:
    period_length = PERIOD_LENGTH
    latest_end_of_horizon = num_tours * (free_time_before_tour.max + tour_dur_periods.max + 1 + time_window_length_periods.max) * period_length

    seed_generation_engine = Random(seed)
    seed_generator = lambda: Random(seed_generation_engine.random())

    periods = generate_tou_rates(end_of_horizon=latest_end_of_horizon, period_length=period_length, energy_price_scale=energy_price_scale, generator=seed_generator())

    chargers, battery = construct_charger(capacity=charger_capacity, duration=full_charge_dur), \
                        construct_battery(generator=seed_generator())
    tour_plans = []
    tour_id_offset = 0
    latest_arrival = periods[0]
    for veh in range(num_vehicles):
        tours = generate_tours(num_tours=num_tours,
                               consumption=Parameter(consumption.min * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge,
                                                     consumption.max * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge),
                               duration=tour_dur_periods, time_window=time_window_length_periods,
                               free_periods=free_time_before_tour, first_period=periods[0], generator=seed_generator())
        for pi in tours:
            pi.id += tour_id_offset
        latest_arrival = max(latest_arrival, tours[-1].latest_arrival)
        tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
        tour_id_offset += len(tours)
    # Remove superfluous periods
    periods = periods[:periods.index(latest_arrival)+1]
    periods[-1].succ = None

    # Start with an empty battery
    battery.initialCharge = battery.minimumCharge

    param = Parameters(fleetSize=num_vehicles, battery=battery, max_charges_between_tours=100)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=param, tourPlans=FleetTourPlan(tour_plans, fleetSize=num_vehicles))
    return instance

def _instances_are_from_same_seed(lhs: SchedulingInstance, rhs: SchedulingInstance) -> bool:
    """
    Checks whether two instances have been generated from the same seed.
    """
    # Check Battery
    if lhs.param != rhs.param:
        return False
    # Check Periods
    for lhs_p, rhs_p in zip(lhs.periods, rhs.periods):
        if (lhs_p.begin, lhs_p.end, lhs_p.energyPrice) != (rhs_p.begin, rhs_p.end, rhs_p.energyPrice):
            return False
    # Check Charger
    for lhs_c, rhs_c in zip(lhs.chargers, rhs.chargers):
        if not lhs_c.equals(rhs_c):
            return False
    # Check Tours
    for lhs_v, rhs_v in zip(lhs.tourPlans, rhs.tourPlans):
        for lhs_t, rhs_t in zip(lhs_v, rhs_v):
            if not lhs_t.equals(rhs_t):
                return False
    return True

@click.group()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.pass_context
def cli(ctx: click.Context, output_directory):
    ctx.ensure_object(dict)
    ctx.obj['output_directory'] = output_directory if isinstance(output_directory, Path) else Path(output_directory)

@cli.command()
@click.pass_context
@click.option('--suffix', default='')
@click.option('--prefix', default='')
@click.option('--num-instances', default=500, type=int)
@click.option('--num-tours', default=list([2]), type=int, multiple=True)
@click.option('--num-chargers', default=list([1]), type=int, multiple=True)
@click.option('--num-vehicles', default=list([1]), type=int, multiple=True)
#@click.option('--min-energy-price', type=float, default=list([0.5]), multiple=True)
#@click.option('--max-energy-price', type=float, default=list([1.0]), multiple=True)
@click.option('--min-soc-consumption', type=float, default=list([0.25]), multiple=True, help='Min SoC consumption of tour ([0,1])')
@click.option('--max-soc-consumption', type=float, default=list([0.75]), multiple=True, help='Max SoC consumption of tour ([0,1])')
@click.option('--min-charger-capacity', type=int, default=list([1]), multiple=True, help='Min charger capacity')
@click.option('--max-charger-capacity', type=int, default=list([1]), multiple=True, help='Max charger capacity')
@click.option('--min-pause-before-tour', type=int, default=list([4]), multiple=True, help='Min length of pause before each tour in periods')
@click.option('--max-pause-before-tour', type=int, default=list([8]), multiple=True, help='Max length of pause before each tour in periods')
@click.option('--min-tour-length', type=int, default=list([8]), multiple=True, help='Min length of tours in periods')
@click.option('--max-tour-length', type=int, default=list([10]), multiple=True, help='Max length of tours in periods')
@click.option('--min-tw-length', type=int, default=list([4]), multiple=True, help='Min length of time window in periods')
@click.option('--max-tw-length', type=int, default=list([4]), multiple=True, help='Max length of time window in periods')
@click.option('--full-charge-dur', type=float, default=300, help='Duration of full charge in minutes')
@click.option('--energy-price-scale', type=float, default=1.0, help='Energy price scale')
@click.option('--seed')
def generate_cli(ctx: click.Context, prefix:str, suffix: str, seed: Optional[str],
                 num_chargers: List[int], num_vehicles: List[int], num_tours: List[int],
                 num_instances: int, full_charge_dur: float,
                 min_pause_before_tour: List[int], max_pause_before_tour: List[int],
                 min_tour_length: List[int], max_tour_length: List[int],
                 min_tw_length: List[int], max_tw_length: List[int],
                 min_charger_capacity: List[int], max_charger_capacity: List[int],
                 min_soc_consumption: List[float], max_soc_consumption: List[float],
                 energy_price_scale: float):
    output_directory = ctx.obj['output_directory']

    if any(c not in (1, 2) for c in num_chargers):
        raise NotImplementedError('Only 1/2 chargers are supported at the moment')

    seed_generator = random.Random(seed)

    to_param = lambda min_max_tpl: Parameter(*min_max_tpl)
    def abbr(key_name: str, value: Any) -> str:
        if value is not None:
            val_str = str(value.value()) if isinstance(value, Parameter) else str(value)
            return val_str+(''.join(x[0] for x in key_name.split('_')[:2]))
        return ''

    generated_instances = []
    for instance_id in range(num_instances):
        next_seed = ''.join(chr(seed_generator.randint(ord('a'), ord('z'))) for _ in range(4))
        for veh_count in num_vehicles:
            for tour_count in num_tours:
                for charger_count in num_chargers:
                    for pause in map(to_param, zip(min_pause_before_tour, max_pause_before_tour)):
                        for tour_length in map(to_param, zip(min_tour_length, max_tour_length)):
                            for charger_capacity in map(to_param, zip(min_charger_capacity, max_charger_capacity)):
                                for soc_consumption in map(to_param, zip(min_soc_consumption, max_soc_consumption)):
                                    for tw_length in map(to_param, zip(min_tw_length, max_tw_length)):
                                        instance_args = dict(
                                            num_tours=tour_count,
                                            seed=next_seed,
                                            num_chargers=charger_count,
                                            avg_energy_price=None,
                                            charger_capacity=charger_capacity,
                                            time_window_length_periods=tw_length,
                                            full_charge_dur=Parameter(full_charge_dur),
                                            free_time_before_tour=pause,
                                            consumption=soc_consumption,
                                            tour_dur_periods=tour_length,
                                            num_vehicles=veh_count,
                                            energy_price_scale=energy_price_scale
                                        )

                                        instance = generate_instance(**instance_args)

                                        for p in instance.periods:
                                            delattr(p, 'pred')
                                            delattr(p, 'succ')

                                        # Write to file
                                        name_components = [prefix] + list(filter(lambda x: len(x)>0, (abbr(k, v) for k, v in instance_args.items()))) \
                                                          + [f'{instance_id}run']
                                        if len(suffix) > 0:
                                            name_components += [suffix]

                                        name = '_'.join(name_components)
                                        instance_dir = output_directory / (name + '.dump.d')
                                        Dump.DumpSchedulingInstance(instance_dir, name, instance, is_discretized=True)
                                        # Create params json
                                        with open(instance_dir / 'parameters.json', 'w') as param_file:
                                            dump_json(instance_args, param_file, cls=Parameter.JSONEncoder)
                                        generated_instances.append(instance)
        for lhs_id, lhs_inst in enumerate(generated_instances):
            for rhs_inst in generated_instances[:lhs_id]:
                assert _instances_are_from_same_seed(lhs_inst, rhs_inst)
        generated_instances.clear()

@cli.command()
@click.pass_context
@click.option('--version', default=1, type=int)
def generate_instance_set(ctx: click.Context, version: int):
    output_directory = ctx.obj['output_directory']

    pause = 8
    full_charge_dur = 8 * PERIOD_LENGTH
    num_vehicles = 4
    num_chargers = 1
    capacity = 2
    num_instances = 5

    if version == 1:
        tw = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10)]
        consumptions = [(0.65, 0.65), (0.75, 0.75), (0.85, 0.85), (0.95, 0.95)]
        tours = [4, 5, 6, 7, 8]
        tour_len = [(6, 8), (8, 10), (10, 12), (6, 10), (8, 12), (6, 12)]
        ep_scale = [1.0, 4.0, 8.0]

    if version == 2:
        tw = [(0, 0), (3, 3), (6, 6), (9, 9)]
        consumptions = [(0.65, 0.65), (0.75, 0.75)]
        tours = [4, 5, 6, 7, 8]
        tour_len = [(6, 6), (10, 10)]
        ep_scale = [1.0, 4.0, 8.0]
        num_instances = 1

    # FEASIBILITY STUDY
    # consumptions = [(0.75, 0.75)]
    # tours = [4]
    # tour_len = [(6, 6)]
    # ep_scale = [1.0]

    for min_time_window, max_time_window in tw:
        for min_consumption, max_consumption in consumptions:
            for num_tours in tours:
                for min_tour_len, max_tour_len in tour_len:
                    for energy_price_scale in ep_scale:
                        ctx.invoke(generate_cli, prefix=f'expl-v{version}', suffix='', seed=f'v{version}',
                                     num_chargers=[num_chargers], num_vehicles=[num_vehicles], num_tours=[num_tours],
                                     num_instances=num_instances, full_charge_dur=full_charge_dur,
                                     min_pause_before_tour=[pause], max_pause_before_tour=[pause],
                                     min_tour_length=[min_tour_len], max_tour_length=[max_tour_len],
                                     min_tw_length=[min_time_window], max_tw_length=[max_time_window],
                                     min_charger_capacity=[capacity], max_charger_capacity=[capacity],
                                     min_soc_consumption=[min_consumption], max_soc_consumption=[max_consumption],
                                   energy_price_scale=energy_price_scale)

if __name__ == '__main__':
    cli()
