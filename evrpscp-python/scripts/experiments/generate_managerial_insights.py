# coding=utf-8
from datetime import datetime

import random
from random import Random
from operator import attrgetter
from typing import List, Optional, Tuple, Union, Any, Callable, NamedTuple
from json import dump as dump_json
from string import ascii_letters
# CLI Stuff
import click
from pathlib import Path
import git

# EVRPSCP
import matplotlib.pyplot as plt
import numpy as np

import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.models import *
from evrpscp.generators.flexible_operations import generate_flexible_tours, calculate_periods_required_for_charging
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper, plot_tours_as_geannt, plot_energy_rate
from funcy import nth, flatten, with_prev, cycle, drop, chain, first
from scipy.interpolate import interp1d

from evrpscp.generators.models.battery import compute_wdf

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
        price_at_time = interp1d(x=list(chain([0], [(timestep*period_dur + period_dur/2) for timestep in range(len(prices))], [len(prices)*period_dur])),
                                 y=list(chain([prices[0]], prices, [prices[-1]])), assume_sorted=True)

    # Generate periods
    periods = []
    for p_id in range(number_of_periods):
        begin, center, end = p_id * period_dur, (p_id + 0.5) * period_dur, (p_id+1) * period_dur
        periods.append(DiscretePeriod(begin=begin, end=end, energyPrice=float(price_at_time(center % PERIODS_PER_DAY*2*PERIOD_LENGTH)) if scale_length else prices[p_id]))


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

def construct_charger(capacity: int, duration: float, battery_size: Optional[float] = None, base_charger: Optional[Charger] = None, **kwargs) -> Charger:
    """
    Constructs a charger based off the Pelletier Fast Charger
    """
    charger_args = dict(id=0, isBaseCharger=False)
    charger_args.update(kwargs)

    if base_charger is None:
        base_charger = Pelletier.FastCharger
    if battery_size is None:
        battery_size = base_charger.chargingFunction.image_upper_bound
    base_charger.capacity = capacity
    # Scale to duration
    scaled_charging_function = base_charger.chargingFunction.scale_domain(new_max=duration).scale_image(new_max=battery_size)
    return Charger(capacity=capacity, chargingFunction=scaled_charging_function,
                   inverseChargingFunction=scaled_charging_function.inverse(), **charger_args)

def generate_tou_rates(end_of_horizon: float, period_length: float = 30, avg_energy_price: Optional[Parameter] = None, energy_price_scale: float = 1.0, generator: Optional[Random] = None):
    ref_rates = Pelletier.WinterRates
    ref_avg_price = sum([x.energyPrice*x.duration/PERIOD_LENGTH for x in ref_rates]) / (ref_rates[-1].end/PERIOD_LENGTH)
    periods = generate_periods(number_of_periods=int(end_of_horizon//period_length), period_dur=period_length, prices=DEFAULT_ENERGY_RATES,
                            reference_average_price=avg_energy_price.generate(rand_gen=generator) if avg_energy_price else ref_avg_price, scale_length=True)
    for p in periods:
        p.energyPrice *= energy_price_scale
    return periods

# min_consumption/max_consumption are given as SoC values
def generate_tours(first_period: DiscretePeriod, num_tours: int, consumption: Parameter, duration: Parameter, time_window: Parameter, free_periods: Parameter, generator: Optional[Random] = None) -> List[DiscreteTour]:
    earliest_next_departure = first_period
    tours = []
    for tour_id in range(num_tours):
        tour_time_before = free_periods.generate(rand_gen=generator)

        _earliest_begin = nth(tour_time_before, earliest_next_departure)
        _dur = duration.generate(rand_gen=generator)
        tw_len = time_window.generate(rand_gen=generator)
        latest_departure = nth(tw_len, _earliest_begin)

        tours.append(
            DiscreteTour(
                id=tour_id, duration=_dur, earliest_departure=_earliest_begin,
                latest_departure=latest_departure,
                consumption=consumption.generate(rand_gen=generator), cost=0.0
            )
        )

        # First period after arrival -> First period where one can leave again
        prev_earliest_nest = earliest_next_departure
        earliest_next_departure = tours[-1].earliest_arrival.succ
        assert earliest_next_departure is nth(_dur + tour_time_before + 1, prev_earliest_nest)

    return tours


def _generate_instance(period_generator: Callable[[], List[DiscretePeriod]], battery_generator: Callable[[], Battery]
                       , charger_generator: Callable[[], List[Charger]],
                       tour_plan_generator: Callable[[List[DiscretePeriod], Battery, List[Charger]], FleetTourPlan]) -> SchedulingInstance:
    periods = period_generator()
    battery = battery_generator()
    chargers = charger_generator()
    tour_plan = tour_plan_generator(periods, battery, chargers)

    # Remove superfluous periods
    latest_arrival = max(max(tour.latest_arrival_time for tour in veh_tour_plan) for veh_tour_plan in tour_plan)
    latest_arrival = next(p for p in periods if latest_arrival in p)
    periods = periods[:periods.index(latest_arrival) + 1]
    periods[-1].succ = None

    param = Parameters(fleetSize=len(tour_plan), battery=battery, max_charges_between_tours=100)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=param, tourPlans=tour_plan)
    return instance

def generate_instance(num_tours: int, seed: Optional[str], num_chargers: int, avg_energy_price: Optional[Parameter],
                      charger_capacity: Parameter, consumption: Parameter, time_window_length_periods: Parameter,
                      full_charge_dur: Parameter, tour_dur_periods: Parameter, free_time_before_tour: Parameter,
                      energy_price_scale: float, num_vehicles: int = 1) -> SchedulingInstance:
    period_length = PERIOD_LENGTH
    latest_end_of_horizon = num_tours * (free_time_before_tour.max + tour_dur_periods.max + 1 + time_window_length_periods.max) * period_length

    seed_generation_engine = Random(seed)
    seed_generator = lambda: Random(seed_generation_engine.random())
    param_generator = seed_generator()

    periods = generate_tou_rates(end_of_horizon=latest_end_of_horizon, period_length=period_length, energy_price_scale=energy_price_scale, generator=seed_generator())
    battery = Pelletier.Battery
    battery.initialCharge = battery.minimumCharge = 0.0
    battery.maximumCharge = 80.0
    #battery = construct_battery(generator=Random(0))
    chargers = construct_charger(capacity=charger_capacity.generate(param_generator), duration=full_charge_dur.generate(param_generator))
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

    instance = SchedulingInstance(periods=periods, chargers=[chargers], param=param, tourPlans=FleetTourPlan(tour_plans, fleetSize=num_vehicles))
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

def _key_value_namer(key_name: str, value: Any) -> str:
    if value is not None:
        return str(value) + (''.join(x[0] for x in key_name.split('_')[:2]) if len(key_name) > 2 else key_name)
    return ''


def _create_name(namer: Callable, delimitter: str = '_', **kwargs):
    prefix, suffix = kwargs.pop('prefix', ''), kwargs.pop('suffix', '')
    # Write to file
    name_components = list(filter(lambda x: len(x) > 0, chain([prefix], (namer(k, v) for k, v in kwargs.items()), [suffix])))

    if any((delimitter in x) for x in name_components):
        raise ValueError(f"Cannot create name from components that have the delimiter {delimitter} in name")
    for _prev, _next in with_prev(sorted(name_components)):
        if _prev == _next:
            raise ValueError(f"Found duplicate name: {_prev}")

    return delimitter.join(name_components)

def _write_instance(instance, output_directory: Path, name: Optional[str] = None, **instance_params):
    _linked_list_connections = {}
    for p in instance.periods:
        _linked_list_connections[p] = (p.pred, p.succ)
        delattr(p, 'pred')
        delattr(p, 'succ')

    if name is None:
        name = _create_name(namer=_key_value_namer, **instance_params)

    instance_params.setdefault('generation_time', datetime.now().isoformat())
    try:
        instance_params.setdefault('commit_hash', git.Repo(search_parent_directories=True).head.object.hexsha)
    except:
        instance_params.setdefault('commit_hash', 'NOT_FOUND')

    instance_dir = output_directory / (name + '.dump.d')
    Dump.DumpSchedulingInstance(instance_dir, name, instance, is_discretized=True)
    # Create params json
    with open(instance_dir / 'parameters.json', 'w') as param_file:
        dump_json(instance_params, param_file, cls=Parameter.JSONEncoder)

    for p in instance.periods:
        pred, succ = _linked_list_connections[p]
        setattr(p, 'pred', pred)
        setattr(p, 'succ', succ)

    return instance_dir


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
                                        if instance_dir.exists():
                                            raise IOError(f'Refusing to overwrite instance in {instance_dir}!')
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
def generate_charger_capacity(ctx: click.Context, version: int):
    output_directory = ctx.obj['output_directory']

    # Parameters
    base_param = dict(num_vehicles = 16, num_fast_chargers = 3, num_tours = 6)
    # 3 Touren pro 24h -> 8h pro Tour inkl. Pause
    # 2h pause, 6h tour
    base_param['tour_len'] = int(round(6*60 / PERIOD_LENGTH))
    # In 24h 2 mal entladen, i.e. 1 Batterieladungen pro 12h
    base_param['consumption'] = round(2 / (base_param['num_tours']), 2)
    num_instances = 50
    # Variables
    time_windows = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    num_fast_chargers = list(range(1, base_param['num_vehicles'] + 1))

    # Take battery from pelletier
    # ACC(D) = 694D^(-0.795)
    battery_price = 5406
    battery = Pelletier.Battery
    battery.minimumCharge = 0.0
    battery.initialCharge = 0.0
    battery.maximumCharge = 45.0
    battery.capacity = battery.maximumCharge - battery.minimumCharge
    # Compute new WDF
    battery.wearCostDensityFunction = compute_wdf(num_intervals=4, capacity=battery.maximumCharge, price=battery_price)

    # Chargers:
    # 50 kWh fast charger -> 120min
    chargers = [
        construct_charger(capacity=base_param['num_fast_chargers'], duration=120, base_charger=Pelletier.FastCharger,
                          battery_size=battery.maximumCharge, id=0)
    ]

    # Construct tours for each vehicle by calling generate_flexible_tours and concatenate them to a schedule
    def tour_plan_generator(periods: List[DiscretePeriod], tw_len: int, num_tours: int, generator: Random):
        tour_plans = []
        tour_id_offset = 0
        for veh in range(base_param['num_vehicles']):
            tours = generate_flexible_tours(periods=periods, num_tours=num_tours,
                                            consumption=base_param['consumption'] * (battery.maximumCharge - battery.minimumCharge),
                                            duration=base_param['tour_len'], time_window_length=tw_len,
                                            free_periods_before_first_tour=4, min_pause=2,
                                            randomize_breaks=True,
                                            generator=generator)
            for pi in tours:
                pi.id += tour_id_offset
            tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
            tour_id_offset += len(tours)
        return FleetTourPlan(tour_plans)

    seed_generator = Random()
    for instance_id in range(num_instances):
        seed = "".join(seed_generator.choices(ascii_letters, k=4))

        # 2 Days, Real World energy prices scaled according to pelletier winter rates
        period_generator = lambda: generate_periods(number_of_periods=PERIODS_PER_DAY * 2, period_dur=PERIOD_LENGTH,
                                                    prices=DEFAULT_ENERGY_RATES,
                                                    reference_average_price=0.2127)

        for charger_capacity in num_fast_chargers:
            chargers[0].capacity = charger_capacity
            for time_window_length in time_windows:
                # Create instance
                instance = _generate_instance(period_generator=period_generator, battery_generator=lambda: battery,
                                              charger_generator=lambda: chargers,
                                              tour_plan_generator=lambda periods, *_: tour_plan_generator(periods=periods,
                                                                                                          tw_len=time_window_length,
                                                                                                          num_tours=base_param['num_tours'],
                                                                                                          generator=Random(seed)))

                for tp in instance.tourPlans:
                    for tour in tp.tours:
                        assert DiscreteTour.FromTour(tour, periods=instance.periods).departure_time_window_length \
                               == time_window_length

                # Write to disk
                _write_instance(instance=instance, output_directory=output_directory,
                                prefix=f"exp2-capacity",
                                seed=seed,
                                tw=time_window_length,
                                charger_capacity=charger_capacity,
                                **base_param)

@cli.command()
@click.pass_context
@click.option('--version', default=1, type=int)
@click.option('--plot', type=bool, is_flag=True)
def generate_energy_price(ctx: click.Context, version: int, plot: bool):
    output_directory = ctx.obj['output_directory']

    # Parameters
    base_param = dict(num_vehicles = 16, num_fast_chargers = 6, num_tours = 6)
    # 3 Touren pro 24h -> 8h pro Tour inkl. Pause
    # 2h pause, 6h tour
    base_param['tour_len'] = int(round(6*60 / PERIOD_LENGTH))
    # In 24h 2 mal entladen, i.e. 1 Batterieladungen pro 12h
    base_param['consumption'] = round(2 / (base_param['num_tours']), 2)
    num_instances = 50
    # Variables
    time_windows = list(range(0, 10+1))
    energy_price_mean_factors = [0.5, 1., 2.]
    energy_price_sigmas = [0.075 * min(energy_price_mean_factors), 0.125 * min(energy_price_mean_factors), 0.25 * min(energy_price_mean_factors)]

    # Take battery from pelletier
    # ACC(D) = 694D^(-0.795)
    battery_price = 5406
    battery = Pelletier.Battery
    battery.minimumCharge = 0.0
    battery.initialCharge = 0.0
    battery.maximumCharge = 45.0
    battery.capacity = battery.maximumCharge - battery.minimumCharge
    # Compute new WDF
    battery.wearCostDensityFunction = compute_wdf(num_intervals=4, capacity=battery.maximumCharge, price=battery_price)

    avg_deg_cost = (battery.wearCostDensityFunction.image_upper_bound - battery.wearCostDensityFunction.image_lower_bound) \
                   / (battery.wearCostDensityFunction.upper_bound - battery.wearCostDensityFunction.lower_bound)

    chargers = [
        construct_charger(capacity=base_param['num_fast_chargers'], duration=120, base_charger=Pelletier.FastCharger,
                          battery_size=battery.maximumCharge, id=0),
        construct_charger(capacity=base_param['num_vehicles'], duration=7.25 * 60,
                          base_charger=Pelletier.SlowCharger, battery_size=battery.maximumCharge, id=1)  # slow charger
    ]

    free_periods_before_first_tour = calculate_periods_required_for_charging(num_vehicles=base_param['num_vehicles'],
                                                                             chargers=chargers,
                                                                             target_soc=base_param['consumption'] * (battery.maximumCharge - battery.minimumCharge),
                                                                             initial_soc=battery.initialCharge,
                                                                             period_length=PERIOD_LENGTH)
    print(f'Min pause at begin: {free_periods_before_first_tour}')

    # Construct tours for each vehicle by calling generate_flexible_tours and concatenate them to a schedule
    def tour_plan_generator(periods: List[DiscretePeriod], tw_len: int, num_tours: int, generator: Random):
        tour_plans = []
        tour_id_offset = 0
        for veh in range(base_param['num_vehicles']):
            tours = generate_flexible_tours(periods=periods, num_tours=num_tours,
                                            consumption=base_param['consumption'] * (battery.maximumCharge - battery.minimumCharge),
                                            duration=base_param['tour_len'], time_window_length=tw_len,
                                            free_periods_before_first_tour=free_periods_before_first_tour, min_pause=2,
                                            randomize_breaks=True,
                                            generator=generator)
            for pi in tours:
                pi.id += tour_id_offset
            tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
            tour_id_offset += len(tours)
        return FleetTourPlan(tour_plans)

    seed_generator = Random()
    total_instances = 0
    for instance_id in range(num_instances):
        seed = "".join(seed_generator.choices(ascii_letters, k=4))
        print(f'\rGenerating {instance_id+1}/{num_instances} ({seed})', end='', flush=True)
        for mean_energy_price_factor in energy_price_mean_factors:
            for sigma_energy_price_factor in energy_price_sigmas:
                # 2 Days, Real World energy prices scaled according to pelletier winter rates
                period_generator = lambda: generate_normally_distributed_tou_rates(
                    periods_in_horizon=2*PERIODS_PER_DAY,
                    period_length=PERIOD_LENGTH,
                    mean_price=mean_energy_price_factor * avg_deg_cost,
                    sigma=sigma_energy_price_factor * avg_deg_cost,
                    generator=Random(seed))

                for time_window_length in time_windows:
                    # Create instance
                    try:
                        instance = _generate_instance(period_generator=period_generator, battery_generator=lambda: battery,
                                                      charger_generator=lambda: chargers,
                                                      tour_plan_generator=lambda periods, *_: tour_plan_generator(periods=periods,
                                                                                                                  tw_len=time_window_length,
                                                                                                                  num_tours=base_param['num_tours'],
                                                                                                                  generator=Random(seed)))
                    except ValueError:
                        print(f'Negative energy price for {sigma_energy_price_factor=}, {mean_energy_price_factor=}')
                        exit(1)
                    assert all(x.energyPrice > 0 for x in instance.periods), f'Negative energy price for {sigma_energy_price_factor=}, {mean_energy_price_factor=}'
                    if plot and time_window_length == time_windows[0]:
                        for tp in instance.tourPlans.schedules:
                            plot_tours_as_geannt([DiscreteTour.FromTour(t, instance.periods) for t in tp.tours],
                                                 periods=instance.periods, veh_id=tp.vehicleID)
                        values = np.array([x.energyPrice for x in instance.periods])
                        calc_deviation = values.std()
                        calc_mean = values.mean()
                        status = f'Sigma: {sigma_energy_price_factor}, Mu: {mean_energy_price_factor}. Calculated sigma: {calc_deviation}, mu: {calc_mean}'
                        plt.title(status)
                        plt.show()
                        plt.title(status)
                        plot_energy_rate(instance.periods)
                        plt.show()
                    # Write to disk
                    _write_instance(instance=instance, output_directory=output_directory,
                                    prefix=f"energy-price-isolated",
                                    seed=seed,
                                    tw=time_window_length,
                                    mean_energy_price_factor=mean_energy_price_factor,
                                    sigma_energy_price_factor=sigma_energy_price_factor,
                                    **base_param)
                    total_instances += 1
    print(f'Done. Generated {num_instances} runs. Total instances: {total_instances}')

@cli.command()
@click.pass_context
def generate_basecase(ctx: click.Context):
    output_directory = ctx.obj['output_directory']

    # Parameters
    base_param = dict(num_vehicles = 16, num_fast_chargers = 6, num_tours = 6)
    # 3 Touren pro 24h -> 8h pro Tour inkl. Pause
    # 2h pause, 6h tour
    base_param['tour_len'] = int(round(6*60 / PERIOD_LENGTH))
    # In 24h 2 mal entladen, i.e. 1 Batterieladungen pro 12h
    base_param['consumption'] = round(2 / (base_param['num_tours']), 2)
    num_instances = 50
    # Variables
    time_windows = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    # Take battery from pelletier
    # ACC(D) = 694D^(-0.795)
    battery_price = 5406
    battery = Pelletier.Battery
    battery.minimumCharge = 0.0
    battery.initialCharge = 0.0
    battery.maximumCharge = 45.0
    battery.capacity = battery.maximumCharge - battery.minimumCharge
    # Compute new WDF
    battery.wearCostDensityFunction = compute_wdf(num_intervals=4, capacity=battery.maximumCharge, price=battery_price)

    # Chargers:
    # 50 kWh fast charger -> 120min
    # TODO More breakpoints here? Such that we have a steeper descend?
    # 7.4 kWh slow charger -> 7.25h
    # Fast and slow charger (one for each vehicle)
    chargers = [
        construct_charger(capacity=base_param['num_fast_chargers'], duration=120, base_charger=Pelletier.FastCharger,
                          battery_size=battery.maximumCharge, id=0), # fast charger
        construct_charger(capacity=base_param['num_vehicles'], duration=7.25*60,
                          base_charger=Pelletier.SlowCharger, battery_size=battery.maximumCharge, id=1) # slow charger
    ]

    def print_stats(tours: List[Tour], chargers: List[Charger], battery: Battery, periods: List[DiscretePeriod]):
        tours = [DiscreteTour.FromTour(tour, periods) for tour in tours]
        num_tours, total_tour_len, total_consumption = len(tours), sum(tour.duration for tour in tours), sum(tour.consumption for tour in tours)
        print("Stats:")
        print(f"{num_tours} Tours over {len(periods)/2} hours: {total_tour_len / 2}h of service -> {total_tour_len/ len(periods):.2%}.")
        print(f"\tTotal consumption: {total_consumption} kWh => {total_consumption/.25} km travelled -> avg speed of {total_consumption/.25 / (total_tour_len/2)} km/h")
        min_charging_time = sum(chargers[0].duration(battery.minimumCharge, tour.consumption) for tour in tours)
        print(f"\tMin charging time: {min_charging_time/60.0:.2f}h ({min_charging_time/((len(periods) - total_tour_len)*PERIOD_LENGTH):.2%} of pause), avg min per tour: ({min_charging_time/len(tours)/60:.2f}h), max: {sum(chargers[-1].duration(battery.minimumCharge, tour.consumption) for tour in tours)/60.0:.2f}.")
        uncovered_periods = []
        for p in periods:
            if not any(tour.latest_departure <= p <= tour.earliest_arrival for tour in tours):
                uncovered_periods.append(p)
        avg_energy_price = sum(p.energyPrice for p in uncovered_periods)/len(uncovered_periods)
        print(f"\tAverage energy price: {avg_energy_price:.2f} => {total_consumption * avg_energy_price:.2f} to recharge,"
              f"average deg cost: {total_consumption * (battery.wearCostDensityFunction.image_upper_bound - battery.wearCostDensityFunction.image_lower_bound) / (battery.wearCostDensityFunction.upper_bound - battery.wearCostDensityFunction.lower_bound)}")

    free_periods_before_first_tour = calculate_periods_required_for_charging(num_vehicles=base_param['num_vehicles'],
                                                                             chargers=chargers,
                                                                             target_soc=base_param['consumption'] * (battery.maximumCharge - battery.minimumCharge),
                                                                             initial_soc=battery.initialCharge,
                                                                             period_length=PERIOD_LENGTH)
    print(f'Min pause at begin: {free_periods_before_first_tour}')

    # Construct tours for each vehicle by calling generate_flexible_tours and concatenate them to a schedule
    def tour_plan_generator(periods: List[DiscretePeriod], tw_len: int, num_tours: int, generator: Random):
        tour_plans = []
        tour_id_offset = 0
        for veh in range(base_param['num_vehicles']):
            tours = generate_flexible_tours(periods=periods, num_tours=num_tours,
                                            consumption=base_param['consumption'] * (battery.maximumCharge - battery.minimumCharge),
                                            duration=base_param['tour_len'], time_window_length=tw_len,
                                            free_periods_before_first_tour=free_periods_before_first_tour, min_pause=2,
                                            randomize_breaks=True,
                                            generator=generator)
            for pi in tours:
                pi.id += tour_id_offset
            tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
            tour_id_offset += len(tours)
        return FleetTourPlan(tour_plans)

    seed_generator = Random()
    for instance_id in range(num_instances):
        seed = "".join(seed_generator.choices(ascii_letters, k=4))

        # 2 Days, Real World energy prices scaled according to pelletier winter rates
        period_generator = lambda: generate_periods(number_of_periods=PERIODS_PER_DAY * 2, period_dur=PERIOD_LENGTH,
                                                    prices=DEFAULT_ENERGY_RATES,
                                                    reference_average_price=0.2127)

        for time_window_length in time_windows:
            # Create instance
            instance = _generate_instance(period_generator=period_generator, battery_generator=lambda: battery,
                                          charger_generator=lambda: chargers,
                                          tour_plan_generator=lambda periods, *_: tour_plan_generator(periods=periods,
                                                                                                      tw_len=time_window_length,
                                                                                                      num_tours=base_param['num_tours'],
                                                                                                      generator=Random(seed)))

            for tp in instance.tourPlans:
                for tour in tp.tours:
                    assert DiscreteTour.FromTour(tour, periods=instance.periods).departure_time_window_length \
                           == time_window_length

            if time_window_length == 0:
                for tp in instance.tourPlans:
                    print(f'Stats for vehicle {tp.vehicleID}:')
                    print_stats(tp.tours, instance.chargers, battery, periods=instance.periods)
                    plot_tours_as_geannt([DiscreteTour.FromTour(t, instance.periods) for t in tp.tours], periods=instance.periods, veh_id=tp.vehicleID)
                    print('\n')
                plt.show()
                plot_energy_rate(instance.periods)
                plt.show()
            # Write to disk
            _write_instance(instance=instance, output_directory=output_directory,
                            prefix=f"exp2-basecase",
                            seed=seed,
                            tw=time_window_length,
                            **base_param)

if __name__ == '__main__':
    cli()
