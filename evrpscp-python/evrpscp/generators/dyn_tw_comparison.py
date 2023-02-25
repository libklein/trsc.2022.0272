# coding=utf-8
import random
from random import Random
from copy import copy, deepcopy
from operator import attrgetter
from typing import List, Optional, Tuple, Union, Any, Dict
from json import dump as dump_json
from scipy.interpolate import interp1d
# CLI Stuff

import click
from pathlib import Path

# EVRPSCP
import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.models import *
from evrpscp.generators.models import *
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper
from funcy import *

PERIODS_PER_DAY=48
PERIOD_LENGTH=24*60/PERIODS_PER_DAY
PERIOD_LENGTH_MINUTES=PERIOD_LENGTH
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

def plot_energy_rates(rates: List[float], instance: DiscretizedInstance = None):
    import pandas as pd
    pd.options.plotting.backend = "plotly"
    from matplotlib import pyplot as plt
    time_df = pd.DataFrame({'price': rates})
    min_price, max_price = min(rates), max(rates)

    if instance:
        for veh, tours in enumerate(instance.tours):
            en_route_col = [min_price for _ in instance.periods]
            for pi in tours:
                start = instance.periods.index(pi.earliest_departure) \
                        + (instance.periods.index(pi.latest_departure) - instance.periods.index(pi.earliest_departure)) // 2
                end = start + pi.duration
                for i in range(start, end + 1):
                    en_route_col[i] = max_price
            time_df[f'veh{veh}'] = en_route_col

    time_df.set_index(pd.date_range('15.01.2021', periods=len(rates), freq=f'{PERIOD_LENGTH_MINUTES}min'), inplace=True)
    fig = time_df.plot(y=['price'] + ([f'veh{k}' for k in instance.vehicles] if instance is not None else []))
    fig.show()

def generate_periods(number_of_periods: int, period_dur: float, prices: List[float],
                     reference_prices: List[float] = None, reference_average_price: float = None,
                     scale_length: bool = False) -> List[DiscretePeriod]:
    if scale_length:
        # Scale prices such that it's length matches the number_of_periods
        # Create a piecewise linear function from the prices
        timestep_size = number_of_periods/len(prices) * period_dur
        price_at_time = interp1d(
            x=[0.] + [(timestep*timestep_size + timestep_size/2) for timestep in range(len(prices))]
              + [len(prices)*timestep_size],
            y=[prices[0], *prices, prices[-1]], assume_sorted=True)

    # Generate periods
    periods = []
    for p_id in range(number_of_periods):
        begin, center, end = p_id * period_dur, (p_id + 0.5) * period_dur, (p_id+1) * period_dur
        periods.append(DiscretePeriod(begin=begin, end=end, energyPrice=float(price_at_time(center)) if scale_length else prices[p_id]))

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

def construct_charger(capacity: int, duration: float, id: int) -> Charger:
    """
    Constructs a charger based off the Pelletier Fast Charger
    """

    base_charger: Charger = Pelletier.FastCharger
    base_charger.capacity = capacity
    # Scale to duration
    scaled_charging_function = base_charger.chargingFunction.scale_domain(new_max=duration)
    return Charger(capacity=capacity, id=id, chargingFunction=scaled_charging_function,
            inverseChargingFunction=scaled_charging_function.inverse(), isBaseCharger=False)

def generate_chargers(capacity: Parameter, duration: Parameter, count: Parameter, intervals: Parameter,
                      generator: Optional[Random] = None):
    num_chargers = count.generate(generator)
    return [construct_charger(capacity=capacity.generate(generator), duration=duration.generate(generator), id=c)
            for c in range(num_chargers)]

def generate_normally_distributed_prices(mean: float, var: float, num_periods: int, generator: Random) -> List[float]:
    return [generator.gauss(mu=mean, sigma=var) for _ in range(num_periods)]

def generate_tou_rates(end_of_horizon: float, generator: Random, period_length: float = 30, avg_energy_price: Optional[float] = None,
                       energy_price_scale: float = 1.0, ep_distribution_method: str = 'real', ep_variance=1.0):
    num_periods = int(end_of_horizon//period_length)
    if ep_distribution_method == 'real':
        prices = DEFAULT_ENERGY_RATES
    else:
        prices = generate_normally_distributed_prices(mean=avg_energy_price, var=ep_variance, num_periods=num_periods, generator=generator)
        assert all(lambda x: x > 0, prices), f'Generated negative prices for mean {avg_energy_price} and var {ep_variance}'

    periods = generate_periods(number_of_periods=num_periods, period_dur=period_length,
                               prices=prices,
                               reference_prices=list(map(attrgetter('energyPrice'), Pelletier.WinterRates)) if avg_energy_price is None else None,
                               reference_average_price=avg_energy_price, scale_length=True)
    for p in periods:
        p.energyPrice *= energy_price_scale

    return periods

# min_consumption/max_consumption are given as SoC values
def generate_tours(first_period: DiscretePeriod, num_tours: int, consumption: Parameter, duration: Parameter, time_window: Parameter, free_periods: Parameter, generator: Optional[Random] = None) -> List[DiscreteTour]:
    prev_arrival = first_period
    tours = []
    for tour_id in range(num_tours):
        tour_time_before = free_periods.generate(rand_gen=generator)

        _tw_center: DiscretePeriod = nth(tour_time_before, iter(prev_arrival))
        _tw_center_pid = int(_tw_center.begin // _tw_center.duration)
        _dur = duration.generate(rand_gen=generator)
        # The last tour should always be static
        _tw_len = time_window.generate(rand_gen=generator) if (tour_id < num_tours - 1) else 1
        _pid_before_begin = max(0, _tw_center_pid - int(_tw_len // 2))
        _pid_after_begin = _tw_center_pid + (_tw_len - (_tw_center_pid - _pid_before_begin))
        assert _pid_after_begin - _pid_before_begin == _tw_len

        _begin_period = nth(_pid_before_begin, first_period)
        _end_period = nth(_pid_after_begin, first_period)

        tours.append(
            DiscreteTour(
                id=tour_id, duration=_dur, earliest_departure=_begin_period,
                latest_departure=_end_period,
                consumption=consumption.generate(rand_gen=generator), cost=0.0
            )
        )

        prev_arrival = nth(_dur + 1, _tw_center)

    return tours

def generate_instance(num_tours: Parameter, seed: Optional[str], num_chargers: Parameter, avg_energy_price: Optional[Parameter],
                      charger_capacity: Parameter, consumption: Parameter, time_window_length_periods: Parameter,
                      full_charge_dur: Parameter, tour_dur_periods: Parameter, free_time_before_tour: Parameter,
                      energy_price_scale: Parameter[float], battery_price_scale: Parameter[float], num_vehicles: Parameter[int],
                      ep_distribution_method: str, ep_variance: Parameter[float],
                      planning_horizon_end: Optional[int] = None) -> SchedulingInstance:
    period_length = PERIOD_LENGTH
    if planning_horizon_end is None:
        latest_end_of_horizon = num_tours.max * (free_time_before_tour.max + tour_dur_periods.max + 1 + time_window_length_periods.max) * period_length
    else:
        latest_end_of_horizon = planning_horizon_end * period_length

    seed_generation_engine = Random(seed)
    seed_generator = lambda: Random(seed_generation_engine.random())
    generators = dict(
        TOU=seed_generator(),
        CHARGER=seed_generator(),
        BATTERY=seed_generator(),
        TOUR_PLANS=seed_generator()
    )

    battery, avg_degradation_cost = generate_battery(capacity=Parameter(80.0), initial_charge=Parameter(0.0), intervals=Parameter(4), battery_price_scale=battery_price_scale,
                               generator=generators['BATTERY'], get_unscaled_avg_deg_cost=True)
    periods = generate_tou_rates(end_of_horizon=latest_end_of_horizon, period_length=period_length,
                                 energy_price_scale=energy_price_scale.generate(generators['TOU']),
                                 ep_distribution_method=ep_distribution_method, ep_variance=ep_variance.generate(generators['TOU']),
                                 avg_energy_price=avg_degradation_cost, generator=generators['TOU'])

    chargers = generate_chargers(capacity=charger_capacity, duration=full_charge_dur, count=num_chargers,
                                  intervals=Parameter(3, 5), generator=generators['BATTERY'])
    # Start with an empty battery
    battery.initialCharge = battery.minimumCharge
    param = Parameters(fleetSize=num_vehicles.generate(seed_generator()), battery=battery, max_charges_between_tours=100)

    tour_plans = []
    tour_id_offset = 0
    latest_arrival = periods[0]
    for veh in range(param.fleetSize):
        veh_generator = Random(generators['TOUR_PLANS'].random())
        tours = generate_tours(num_tours=num_tours.generate(veh_generator),
                               consumption=Parameter(consumption.min * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge,
                                                     consumption.max * (battery.maximumCharge - battery.minimumCharge) + battery.minimumCharge),
                               duration=tour_dur_periods, time_window=time_window_length_periods,
                               free_periods=free_time_before_tour, first_period=periods[0], generator=veh_generator)
        for pi in tours:
            pi.id += tour_id_offset
        latest_arrival = max(latest_arrival, tours[-1].latest_arrival)
        tour_plans.append(TourPlan([t.ToContinuousTour() for t in tours], vehicleID=veh))
        tour_id_offset += len(tours)

    # Remove superfluous periods
    periods = periods[:periods.index(latest_arrival)+1]
    periods[-1].succ = None

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=param,
                                  tourPlans=FleetTourPlan(tour_plans, fleetSize=param.fleetSize))

    return instance

def _instances_are_from_same_seed(lhs: SchedulingInstance, rhs: SchedulingInstance) -> bool:
    """
    Checks whether two instances have been generated from the same seed.
    """
    # Check Battery
    if lhs.param != rhs.param:
        return False
    # Check Periods
    lhs_avg_energy_price = list(map(lambda x: x.energyPrice, lhs.periods))
    rhs_avg_energy_price = list(map(lambda x: x.energyPrice, rhs.periods))
    if abs(sum(lhs_avg_energy_price)/len(lhs_avg_energy_price) - sum(rhs_avg_energy_price)/len(rhs_avg_energy_price) > 0.01):
        return False
    if any(lambda x: abs(x[0].energyPrice-x[1].energyPrice) > 0.01, zip(lhs.periods, rhs.periods)) or len(lhs.periods) != len(rhs.periods):
        return False
    # Check Charger
    for lhs_c, rhs_c in zip(lhs.chargers, rhs.chargers):
        if not lhs_c.equals(rhs_c):
            return False
    # Check Tours
    for lhs_v, rhs_v in zip(lhs.tourPlans, rhs.tourPlans):
        for lhs_t, rhs_t in zip(lhs_v, rhs_v):
            if (lhs_t.duration_time, lhs_t.consumption, lhs_t.cost) != (rhs_t.duration_time, rhs_t.consumption, rhs_t.cost) \
                    or lhs_t.departure_time_window_length < rhs_t.departure_time_window_length\
                    or lhs_t.earliest_departure_time > rhs_t.earliest_departure_time\
                    or lhs_t.latest_departure_time < rhs_t.latest_departure_time:
                    return False
        # Last tour should be equal
        if not lhs_v[-1].equals(rhs_v[-1]):
            return False
    return True

def generate_for_setting(consumption: Parameter[float], num_tours: Parameter[int], tour_dur_periods: Parameter[int],
                         energy_price_scale: Parameter[float], seed = None, **kwargs) -> List[Tuple[Dict, SchedulingInstance]]:
    """
    Generates instances for this problem setting
    """
    static_params = dict(
        free_time_before_tour=Parameter(8),
        full_charge_dur=Parameter(8 * PERIOD_LENGTH),
        num_vehicles=Parameter(4),
        num_chargers=Parameter(1),
        charger_capacity=Parameter(2),
        consumption=consumption,
        num_tours=num_tours,
        tour_dur_periods=tour_dur_periods,
        energy_price_scale=energy_price_scale,
        battery_price_scale=Parameter(1.0),
        seed=seed
    )
    static_params.update(kwargs)

    generated = []

    tw = [(1, 1), (4, 4), (9, 9), (10, 10)]
    max_tw = max(flatten(tw))
    latest_end_of_horizon = num_tours.max * (static_params['free_time_before_tour'].max
                                             + tour_dur_periods.max + 1 + max_tw)
    for min_tw_len, max_tw_len in tw:
        static_params['time_window_length_periods'] = Parameter(min_tw_len, max_tw_len)
        instance = generate_instance(avg_energy_price=None, planning_horizon_end=latest_end_of_horizon,
                                     **static_params)
        generated.append((dict(static_params), instance))

    for id, (_, inst) in enumerate(generated):
        for _, inst2 in generated[:id]:
            assert _instances_are_from_same_seed(inst, inst2)

    return generated


@click.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.option('--suffix', default='')
@click.option('--prefix', default='')
def generate(output_directory: Path, suffix: str, prefix: str):
    def abbr(key_name: str, value: Any) -> str:
        if value is not None:
            val_str = str(value)
            return val_str+(''.join(x[0] for x in key_name.split('_')[:2]))
        return ''

    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    tour_duration_periods = Parameter(6)

    # Instances - Version 4 - 09.05
    # Goal: Influence of charger capacity and speed
    consumptions = [(0.65, 0.65)]
    tours = [6]
    tour_lengths = [(4, 10)]
    ep_scale = [1.0]
    deg_scale = [1.0, 0.25, 0.00001]
    num_instances = 25
    seed_generator = Random(f'v4')
    full_charge_durations = [4 * PERIOD_LENGTH, 2 * PERIOD_LENGTH]
    charger_capacities = [4, 2]
    generate_seed = lambda: ''.join(chr(seed_generator.randint(ord('a'), ord('z'))) for _ in range(4))
    price_distributions = ['gauss', 'real']
    tou_variance = [0.025, 0.05, 0.075, 0.1]

    for run in range(num_instances):
        seed = generate_seed()
        for consumption in map(Parameter, consumptions):
            for num_tours in map(Parameter, tours):
                for tour_len in map(Parameter, tour_lengths):
                    for energy_price_scale in map(Parameter, ep_scale):
                        for full_charge_dur in map(Parameter, full_charge_durations):
                            for charger_capacity in map(Parameter, charger_capacities):
                                for battery_price_scale in map(Parameter, deg_scale):
                                    for ep_variance in map(Parameter, tou_variance):
                                        for price_distribution in price_distributions:
                                            if price_distribution == 'real':
                                                if ep_variance != Parameter(tou_variance[0]):
                                                    continue
                                                ep_variance = Parameter(1.0)
                                            instance_batch = generate_for_setting(consumption=consumption, num_tours=num_tours,
                                                                                  tour_dur_periods=tour_len,
                                                                                  energy_price_scale=energy_price_scale, seed=seed,
                                                                                  full_charge_dur=full_charge_dur,
                                                                                  charger_capacity=charger_capacity,
                                                                                  battery_price_scale=battery_price_scale,
                                                                                  ep_distribution_method=price_distribution,
                                                                                  ep_variance=ep_variance)

                                            for instance_args, instance in instance_batch:
                                                for p in instance.periods:
                                                    delattr(p, 'pred')
                                                    delattr(p, 'succ')

                                                # Write to file
                                                name_components = [prefix] + list(filter(lambda x: len(x) > 0, (abbr(k, v) for k, v in sorted(instance_args.items(), key=lambda x: x[0]))))
                                                if len(suffix) > 0:
                                                    name_components += [suffix]

                                                name = '_'.join(name_components)
                                                instance_dir = output_directory / (name + '.dump.d')
                                                Dump.DumpSchedulingInstance(instance_dir, name, instance, is_discretized=True)
                                                # Create params json
                                                with open(instance_dir / 'parameters.json', 'w') as param_file:
                                                    dump_json(instance_args, param_file, cls=Parameter.JSONEncoder)

if __name__ == '__main__':
    generate()