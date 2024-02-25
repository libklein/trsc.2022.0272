# coding=utf-8
import sys
from datetime import datetime

import random
from random import Random
from typing import List, Optional, Tuple, Union, Any, Callable, NamedTuple
from string import ascii_letters
from pathlib import Path

import matplotlib.pyplot as plt

import evrpscp.data.pelletier as Pelletier
from evrpscp import *
from evrpscp.generators.models import *
from evrpscp.generators.flexible_operations import generate_flexible_tours, calculate_periods_required_for_charging
from evrpscp.generators.util import generate_instance, TOURatePlan, TOURateWrapper, plot_tours_as_geannt, \
    plot_energy_rate
from funcy import nth, flatten, with_prev, cycle, drop, chain, first
from scipy.interpolate import interp1d

from evrpscp.generators.models.battery import compute_wdf
from evrpscp.generators.models.charger import simulate_charger
from evrpscp.models.charger import create_charger

PERIODS_PER_DAY = 48
PERIOD_LENGTH = 24 * 60 / PERIODS_PER_DAY
# Hourly rates. Taken from
# https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=GRAPH&areaType=BZN&atch=false&dateTime.dateTime=15.01.2021+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A83F!BZN|10Y1001A1001A82H&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
# and https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=GRAPH&areaType=BZN&atch=false&dateTime.dateTime=16.01.2021+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A83F!BZN|10Y1001A1001A82H&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
HOURLY_DEFAULT_ENERGY_RATES = [58.41, 55.65, 52.16, 51.18, 49.51, 48.52, 48.74, 51.43, 61.16, 64.75, 63.77, 62.13,
                               55.01, 50.66, 49.48, 50.65, 57.7, 65.01, 66.4, 63.76, 57.47, 52.98, 53.2, 50.56, 48.98,
                               48.66, 47.7, 46.6, 48, 51.52, 62.97, 77.94, 94.07, 97.76, 90.69, 86.97, 79.25, 76.64,
                               77.14, 76.98, 82, 99.5, 98.02, 95.01, 77.2, 67.18, 63.19, 55.42]
# Interpolate
DEFAULT_ENERGY_RATES = []
for rate, prev_rate in with_prev(HOURLY_DEFAULT_ENERGY_RATES, fill=HOURLY_DEFAULT_ENERGY_RATES[0]):
    DEFAULT_ENERGY_RATES.append(rate)
    DEFAULT_ENERGY_RATES.append((rate + prev_rate) / 2)
assert len(DEFAULT_ENERGY_RATES) == PERIODS_PER_DAY * 2


def generate_periods(number_of_periods: int, period_dur: float, prices: List[float],
                     reference_prices: List[float] = None, reference_average_price: float = None,
                     scale_length: bool = False) -> List[DiscretePeriod]:
    """
    Generates a list of periods based off the given prices. The prices are interpolated to the given number of periods.
    :param number_of_periods: The number of periods to generate
    :param period_dur: The duration of each period
    :param prices: The prices to interpolate
    :param reference_prices: The prices to use as a reference. If specified, the prices will be scaled such that their average matches the average of the reference prices.
    :param reference_average_price: The average price to use as a reference. Can be provided instead of reference_prices.
    :param scale_length: Whether to scale the prices list to match the number of periods by interpolation.
    """
    if scale_length:
        # Scale prices such that it's length matches the number_of_periods
        # Create a piecewise linear function from the prices
        price_at_time = interp1d(x=list(
            chain([0], [(timestep * period_dur + period_dur / 2) for timestep in range(len(prices))],
                  [len(prices) * period_dur])),
            y=list(chain([prices[0]], prices, [prices[-1]])), assume_sorted=True)

    # Generate periods
    periods = []
    for p_id in range(number_of_periods):
        begin, center, end = p_id * period_dur, (p_id + 0.5) * period_dur, (p_id + 1) * period_dur
        periods.append(DiscretePeriod(begin=begin, end=end, energyPrice=float(
            price_at_time(center % PERIODS_PER_DAY * 2 * PERIOD_LENGTH)) if scale_length else prices[p_id]))

    # Determine scaling factor - used to ensure that the average price corresponds to the average price of the reference prices
    if reference_prices is not None or reference_average_price is not None:
        reference_average_price = sum(reference_prices) / len(
            reference_prices) if reference_average_price is None else reference_average_price
        scaling_factor = reference_average_price / (sum(map(lambda x: x.energyPrice, periods)) / len(periods))

        for p in periods:
            p.energyPrice *= scaling_factor

    for cur, prev in drop(1, with_prev(periods)):
        prev.succ = cur
        cur.prev = prev

    return periods


def _write_instance(instance, output_path: Path):
    # Disconnect periods to allow recursive serialization
    _linked_list_connections = {}
    for p in instance.periods:
        _linked_list_connections[p] = (p.pred, p.succ)
        delattr(p, 'pred')
        delattr(p, 'succ')

    Dump.DumpSchedulingInstance(output_path, output_path.name.replace('.dump.d', ''), instance, is_discretized=True)

    # Reconnect periods
    for p in instance.periods:
        pred, succ = _linked_list_connections[p]
        setattr(p, 'pred', pred)
        setattr(p, 'succ', succ)


def create_battery():
    battery_capacity = 80.0  # kWh
    # Compute WDF with 4 intervals and a price of 20000 €
    wdf = compute_wdf(num_intervals=4, capacity=battery_capacity, price=20_000.0)
    battery = Battery(capacity=battery_capacity, initialCharge=0.0, minimumCharge=0.0,
                      maximumCharge=battery_capacity, wearCostDensityFunction=wdf)
    return battery


def create_chargers(battery: Battery) -> List[Charger]:
    assert 50.0 < battery.capacity
    inverse_charging_function = PiecewiseLinearFunction.CreatePWLFromSlopeAndUB([
        # ub, slope (1/[kWh/t])
        (10.0, 1. / 2.0),
        (30.0, 1. / 1.0),
        (50.0, 1. / 0.5),
        (battery.capacity, 1. / 0.25)
    ])
    return [
        Charger(capacity=1, id=0, chargingFunction=inverse_charging_function.inverse(),
                inverseChargingFunction=inverse_charging_function,
                isBaseCharger=False)
    ]


def create_tour(id: int, start_time: float, time_window_lenght: float, duration: float, consumption: float,
                cost: float = 0):
    return Tour(id=id, earliest_departure_time=start_time - time_window_lenght / 2,
                latest_departure_time=start_time + time_window_lenght / 2,
                consumption=consumption, duration_time=duration, cost=cost)


def create_tour_plans(battery_capacity: float):
    return [
        TourPlan(
            [
                create_tour(id=0, start_time=60. * 2, time_window_lenght=60. * 2, duration=60. * 4,
                            consumption=battery_capacity / 2.),
                create_tour(id=1, start_time=60. * 10, time_window_lenght=60., duration=90.,
                            consumption=battery_capacity / 2.),
            ], 0),

        TourPlan(
            [
                create_tour(id=0, start_time=60. * 3, time_window_lenght=60. * 3, duration=60. * 4,
                            consumption=battery_capacity / 2.),
                create_tour(id=1, start_time=60. * 9, time_window_lenght=60., duration=90.,
                            consumption=battery_capacity / 2.),
            ], 1),

        TourPlan(
            [
                create_tour(id=0, start_time=30. * 5, time_window_lenght=30. * 3, duration=60. * 4,
                            consumption=battery_capacity / 2.),
                create_tour(id=1, start_time=60. * 10, time_window_lenght=30., duration=90.,
                            consumption=battery_capacity / 2.),
            ], 2)
    ]


def generate(output_path: Path):
    fleet_size = 3
    # Create planning horizon
    period_length = 30  # A period is 30 minutes long
    planning_horizon_length = 12 * 60  # 12 hours
    assert planning_horizon_length % period_length == 0
    periods = generate_periods(planning_horizon_length // period_length, period_length, DEFAULT_ENERGY_RATES,
                               scale_length=True)
    # Create battery
    battery = create_battery()
    # Create parameters
    parameters = Parameters(fleetSize=fleet_size, battery=battery)
    # Create chargers
    chargers = create_chargers(battery)
    # Create tours
    tour_plans = FleetTourPlan(create_tour_plans(battery.capacity), fleetSize=fleet_size)
    # Create instance
    instance = SchedulingInstance(periods=periods, chargers=chargers, param=parameters,
                                  tourPlans=tour_plans)

    # Ensure validity of the discrete instance
    DiscretizedInstance.DiscretizeInstance(instance)

    _write_instance(instance=instance, output_path=output_path)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output-path-name>")
        sys.exit(1)

    path = Path(sys.argv[1])
    generate(path)

if __name__ == '__main__':
    main()
