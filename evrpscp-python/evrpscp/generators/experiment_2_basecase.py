# coding=utf-8
from copy import copy
from dataclasses import dataclass
from math import ceil
from operator import attrgetter
from pathlib import Path
from typing import List, Union, Optional, Tuple
from pprint import pprint as print
from sys import stderr

import click
from dataclasses_json import dataclass_json
from evrpscp.generators.models import *
from evrpscp import *
from evrpscp.models.discretization import DiscretizedInstance
from evrpscp.generators.util import linearize_charger
from random import Random
import evrpscp.data.pelletier as PelletierData

from funcy import nth, chain, keep, cycle, with_prev, flatten, drop, first, dropwhile

PERIOD_LENGTH_MINUTES = 30
PERIODS_PER_HOUR = 2
PERIODS_PER_DAY = 24 * PERIODS_PER_HOUR

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
# Scale by 2
# DEFAULT_ENERGY_RATES = list(map(lambda x: x * 2, DEFAULT_ENERGY_RATES))

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

    time_df.set_index(pd.date_range('15.01.2021', periods=len(instance.periods), freq=f'{PERIOD_LENGTH_MINUTES}min'), inplace=True)
    fig = time_df.plot(y=['price'] + [f'veh{k}' for k in instance.vehicles])
    fig.show()

RUNS = 9
NUM_DAYS = 2
FLEET_SIZE = 6
BATTERY_CAPACITY = 80

@dataclass_json
@dataclass
class InstanceParameters:
    seed: str
    run: int
    fleet_size: int
    number_of_days: int
    scenario: int
    dynamic_tours: bool
    linearized: bool
    consider_capacity: bool

    def __post_init__(self):
        self.TOUR_RAND_GEN = Random(self.seed)
        self.VEHICLE_RAND_GEN = [Random(self.TOUR_RAND_GEN.random()) for _ in range(self.fleet_size)]

    @property
    def instancename(self) -> str:
        return f'exp2-{self.run}-{self.fleet_size}v-{self.number_of_days}d-scenario{self.scenario}-' \
               f'{"dyn" if self.dynamic_tours else "static"}-{"linear" if self.linearized else "pwl"}-' \
               f'{"limited" if self.consider_capacity else "unlimited"}'


def generate_tour(first_period: DiscretePeriod, departure: Parameter, duration: Parameter, consumption: Parameter, tw_factor: float, generator: Random, *args, **kwargs) -> Optional[DiscreteTour]:
    last_period = first(p for p in first_period if p.succ is None)
    _duration = duration.generate(generator)
    _departure = departure.generate(generator)
    earliest_departure = nth(_departure, first_period)
    if not earliest_departure or earliest_departure is last_period:
        earliest_departure = last_period.prev.prev
    assert earliest_departure is not None
    latest_departure = nth(_departure + int(round(tw_factor * _duration)), first_period)

    if latest_departure is None:
        latest_departure = last_period.prev
        _duration = 1
    elif nth(_duration, latest_departure) is None:
        if latest_departure is last_period:
            earliest_departure = earliest_departure.prev
            latest_departure = latest_departure.prev
        # If the tour
        _arrival = last_period
        _duration = first(dropwhile(lambda p: p[1] is not _arrival, enumerate(latest_departure)))[0]

    return DiscreteTour(earliest_departure=earliest_departure, latest_departure=latest_departure,
                        consumption=consumption.generate(generator), duration=_duration, cost=kwargs.get('cost', 0.0), *args, **kwargs)


def generate_schedule_scenario_1(periods: List[DiscretePeriod], dynamic_tours: bool, battery_capacity: float, generator: Random = Random()) -> List[DiscreteTour]:
    num_tours = len(periods) // PERIODS_PER_DAY
    tours = [generate_tour(periods[d * PERIODS_PER_DAY],
                           departure=Parameter(8 * PERIODS_PER_HOUR, 10 * PERIODS_PER_HOUR),
                           duration=Parameter(10*PERIODS_PER_HOUR, 12*PERIODS_PER_HOUR),
                           consumption=Parameter(0.85 * battery_capacity, 0.95 * battery_capacity),
                           tw_factor=0.0, generator=generator, id=d
                           ) for d in range(num_tours)]
    if dynamic_tours:
        for t in tours:
            t.latest_departure = t.earliest_departure.succ

    return tours

def generate_schedule_scenario_2(periods: List[DiscretePeriod], dynamic_tours: bool, battery_capacity: float, generator: Random = Random()) -> List[DiscreteTour]:
    TOUR_DURATIONS = [
        Parameter(3 * PERIODS_PER_HOUR, 4*PERIODS_PER_HOUR),
        Parameter(4 * PERIODS_PER_HOUR, 5 * PERIODS_PER_HOUR),
        Parameter(4 * PERIODS_PER_HOUR, 5 * PERIODS_PER_HOUR),
        Parameter(5 * PERIODS_PER_HOUR, 5 * PERIODS_PER_HOUR),
        Parameter(3 * PERIODS_PER_HOUR, 4 * PERIODS_PER_HOUR)
    ]
    TOUR_CONSUMPTIONS = [
        Parameter(0.5 * battery_capacity, 0.55 * battery_capacity),
        Parameter(0.40 * battery_capacity, 0.50 * battery_capacity),
        Parameter(0.40 * battery_capacity, 0.50 * battery_capacity),
        Parameter(0.40 * battery_capacity, 0.50 * battery_capacity),
        Parameter(0.40 * battery_capacity, 0.50 * battery_capacity)
    ]
    TOUR_DEPARTURES = [
        Parameter(7*PERIODS_PER_HOUR, 7*PERIODS_PER_HOUR),
        Parameter(13*PERIODS_PER_HOUR, 14*PERIODS_PER_HOUR),
        Parameter(21*PERIODS_PER_HOUR, 22*PERIODS_PER_HOUR),
        Parameter((24 + 8) * PERIODS_PER_HOUR, (24 + 9) * PERIODS_PER_HOUR),
        Parameter((24 + 15) * PERIODS_PER_HOUR, (24 + 17) * PERIODS_PER_HOUR)
    ]
    tours = []
    next_id = 0
    for dep, dur, cons in zip(TOUR_DEPARTURES, TOUR_DURATIONS, TOUR_CONSUMPTIONS):
        if (next_tour := generate_tour(periods[0], departure=copy(dep), duration=copy(dur),
                                      consumption=cons, tw_factor=0.5 if dynamic_tours else 0.0, generator=generator, id=next_id)) is not None:
            tours.append(next_tour)
            next_id += 1
    return tours

def merge_tours(candidate: DiscreteTour, target: DiscreteTour, battery_capacity: float = 10000) -> DiscreteTour:
    duration = candidate.duration + target.duration
    tw_length = candidate.departure_time_window_length + target.departure_time_window_length
    return DiscreteTour(id=candidate.id, duration=duration, earliest_departure=candidate.earliest_departure,
                        latest_departure=nth(tw_length, candidate.earliest_departure),
                        consumption=min(battery_capacity, candidate.consumption + target.consumption),
                        cost=candidate.cost + target.cost)

def generate_schedule_scenario_3(periods: List[DiscretePeriod], dynamic_tours: bool, battery_capacity: float,
                                     generator: Random = Random()) -> List[DiscreteTour]:
    TOUR_DURATION = Parameter(2 * PERIODS_PER_HOUR, 3*PERIODS_PER_HOUR)
    TOUR_CONSUMPTION = Parameter(0.25 * battery_capacity, 0.30 * battery_capacity)
    TOUR_DEPARTURES = [
        Parameter(4*PERIODS_PER_HOUR, 4*PERIODS_PER_HOUR),
        Parameter(8*PERIODS_PER_HOUR, 9*PERIODS_PER_HOUR),
        Parameter(13*PERIODS_PER_HOUR, 14*PERIODS_PER_HOUR),
        Parameter(18*PERIODS_PER_HOUR, 18*PERIODS_PER_HOUR),
        Parameter(22*PERIODS_PER_HOUR, 22*PERIODS_PER_HOUR)
    ]
    num_days = len(periods) // PERIODS_PER_DAY
    tours = []
    for day in range(num_days):
        first_period_of_day = periods[day*PERIODS_PER_DAY]
        daily_tours = [
            generate_tour(first_period_of_day, departure=copy(TOUR_DEPARTURES[ti]), duration=copy(TOUR_DURATION),
                          consumption=TOUR_CONSUMPTION, tw_factor=0.5 if dynamic_tours else 0.0, generator=generator, id=3*day + ti)
            for ti in range(len(TOUR_DEPARTURES))
        ]
        tours.extend(keep(daily_tours))
    return tours

def generate_vehicle_tour_plan_exp2(scenario: int, *args, **kwargs) -> List[DiscreteTour]:
    if scenario == 1:
        return generate_schedule_scenario_1(*args, **kwargs)
    elif scenario == 2:
        return generate_schedule_scenario_2(*args, **kwargs)
    elif scenario == 3:
        return generate_schedule_scenario_3(*args, **kwargs)
    elif scenario == 4:
        return generate_schedule_scenario_4(*args, **kwargs)
    else:
        raise NotImplementedError

def generate_battery_exp2(capacity: float) -> Battery:
    scale = 1
    deg_rate_from = [
        (0.25, 0.48 * scale),
        (0.5, 0.52 * scale),
        (0.75, 0.58 * scale),
        (1.0, 0.79 * scale)
    ]

    segments = []
    prev_ub, prev_img_ub = 0.0, 0.0
    for next_ub, slope in deg_rate_from:
        segments.append(PiecewiseLinearSegment(
            lowerBound=prev_ub, upperBound=next_ub*capacity,
            imageLowerBound=prev_img_ub, imageUpperBound=prev_img_ub + (next_ub*capacity-prev_ub)*slope,
            slope=slope
        ))
        prev_ub, prev_img_ub = segments[-1].upperBound, segments[-1].imageUpperBound
    return Battery(capacity=capacity, initialCharge=0.0, maximumCharge=capacity, minimumCharge=0.0,
                   wearCostDensityFunction=PiecewiseLinearFunction.CreatePWL(segments))

def generate_charging_infrastructure_exp2(battery_capacity: float, fleet_size: int, linearize: bool, capacity: List[int]) -> List[Charger]:
    """
    n (mode 2) base chargers: 240V, 10 kW
    2 mode 3 chargers: 240V, 19.2 kW
    0 mode 4 charger: DC, 80 kW
    """
    slow_charger: Charger = PelletierData.SlowCharger
    # Scale fast charger such that it needs 330 minutes for full charge
    fast_phi: PiecewiseLinearFunction = PelletierData.FastCharger.chargingFunction.scale_slope(PelletierData.FastCharger.fullChargeDuration/330.0, scale_image=False)
    chargers = [
        Charger(100, 0, chargingFunction=slow_charger.chargingFunction,
                inverseChargingFunction=slow_charger.inverseChargingFunction, isBaseCharger=True),
        Charger(capacity[0], 1, chargingFunction=fast_phi,
                inverseChargingFunction=fast_phi.inverse(), isBaseCharger=False),
    ]
    if linearize:
        return list(map(lambda f: linearize_charger(f, battery_capacity), chargers))
    else:
        return chargers

def generate_periods_exp2(number_of_days: int, desired_prices: List[float] = DEFAULT_ENERGY_RATES, reference_prices: List[float] = None, reference_average_price: float = None) -> List[DiscretePeriod]:
    """
    Generation methodology:
    * Calculate average of reference prices and desired prices
    * Scale desired prices such that average price matches
    """
    periods_in_horizon = PERIODS_PER_DAY*number_of_days
    period_length = PERIOD_LENGTH_MINUTES

    # TODO Refactor
    #   * Create an iterator over prices
    #   * If avg price is given, cylce(avg_price)
    #   * Then for each day draw PERIODS_PER_DAY prices

    periods = []
    for d in range(number_of_days):
        if reference_prices is not None or reference_average_price is not None:
            if reference_average_price is None:
                todays_ref_price = reference_prices if len(reference_prices) <= PERIODS_PER_DAY else reference_prices[PERIODS_PER_DAY*d:PERIODS_PER_DAY*(d+1)]
                reference_average_price = sum(todays_ref_price) / len(todays_ref_price) if reference_average_price \
                                                                                           is None else reference_average_price
        else:
            reference_average_price = sum(map(attrgetter('energyPrice'), PelletierData.WinterRates)) / len(PelletierData.WinterRates)

        todays_desired_prices = desired_prices if len(desired_prices) <= PERIODS_PER_DAY else desired_prices[PERIODS_PER_DAY*d:PERIODS_PER_DAY*(d+1)]

        avg_price_of_desired_rates = sum(todays_desired_prices) / len(todays_desired_prices)

        scaling_factor = reference_average_price / avg_price_of_desired_rates
        scaled_energy_prices = [round(scaling_factor * price, 6) for price in desired_prices]

        assert periods_in_horizon % len(scaled_energy_prices) == 0

        periods.extend(DiscretePeriod(begin=period_length*i + d*PERIODS_PER_DAY*PERIOD_LENGTH_MINUTES,
                                      end=period_length*(i+1) + d*PERIODS_PER_DAY*PERIOD_LENGTH_MINUTES,
                                      energyPrice=price) for i, price in zip(range(PERIODS_PER_DAY), cycle(scaled_energy_prices)))

    for cur, prev in drop(1, with_prev(periods)):
        prev.succ = cur
        cur.prev = prev

    return periods

def generate_instance(generation_param: InstanceParameters) -> SchedulingInstance:
    fleet_size = generation_param.fleet_size
    number_of_days = generation_param.number_of_days
    linearize = generation_param.linearized
    dynamic_tours = generation_param.dynamic_tours
    scenario = generation_param.scenario
    consider_capacity = generation_param.consider_capacity

    battery = generate_battery_exp2(capacity=80.0)
    periods = generate_periods_exp2(number_of_days=number_of_days, desired_prices=DEFAULT_ENERGY_RATES)
    chargers = generate_charging_infrastructure_exp2(battery_capacity=battery.capacity, fleet_size=fleet_size,
                                                     linearize=linearize, capacity=[2] if consider_capacity else [100])

    vehicle_tours = FleetTourPlan(
        [TourPlan(tours=[t.ToContinuousTour() for t in generate_vehicle_tour_plan_exp2(periods=periods,
                                                                                       scenario=scenario,
                                                                                       dynamic_tours=dynamic_tours,
                                                                                       battery_capacity=battery.capacity,
                                                                                       generator=generator)],
                  vehicleID=v_id)
         for v_id, generator in enumerate(generation_param.VEHICLE_RAND_GEN)])
    next_id = 0
    for vt in vehicle_tours:
        for t in vt:
            t.id = next_id
            next_id += 1

    parameters = Parameters(fleetSize=fleet_size, battery=battery, max_charges_between_tours=100)

    instance = SchedulingInstance(periods=periods, chargers=chargers, param=parameters, tourPlans=vehicle_tours)
    return instance


def check_feasibility_mip(instance: SchedulingInstance) -> bool:
    from dyn_tour_mip import Solver
    mip_solver = Solver(instance=instance, time_limit=120, threads=0)
    mip_solver.model.cplex.display = 0
    mip_solver.model.cplex.log_output = False
    mip_solver.model.cplex.parameters.mip.limits.solutions = 1
    solution, model = mip_solver.solve()
    if not model.cplex.solution and not model.cplex.solve_details.has_hit_limit():
        return False
    return True

def check_feasibility_col_gen(instance: SchedulingInstance) -> bool:
    from column_generation import Solver
    from tempfile import TemporaryDirectory
    solve_time_limit = 20
    with TemporaryDirectory(dir='/tmp') as tmp_dir:
        solver = Solver(instance=instance, time_limit=solve_time_limit, dynamic_tours=True, output_dir=tmp_dir, log_dir=tmp_dir, profile=True)
    sol, details = solver.solve()
    return sol is not None or details['Runtime'] >= solve_time_limit


def inclusive_range(start: Union[int, float], end: Union[int, float], step: Union[int, float] = 1):
    _next = start
    while _next <= end:
        yield _next
        _next += step

@click.command()
@click.option('-o', '--output-directory', default=Path('.'),
              type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
def generate_experiment_2(output_directory: Path):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    for scenario in (2,):
        run = 0
        seed_generator = Random(run)
        while run < RUNS:
            seed = str(seed_generator.random())
            run_instances = generate_run_instances(seed=seed, run=run, scenario=scenario)
            if run_instances is None:
                print(f'Run {run} of scenario {scenario} is infeasible! Seed: {seed}. Retrying...', stream=stderr)
                continue
            #for inst in run_instances:
            #    plot_energy_rates(DEFAULT_ENERGY_RATES, instance=DiscretizedInstance.DiscretizeInstance(inst[0]))

            #exit(0)

            for inst, gen_param in run_instances:
                inst_output_dir = (output_directory / (gen_param.instancename + '.dump.d'))

                for p in inst.periods:
                    delattr(p, 'pred')
                    delattr(p, 'succ')

                Dump.DumpSchedulingInstance(directory=inst_output_dir, instance_name=gen_param.instancename,
                                            instance=inst, is_discretized=True)
                with open(str(inst_output_dir / 'info.json'), 'w') as param_file:
                    param_file.write(gen_param.to_json())

            run += 1


def generate_run_instances(seed: str, run: int, scenario: int) -> Optional[List[Tuple[SchedulingInstance, InstanceParameters]]]:
    instances = []
    for dynamic_tours, consider_capacity, non_linear in (
            (False, False, False), # A
            (True, False, False), # B
            (True, True, False), # C
            (True, True, True),  # D
            (False, True, False),  # E
            (False, True, True)  # F
    ):
        gen_param = InstanceParameters(
            seed=seed,
            run=run,
            fleet_size=FLEET_SIZE,
            number_of_days=NUM_DAYS,
            dynamic_tours=dynamic_tours,
            linearized=not non_linear,
            scenario=scenario,
            consider_capacity=consider_capacity
        )

        inst = generate_instance(gen_param)
        if consider_capacity and not dynamic_tours:
            try:
                #plot_energy_rates(DEFAULT_ENERGY_RATES, DiscretizedInstance.DiscretizeInstance(inst))
                #exit(0)
                if not check_feasibility_col_gen(inst):
                    print(f'Instance {gen_param.instancename} is infeasible!')
                    return None
            except:
                raise RuntimeError(f'Col Gen threw an exception! Instance data: {gen_param.to_json()}')

        instances.append((inst, gen_param))
    return instances

if __name__ == '__main__':
    generate_experiment_2()
