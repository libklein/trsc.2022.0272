# coding=utf-8
import json
import pickle
import random
from copy import copy, deepcopy
from enum import Enum
from itertools import cycle
from typing import Union, Tuple, List, Dict, Optional, SupportsFloat
from pathlib import Path
from sys import stderr

import pandas
import pandas as pd
import re
from json import load as parse_json

import evrpscp.data.pelletier as Pelletier
from evrpscp.parameters import EPS
from dto_mip import create_discrete_periods
from evrpscp import *
from funcy import first, nth

from matplotlib import pyplot as plt


def plot_energy_rate(periods: List[DiscretePeriod]):
    plt.plot(list(range(len(periods))), [x.energyPrice for x in periods])

def plot_tours_as_geannt(tours: List[DiscreteTour], periods: List[DiscretePeriod], veh_id: int = 0):
    def get_center_period(beg: DiscretePeriod, end: DiscretePeriod):
        count = 0
        next_period = beg
        while next_period.begin != end.begin:
            next_period = next_period.succ
            count += 1

        return nth(count//2, iter(beg))

    x_axis_beg_and_span = [(periods.index(get_center_period(tour.earliest_departure, tour.latest_departure)), tour.duration)
                           for tour in tours]
    plt.broken_barh(xranges=x_axis_beg_and_span, yrange=(veh_id*2, 1))
    #plt.xticks([x/2 for x in range(0, len(periods), 4)])
    every_6_hours = [x for x in range(0, len(periods)+1, 12)]
    plt.xticks(every_6_hours, [f'+{x/2}h' for x in every_6_hours])

def percentage_float_formatter(value: SupportsFloat, num_digits = 2) -> str:
    return f'{{:.{num_digits}%}}'.format(value)

def try_convert(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string

def parse_parameters_from_instance(instance_directory: Path) -> Optional[Dict]:
    for param_filename in ('parameters.json', 'info.json'):
        if (param_path := instance_directory / param_filename).exists():
            with open(param_path, 'r') as param_file:
                return json.load(param_file)

def convert_params_to_strings(param: Dict) -> Dict:
    param = param.copy()
    for key, value in param.items():
        if isinstance(value, list) or isinstance(value, tuple):
            param[key] = '-'.join(map(str, value)) if any(x != value[0] for x in value) else str(value[0])
        elif not isinstance(value, str):
            param[key] = str(value)
    return param


def create_result_df(result_files: List[Path], instance_name_re: re.Pattern,
                     instance_directory: Optional[Path] = None, log_directory: Optional[Path] = None,
                     silence_warning=False, time_limit: Optional[float] = None,
                     return_invalid_solutions=False, convert_params_to_string=False) \
        -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    valid_instances = []
    invalid_instances = []
    for sol_file in result_files:
        if match := instance_name_re.search(str(sol_file.name)):
            param = {key: try_convert(val) for key, val in match.groupdict().items()}
            if instance_directory is not None:
                if (instance_param := parse_parameters_from_instance(instance_directory / sol_file.name)) is not None:
                    param = instance_param
                    param['instance_path'] = instance_directory / sol_file.name
            if convert_params_to_string:
                param = convert_params_to_strings(param)

            param = dict(name=sol_file.name, path=sol_file, **param)

            solution = read_solution(sol_file)
            if solution is None:
                if not silence_warning:
                    print(f'Could not find json solution file for result dump {sol_file}')
                    if log_directory is not None:
                        log_file = (log_directory / sol_file.name) / 'console.log'
                        try:
                            print(f'Last lines of log {log_file}:')
                            print('#################################')
                            with open(log_file, 'r') as log_file_handle:
                                print(''.join(log_file_handle.readlines()[-4:]))
                            print('#################################')
                        except:
                            pass
                if return_invalid_solutions:
                    invalid_instances.append(param)
            else:
                valid_instances.append(dict(**solution, **param))
    df = pd.DataFrame(valid_instances)
    if len(df) > 0:
        if time_limit is not None:
            df['timeout'] = df['Runtime'] >= time_limit
            if not (df['Runtime'] <= time_limit + 60.0).all():
                print("Error: Found instances with longer computation times than allowed:", file=stderr)
                for row in (df[df['Runtime'] > time_limit + 60.0]).itertuples():
                    print(row.name, row.Runtime, file=stderr)
            df['Runtime'].clip(upper=time_limit, inplace=True)
        else:
            df['timeout'] = (df['MIPGap'] > EPS) | (df['MIPGap'] < -EPS)
        df['unsolved'] = (df['timeout']) & (df['MIPGap'] < -EPS)
        df['optimal'] = (df['MIPGap'] < EPS) & (~df['timeout'])
        df['infeasible'] = (~df['timeout']) & (df['MIPGap'] == -1.0)
        df['name'] = df['name'].astype(str)

        df.loc[df.infeasible, 'MIPGap'] = 0.0
        df.loc[df.unsolved, 'MIPGap'] = 1.0

    if return_invalid_solutions:
        return df, pd.DataFrame(invalid_instances)
    else:
        return df

def read_solution(sol_dir: Path, summary=True) -> Optional[Dict]:
    try:
        solution_files = sol_dir.rglob('solution-*.json')
        for solution_file_name in solution_files:
            with open(solution_file_name) as sol_file:
                sol_content = parse_json(sol_file)
                sol_info, sol_details = sol_content['SolutionInfo'], sol_content['ScheduleDetails']
                parse_maybe_infeasible = lambda x: -1.0 if x == 'infeasible' else float(x)
                if summary:
                    return {
                        'Runtime': float(sol_info['Runtime']),
                        'ObjVal': parse_maybe_infeasible(sol_info['ObjVal']),
                        'ObjBound': parse_maybe_infeasible(sol_info['ObjBound']),
                        'MIPGap': parse_maybe_infeasible(sol_info['MIPGap']),
                        'EnergyCost': parse_maybe_infeasible(sol_details['EnergyCost']),
                        'DegradationCost': parse_maybe_infeasible(sol_details['DegradationCost']),
                        'IterCount': float(sol_info['IterCount']),
                        'NodeCount': float(sol_info['NodeCount'])
                    }
                else:
                    return sol_content
    except Exception as e:
        return None


def parse_schedule_from_solution(sol_dir: Path) -> Optional[Tuple[Dict, FleetSchedule, 'Node']]:
    schedule_file = list(sol_dir.rglob('solution-*.pickle'))
    assert len(schedule_file) == 1
    with open(schedule_file[0], 'rb') as schedule_input_stream:
        results = pickle.load(schedule_input_stream)
        return results['result'], results['parsed_solution'], results['bnb_tree']

class TOURatePlan(Enum):
    SUMMER = 'SummerRates'
    WINTER = 'WinterRates'

class TOURateWrapper(list):
    def __init__(self, copied_plan: List[Period] = None, name: str = None):
        super(TOURateWrapper, self).__init__(copied_plan)
        if name is None:
            raise ValueError()
        self.name = name

ScalarOrInterval = Union[float, Tuple[float, float]]

def linearize_charger(charger: Charger, max_soc: float, overestimate=False) -> Charger:
    charge_rate = max(s.slope for s in charger) if overestimate else max_soc/charger.fullChargeDuration
    full_charge_dur = max_soc/charge_rate if overestimate else charger.fullChargeDuration

    new_seg = PiecewiseLinearSegment(0., full_charge_dur, 0., max_soc, charge_rate)
    last_seg = charger.chargingFunction.segments[-1]
    last_seg.lowerBound = new_seg.upperBound
    linearized_charging_function = PiecewiseLinearFunction([
        charger.chargingFunction.segments[0],
        new_seg,
        last_seg
    ])
    # linearized_charging_function.dump_to_console()
    # Create new charger with single segment
    return Charger(charger.capacity, charger.id, linearized_charging_function,
                   linearized_charging_function.inverse(), charger.isBaseCharger)


def scale_battery_price(scale=1.0) -> Battery:
    battery = Pelletier.Battery
    return Battery(battery.capacity, battery.initialCharge, battery.maximumCharge, battery.minimumCharge,
                   wearCostDensityFunction=battery.wearCostDensityFunction.scale_slope(scale))

def generate_randomly(interval: Union[int, float, Tuple[int, int], Tuple[float, float]]) -> Union[int, float]:
    if isinstance(interval, tuple):
        return round(random.uniform(*interval), 2)
    elif isinstance(interval, int):
        return interval
    elif isinstance(interval, float):
        return interval
    else:
        raise ValueError('Invalid parameter type!')


def generate_tour(tour_length: ScalarOrInterval, tour_consumption: ScalarOrInterval, tour_arrival: ScalarOrInterval,
                  tour_departure: ScalarOrInterval, offset: Union[float, int] = 0, tour_id=0, battery_capacity=1.0) \
        -> Tour:
    departure = generate_randomly(tour_departure) + offset
    if tour_arrival is not None:
        arrival = generate_randomly(tour_arrival)
    else:
        arrival = departure + generate_randomly(tour_length)
    # Ensure that min/max duration is set
    arrival = min(max(departure + tour_length[0], arrival), departure + tour_length[1])
    assert arrival >= departure + tour_length[0]
    return Tour(departure=departure,
                arrival=arrival,
                _route=Route(generate_randomly(tour_consumption)*battery_capacity, 0.0, duration=arrival - departure,
                             vertices=['NO_ROUTE_INFO'], id=tour_id))


def generate_tour_plan(number_of_days: int, tours_per_day: int, tour_length: Tuple[ScalarOrInterval],
                       tour_arrival: Tuple[ScalarOrInterval], tour_departure: Tuple[ScalarOrInterval],
                       tour_consumption: Tuple[ScalarOrInterval], vehicle_id: int, battery_capacity: float) -> TourPlan:
    tours = []
    for day in range(number_of_days):
        for tour_id, next_tour_length, next_tour_arrival, next_tour_departure, next_tour_consumption in zip(
                range(tours_per_day),
                cycle(tour_length),
                cycle(tour_arrival),
                cycle(tour_departure),
                cycle(tour_consumption)):
            tours.append(generate_tour(tour_length=next_tour_length,
                                       tour_consumption=next_tour_consumption,
                                       tour_arrival=next_tour_arrival,
                                       tour_departure=next_tour_departure,
                                       tour_id=tour_id + day * tours_per_day + vehicle_id * tours_per_day * number_of_days,
                                       offset=day * 24 * 60,
                                       battery_capacity=battery_capacity))

    tours.sort(key=lambda tour: tour.departure_time)
    # Fix overlapping tours
    removed_tours = []
    for prev_pi, pi in zip(tours, tours[1:]):
        if prev_pi.arrival_time > pi.departure_time:
            print(f'Tours {prev_pi} ({prev_pi.departure_time:.2f}-{prev_pi.arrival_time:.2f}) and {pi} ({pi.departure_time:.2f}-{pi.arrival_time:.2f}) overlap!')
            if prev_pi.duration > pi.duration:
                prev_pi.arrival = pi.departure_time - 60.0
            else:
                prev_pi.arrival = pi.departure_time
                pi.departure += 60.0

            if prev_pi.duration <= 60.0 or pi.duration <= 60.0:
                # Merge the tours
                removed_tours.append(prev_pi)
                pi.departure = prev_pi.departure_time

    for pi in removed_tours:
        tours.remove(pi)

    return TourPlan(tours, vehicleID=vehicle_id)


def scale_tou_plan(period_length: float, number_of_days: int, tou_periods: List[Period]):
    periods_per_hour = 60 / period_length
    if (period_length * number_of_days * 24 * periods_per_hour) % tou_periods[-1].end != 0:
        raise NotImplementedError(f'Cannot fit tour rates to schedule!')
    scaled_plan = deepcopy(tou_periods)
    for day_id in range(1, number_of_days):
        for p in tou_periods:
            shifted_period = copy(p)
            shifted_period.begin = p.begin + 24 * 60 * day_id
            shifted_period.end = p.end + 24 * 60 * day_id
            scaled_plan.append(shifted_period)
    return scaled_plan


def generate_instance(tours_per_day=2,
                      tour_length=((5 * 60, 8 * 60),),
                      tour_arrival=((14 * 60, 16 * 60)),
                      tour_departure=((6 * 60, 10 * 60), (18 * 60, 24 * 60)),
                      tour_consumption=((0.40, 0.47),),
                      fleet_size=6,
                      fast_charger_count=1,
                      number_of_days=3,
                      period_length=30,
                      tou_rates: Union[List[Period], TOURatePlan] = TOURatePlan.SUMMER,
                      seed: str = None,
                      target_consumption=None,
                      battery=Pelletier.Battery) -> SchedulingInstance:
    # Seed the random number generator
    random.seed(seed)

    periods_per_hour = 60 / period_length
    tou_periods = getattr(Pelletier, tou_rates.value) if isinstance(tou_rates, TOURatePlan) else tou_rates
    assert isinstance(tou_periods, list)
    if tou_periods[-1].end != period_length * number_of_days * 24 * periods_per_hour:
        tou_periods = scale_tou_plan(period_length=period_length, number_of_days=number_of_days,
                                     tou_periods=tou_periods)
    # Generate periods
    periods = create_discrete_periods(tou_periods, period_length)
    # Generate Parameters
    param = Parameters(fleetSize=fleet_size, battery=battery, numberOfDays=number_of_days, serviceBegin=0.0,
                       serviceEnd=24 * periods_per_hour, max_charges_between_tours=2)
    # Generate tour plans for each vehicle
    tour_plans = []
    for k in range(fleet_size):
        tour_plans.append(generate_tour_plan(number_of_days=number_of_days,
                                             tours_per_day=tours_per_day,
                                             tour_length=tour_length,
                                             tour_arrival=tour_arrival,
                                             tour_departure=tour_departure,
                                             tour_consumption=tour_consumption,
                                             vehicle_id=k,
                                             battery_capacity=param.battery.capacity))
        # Check tour plan against end of instance
        for pi in tour_plans[-1]:
            if pi.arrival_time >= number_of_days * 24 * 60:
                pi.arrival = (number_of_days * 24 * 60) - (2*period_length)

    if target_consumption:
        # Adjust tour consumptions. Weight tours by their length and assign load accordingly
        total_weight = sum(pi.consumption for veh_tours in tour_plans for pi in veh_tours)
        weights: Dict[Tour, float] = {pi: (pi.consumption / total_weight) for veh_tours in tour_plans for pi in veh_tours}

        for pi, weight in weights.items():
            pi.consumption = round(target_consumption * param.battery.capacity * weight, 2)
            assert pi.consumption <= param.battery.maximumCharge

    # Generate chargers
    chargers = [Pelletier.SlowCharger, Pelletier.FastCharger]
    chargers[0].capacity = fleet_size
    chargers[1].capacity = fast_charger_count

    return SchedulingInstance(periods, chargers, param, FleetTourPlan(tour_plans, fleet_size))
