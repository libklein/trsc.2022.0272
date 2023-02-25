# coding=utf-8
from pathlib import Path

from evrpscp.models import Battery, Tour, Route, Parameters, SchedulingInstance, FleetTourPlan, TourPlan
from .common import parseBattery, parseCharger, parseRates

# Parses tour plans from pelletier instance
def parseFleetSchedule(instance: Path, battery: Battery, periodLength = 30.0) -> FleetTourPlan:
    if not instance.is_file():
        raise ValueError(f'Cannot open {instance}! It is not a file')
    """ Routen dann eine leere Zeile dann Routen fürs nächste ECV """
    vehicle_plans = [list()]
    next_tour_id = 0
    with open(instance, 'r') as _tour_file:
        for _line in _tour_file.readlines()[:-1]:
            if _line == '\n':
                vehicle_plans.append([])
                continue

            departure_period, arrival_period, consumption = tuple(map(float, _line.split(' ')))
            duration = (arrival_period - departure_period) * periodLength
            # Convert period id to time
            vehicle_plans[-1].append(Tour(
                id=next_tour_id,
                duration_time=duration,
                earliest_departure_time=(departure_period*periodLength),
                latest_departure_time=(departure_period*periodLength) + 1.0,
                consumption=battery.capacity*consumption,
                cost=0.0
            ))
            next_tour_id += 1

    # Construct FleetTourPlan
    return FleetTourPlan(schedules=[
        TourPlan(tours=x, vehicleID=i) for i, x in enumerate(vehicle_plans)
    ], fleetSize=len(vehicle_plans))

# Parses pelletier base case instance from json files in specified directory
def PelletierBaseCase(path: Path, fleetSize = 5):
    if not path.is_dir():
        raise ValueError(f'Path {path} is not a directory!')
    battery = parseBattery(path / 'Battery.json')
    # TODO parse chargers by name
    fast_charger = parseCharger(path / 'FastCharger.json')
    slow_charger = parseCharger(path / 'SlowCharger.json')
    periods = parseRates(path / f'Rates.json', days=3)
    param = Parameters(fleetSize=fleetSize, battery=battery)

    return SchedulingInstance(periods=periods, chargers=[fast_charger, slow_charger], param=param)


def parsePelletier(instance_path: Path, param_path: Path) -> SchedulingInstance:
    base_inst = PelletierBaseCase(param_path)
    fleet_schedule = parseFleetSchedule(instance_path, battery=base_inst.param.battery)
    base_inst.tourPlans = fleet_schedule
    base_inst.param.fleetSize = len(base_inst.tourPlans)
    return base_inst

