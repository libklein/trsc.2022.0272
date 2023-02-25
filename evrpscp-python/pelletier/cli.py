# coding=utf-8
import math
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, List

import click
from bidict import bidict

import column_generation.column
import evrpscp
import pelletier.models
from column_generation.output import write_solutions, SolveDetails
from evrpscp import DiscretizedInstance, Dump, DiscretePeriod, DiscreteTour, Charger as EVSPCharger, \
    FleetChargingSchedule
from pelletier.conversion import convert_instance
from pelletier.mip import PelletierMIP


@click.group
def cli():
    pass


@cli.command('convert')
@click.argument('instance', type=click.Path(exists=True, path_type=Path))
@click.argument('output', type=click.Path(exists=False, path_type=Path))
@click.option("--period-length", type=float, default=30.0)
def convert(instance: Path, output: Path, period_length: float):
    dump = Dump.ParseDump(instance)

    pelletier_instance = dump.instance
    for plan in pelletier_instance.tourPlans:
        for tour in plan:
            departure = (tour.latest_departure_time - tour.earliest_departure_time) / 2 + tour.earliest_departure_time
            departure_period_index = departure // period_length
            tour.earliest_departure_time = period_length * departure_period_index
            tour.latest_departure_time = tour.earliest_departure_time + period_length

    dump.DumpSchedulingInstance(output, instance_name=dump.name, instance=pelletier_instance, is_discretized=True)


@cli.command('run')
@click.argument('instance', type=click.Path(exists=True, path_type=Path))
@click.option('--max-num-charges', type=int, default=2)
@click.option('--period-duration', type=float, default=30)
@click.option('-o', '--output-directory', type=click.Path(path_type=Path), default=Path('.'))
@click.option('--gap', type=float, default=1e-4)
def run(instance: Path, max_num_charges: int, period_duration: float, output_directory: Path, gap: float):
    # Parse and discretize
    dump = Dump.ParseDump(instance)
    parsed_instance = dump.instance
    discrete_instance = DiscretizedInstance.DiscretizeInstance(parsed_instance, period_duration)
    # Convert
    pelletier_instance = convert_instance(discrete_instance, max_number_of_charges=max_num_charges)
    # Construct MIP
    pelletier_mip = PelletierMIP(pelletier_instance)
    pelletier_mip._model.parameters.mip.tolerances.mipgap = gap

    sol = pelletier_mip.solve()
    if sol is None:
        print("Failed to find feasible solution")
        schedule = None
    else:
        print(f"Found solution with value {sol.objective_value:.2f} in {sol.solve_details.time:.2f}s")

        # Do nessesary conversion
        cols = construct_columns(pelletier_instance, discrete_instance,
                                 pelletier_mip.soc_values, pelletier_mip.active_x)

        schedule = FleetChargingSchedule(sorted([x.create_vehicle_schedule(discrete_instance) for x in cols.values()],
                                                key=lambda schedule: schedule.vehicleID))
        schedule.calculate_cost(discrete_instance.parameters.battery)

    write_solutions(schedule, pelletier_mip.solve_details, dump.name, output_directory=output_directory)



def construct_vehicle_column(periods: List[DiscretePeriod], vehicle: int,
                             evsp_chargers: List[EVSPCharger], pelletier_chargers: List[pelletier.models.Charger],
                             pelletier_tours: List[pelletier.models.Route], evsp_tours: List[DiscreteTour],
                             soc_values: Dict[pelletier.models.DiscretePeriod, float], battery: evrpscp.Battery,
                             active_x: Dict[pelletier.models.DiscretePeriod, pelletier.models.Charger]) \
        -> column_generation.Column:
    tour_mapper = bidict({r: next(t for t in evsp_tours if t.id == r.id) for r in pelletier_tours})
    charger_mapper = bidict({s: next(f for f in evsp_chargers if f.id == s.id) for s in pelletier_chargers})
    # Default values
    energy_charged: Dict[DiscretePeriod, float] = {p: 0.0 for p in periods}
    degradation_cost: Dict[DiscretePeriod, float] = {p: 0.0 for p in periods}
    charger_usage: Dict[Tuple[DiscretePeriod, EVSPCharger], bool] = {(p, f): False for p, f in
                                                                     product(periods, evsp_chargers)}
    # Set tour departures
    tour_departures: Dict[DiscreteTour, DiscretePeriod] = \
        {next(t for t in evsp_tours if t.id == r.id): r.departure_period for r in pelletier_tours}

    for prev_period, period in zip(periods, periods[1:]):
        delta_soc = soc_values[period] - soc_values[prev_period]
        if delta_soc <= 1e-6:
            continue
        energy_charged[prev_period] = delta_soc * battery.capacity
        degradation_cost[prev_period] = battery.wearCost(soc_values[prev_period] * battery.capacity,
                                                         (soc_values[prev_period] + delta_soc) * battery.capacity)
        charger_usage[prev_period, charger_mapper[active_x[prev_period]]] = True

    col = column_generation.Column(
        energy_charged=energy_charged,
        degradation_cost=degradation_cost,
        charger_usage=charger_usage,
        tour_departures=tour_departures,
        vehicle=vehicle,
        objective=0
    )
    col.objective = col.cost
    return col

def construct_columns(pelletier_instance: pelletier.models.Instance, evsp_instance: DiscretizedInstance,
                      soc_values: Dict[Tuple[pelletier.models.DiscretePeriod, int], float],
                      active_x: Dict[Tuple[pelletier.models.DiscretePeriod, int], pelletier.models.Charger]) -> Dict[int, column_generation.Column]:
    columns = {}
    for k in pelletier_instance.vehicles:
        vehicle_soc = {p: val for (p, veh), val in soc_values.items() if veh == k}
        active_veh_x = {p: val for (p, veh), val in active_x.items() if veh == k}
        columns[k] = construct_vehicle_column(pelletier_instance.periods, vehicle=k,
                                       evsp_chargers=evsp_instance.chargers, pelletier_chargers=pelletier_instance.chargers,
                                       pelletier_tours=pelletier_instance.routes[k], evsp_tours=evsp_instance.tours[k],
                                       soc_values=vehicle_soc, battery=evsp_instance.parameters.battery, active_x=active_veh_x)

    return columns