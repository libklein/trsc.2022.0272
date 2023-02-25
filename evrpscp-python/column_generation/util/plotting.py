# coding=utf-8
from typing import Optional, Dict, Iterable, List

from funcy import first

from column_generation import Column
from evrpscp import Charger
from terminaltables import AsciiTable
from dataclasses import dataclass

def plot_periods(periods) -> str:
    rows = [('Begin', 'End', 'Energy Price')]
    rows += [(p.begin, p.end, p.energyPrice) for p in periods]
    return str(AsciiTable(rows).table)


def plot_capacity_duals(instance, capacity_duals):
    rows = [['Period'] + [str(c) for c in instance.chargers]]
    for p in instance.periods:
        rows.append([str(p)] + [capacity_duals[p, f] for f in instance.chargers])
    return str(AsciiTable(rows).table)

def plot_vehicle_schedule_comparison(lhs_sched: 'VehicleChargingSchedule', rhs_sched: 'VehicleChargingSchedule', instance, coverage_dual, capacity_duals, hide_both_empty=True):
    next_lhs_op_id = 0
    next_rhs_op_id = 0

    def get_period_of(op):
        for p in instance.periods:
            if p.begin <= op.begin.begin < p.end:
                return p
        return None

    def get_dual(op, period):
        if hasattr(op, 'charger'):
            return capacity_duals.get((period, op.charger), 0)
        return None

    rows = [('Begin', 'Price', 'LHS Dual', 'LHS Ops', 'RHS Dual' , 'RHS Ops')]

    # TODO Print stats, i.e., charges between tours and number of partial periods

    @dataclass
    class Activity:
        lhs_op: 'Operation' = None
        rhs_op: 'Operation' = None

    activities = {p: Activity() for p in instance.periods}

    for lhs_op in lhs_sched:
        activities[get_period_of(lhs_op)].lhs_op = lhs_op
    for rhs_op in rhs_sched:
        activities[get_period_of(rhs_op)].rhs_op = rhs_op

    # Create rows
    for p, activity in activities.items():
        if not hide_both_empty or activity.lhs_op is not None or activity.rhs_op is not None:
            rows.append((p.begin, p.energyPrice,
                         get_dual(activity.lhs_op, p), activity.lhs_op,
                         get_dual(activity.rhs_op, p), activity.rhs_op))

    rows = list(map(lambda row: list(map(lambda x: '' if x is None else str(x), row)), rows))

    print(AsciiTable(rows).table)

    print(AsciiTable([
        ['Field', 'LHS Value', 'RHS Value'],
        ['Total cost', lhs_sched.cost, rhs_sched.cost],
        ['Energy cost', lhs_sched.energy_cost, rhs_sched.energy_cost],
        ['Degradation cost', lhs_sched.degradation_cost, rhs_sched.degradation_cost],
        ['Tour cost', lhs_sched.tour_cost, rhs_sched.tour_cost],
        ['Total Charge', lhs_sched.total_charge, rhs_sched.total_charge],
        ['Number of departures', lhs_sched.number_of_departures, rhs_sched.number_of_departures],
        ['Length', len(lhs_sched), len(rhs_sched)],
    ]).table)
    return rows

def plot_column_charger_usage(columns: Iterable[Column], charger: Charger, weights: Optional[Dict[Column, float]], hide_unused_periods=True):
    num_cols = sum(1 for _ in columns)
    periods = next(iter(columns)).energy_charged.keys()

    if not weights:
        weights = {col: 1.0 for col in columns}

    rows = [['Period'] + [f'V{col.vehicle}: Column_{col.id}' for col in columns]]
    for p in periods:
        if not hide_unused_periods or any(col.charger_usage[p, charger] for col in columns):
            rows.append([str(p)] + [f'{weights[col] * col.charger_usage[p, charger]:.2f}' for col in columns])

    return AsciiTable(rows).table