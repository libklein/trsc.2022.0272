# coding=utf-8
from itertools import zip_longest, product
from typing import Dict, Any, Tuple, List
from funcy import pairwise, nth
from parameters import *
from evrpscp import DiscretizedInstance, SchedulingInstance


def check_param_run(param: InstanceParameter, instances: Dict[Any, SchedulingInstance]):
    """
    Checks generated instances for validity.
    """
    instances: List[Tuple[Any, SchedulingInstance]] = sorted(instances.items(), key=lambda y: y[0])
    if param == CHARGER_CAPACITY:
        for base_capacity, instance in instances:
            assert sum(f.capacity for f in instance.chargers) == base_capacity
        for (_, instance), (_, prev_instance) in pairwise(instances):
            assert all(lhs_f.chargingFunction == rhs_f.chargingFunction
                       for lhs_f, rhs_f in zip_longest(instance.chargers, prev_instance.chargers))
    elif param == CHARGER_COMPLEXITY:
        for complexity, instance in instances:
            assert all(complexity == len(f.chargingFunction) for f in instance.chargers)
            assert all(complexity == len(f.inverseChargingFunction) for f in instance.chargers)
        for (_, instance), (_, prev_instance) in pairwise(instances):
            for f, g in zip_longest(instance.chargers, prev_instance.chargers):
                assert f.fullChargeDuration == g.fullChargeDuration
                assert f.capacity == g.capacity
    elif param in (NUM_CHARGER_TYPES_CONSTANT_TOTAL, NUM_CHARGER_TYPES_VARYING_TOTAL):
        for num_types, instance in instances:
            assert len(instance.chargers) == num_types
    elif param == WDF_COMPLEXITY:
        for num_intervals, instance in instances:
            assert len(instance.param.battery.wearCostDensityFunction) == num_intervals
    elif param == TIME_WINDOW_LENGTH:
        for tw_len_periods, instance in instances:
            for tour in (tour for tp in instance.tourPlans for tour in tp):
                assert tour.latest_departure_time - tour.earliest_departure_time == PERIOD_LENGTH_MINUTES * (tw_len_periods + 1)
                assert tour.latest_arrival_time - tour.earliest_arrival_time == PERIOD_LENGTH_MINUTES * (tw_len_periods + 1)
    elif param == FLEET_SIZE:
        # First x tours should be equal
        for (num_veh_lhs, instance_lhs), (num_veh_rhs, instance_rhs) in pairwise(instances):
            assert instance_lhs.tourPlans[:num_veh_lhs] == instance_rhs.tourPlans[:num_veh_lhs]

    if param == NUM_CHARGER_TYPES_VARYING_TOTAL:
        # Total capacity should grow with the number of chargers
        for (num_types_lhs, instance_lhs), (num_types_rhs, instance_rhs) in pairwise(instances):
            assert sum(f.capacity for f in instance_lhs.chargers)/num_types_lhs \
                   == sum(f.capacity for f in instance_rhs.chargers)/num_types_rhs

        for (_, instance), (_, prev_instance) in pairwise(instances):
            # Chargers should equal (at least for [:num_types])
            for f, g in zip(instance.chargers, prev_instance.chargers):
                assert f == g
    elif param == NUM_CHARGER_TYPES_CONSTANT_TOTAL:
        # Total capacity should be equal
        for (_, cur_instance), (_, next_instance) in pairwise(instances):
            assert sum(f.capacity for f in cur_instance.chargers) == sum(f.capacity for f in cur_instance.chargers)

        for (_, instance), (_, prev_instance) in pairwise(instances):
            # Chargers should equal (at least for [:num_types])
            for f, g in zip(instance.chargers, prev_instance.chargers):
                assert f.chargingFunction == g.chargingFunction and f.inverseChargingFunction == g.inverseChargingFunction

    if param not in (NUM_DAYS, TIME_WINDOW_LENGTH):
        # Tours should be equal
        for (_, lhs), (_, rhs) in product(instances, instances):
            if param is not FLEET_SIZE:
                assert lhs.tourPlans == rhs.tourPlans
            else:
                for lhs_tp, rhs_tp in zip(lhs.tourPlans, rhs.tourPlans):
                    assert lhs_tp == rhs_tp
    if param not in (CHARGER_CAPACITY, CHARGER_COMPLEXITY, NUM_CHARGER_TYPES_VARYING_TOTAL, NUM_CHARGER_TYPES_CONSTANT_TOTAL):
        for (_, lhs), (_, rhs) in product(instances, instances):
            if param is not FLEET_SIZE:
                assert lhs.chargers == rhs.chargers
            else:
                for f, g in zip_longest(lhs.chargers, rhs.chargers):
                    assert f.chargingFunction == g.chargingFunction
    if param is not WDF_COMPLEXITY:
        # Battery should equal
        for (_, lhs), (_, rhs) in product(instances, instances):
            assert lhs.param.battery == rhs.param.battery
