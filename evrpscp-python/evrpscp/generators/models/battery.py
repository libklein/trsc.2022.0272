# coding=utf-8
from typing import Tuple, Union, Callable

import numpy as np

from evrpscp import Battery, PiecewiseLinearFunction, PiecewiseLinearSegment
from evrpscp.generators.models import Parameter, generate_pwl
from random import Random

# ACC(D) = 694D^(-0.795), from
def compute_wdf(num_intervals: int, capacity: float, price: float, acc: Callable[[float], float]=lambda x: 694*x**(-0.795), cycle_includes_discharging: bool = True) -> PiecewiseLinearFunction:
    percent_per_interval = 1.0/num_intervals
    # rhs
    delta_q = capacity/num_intervals
    y = [price/(acc((x+1)*percent_per_interval)*2*delta_q) for x in range(num_intervals)]

    # lhs
    A = np.tri(N=num_intervals)
    # Compute interval_wear_cost. We have interval_wear_cost[0] == W(max), ..., W(0)
    # W(0) is the wear cost of kwh interval [0, percent_per_interval*capacity]
    interval_wear_costs = np.linalg.solve(A, y)
    # Whether wear cost should cover charging and discharging
    if cycle_includes_discharging:
        interval_wear_costs *= 2
    # Construct PWL.
    return PiecewiseLinearFunction.CreatePWLFromSlopeAndUB([((x+1)*percent_per_interval*capacity, slope)
                                                            for slope,x in zip(reversed(interval_wear_costs), range(num_intervals))])

def generate_battery(capacity: Parameter = Parameter(80.0), initial_charge: Parameter = Parameter(0.0),
                     intervals: Parameter = Parameter(2, 4), battery_price_scale: Parameter[float] = Parameter(1.0),
                     generator: Random = Random(), get_unscaled_avg_deg_cost = False) -> Union[Battery, Tuple[Battery, float]]:
    assert initial_charge.max <= capacity.generate(generator)
    num_intervals = intervals.generate(generator)
    bat = Battery(capacity.generate(generator), initialCharge=initial_charge.generate(generator),
                   maximumCharge=capacity.generate(generator), minimumCharge=0.0,
                   wearCostDensityFunction=generate_pwl(ub=capacity.generate(generator), is_image_ub=False, is_concave=False, min_intervals=num_intervals, max_intervals=num_intervals, generator=generator))

    avg_degradation_cost = (bat.wearCostDensityFunction.image_upper_bound-bat.wearCostDensityFunction.image_lower_bound) \
                           / (bat.wearCostDensityFunction.upper_bound - bat.wearCostDensityFunction.lower_bound)

    price_scale_factor = battery_price_scale.generate(rand_gen=generator)
    bat.wearCostDensityFunction = bat.wearCostDensityFunction.scale_slope(price_scale_factor, scale_image=True)
    if get_unscaled_avg_deg_cost:
        return bat, avg_degradation_cost
    else:
        return bat
