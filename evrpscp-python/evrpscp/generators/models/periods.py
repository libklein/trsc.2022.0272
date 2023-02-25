# coding=utf-8
from typing import List, Optional, Callable

from evrpscp import DiscretePeriod
from evrpscp.generators.models import Parameter
from random import Random


def generate_tou_rates(end_of_horizon: float, period_length: float, cost: Parameter = Parameter(0.05, 1.0), generator: Random = Random()):
    # Generate TOU plan and return discrete periods
    period_count = int(end_of_horizon // period_length)
    periods = [DiscretePeriod(begin=int(i * period_length), end=int((i + 1) * period_length),
                              energyPrice=round(cost.generate(generator), 2)) for i in range(period_count)]
    return periods

def _create_periods(periods_in_horizon: int, period_length: float, generate_price: Callable[[], float], round_decimals: Optional[int] = 2, min_price: float = 0.01):
    periods = [
        DiscretePeriod(begin=int(i*period_length), end=int((i+1)*period_length),
                       energyPrice=max(min_price, round(generate_price(), round_decimals) if round_decimals else generate_price()))
        for i in range(periods_in_horizon)
    ]
    for p_prev, p_next in zip(periods, periods[1:]):
        p_prev.succ = p_next
        p_next.pred = p_prev
    return periods

def generate_tou_rates_discrete(periods_in_horizon: int, period_length: float, cost: Parameter, generator: Random = Random()) -> List[DiscretePeriod]:
    return _create_periods(periods_in_horizon=periods_in_horizon, period_length=period_length, generate_price=lambda: cost.generate(generator))

def generate_normally_distributed_tou_rates(periods_in_horizon: int, period_length: float,
                                            mean_price: Optional[float] = None, sigma: Optional[float] = None,
                                            generator: Random = Random(), round_decimals: Optional[int] = 2):
    if mean_price is None:
        raise ValueError("Mean price needs to be specified!")
    elif sigma is None:
        raise ValueError("Variance needs to be specified!")

    # Can be compared safely to periods generated from the same seed
    return _create_periods(periods_in_horizon=periods_in_horizon, period_length=period_length,
                           generate_price=lambda: generator.gauss(mean_price, sigma), min_price=-1000.0,
                           round_decimals=round_decimals)