# coding=utf-8
from decimal import *
from typing import List, Optional
from evrpscp import PiecewiseLinearSegment, PiecewiseLinearFunction, Charger
import json

def create_pwl_function(battery_capacity_ah: Decimal, battery_capacity_kwh: Decimal, breakpoints: List[Decimal]
                   , currents: List[Decimal]) -> PiecewiseLinearFunction:
    """
    battery_capacity_ah: The battery capacity in Ah from breakpoints[0] to breakpoints[-1]
    battery_capacity_kwh: The battery capacity in kWh from breakpoints[0] to breakpoints[-1]
    breakpoints: Breakpoints of the charging function, given in SoC
    currents: Charging current for each segment, given in Amps/h

    Creates PWL of SoC evolution over time, i.e. soc_recharged [kwh] = f(duration [minutes]).
    Slope is hence of unit min/kwh
    """
    assert len(breakpoints) == len(currents) + 1
    assert all(Decimal(0) <= br <= Decimal(1.0) for br in breakpoints)
    # Create segments
    segments = []
    total_time = Decimal(0)  # Minutes
    for amp_current, (soc_lb, soc_ub) in zip(currents, zip(breakpoints, breakpoints[1:])):
        # Recharging the energy on segment (soc_lb, soc_ub) at a rate of amp_current A/h
        duration = (soc_ub - soc_lb) * battery_capacity_ah / amp_current  # hours
        duration *= Decimal(60.0)  # Minutes
        charge_rate = ((soc_ub - soc_lb) * battery_capacity_kwh) / duration
        segments.append(PiecewiseLinearSegment(imageLowerBound=float(soc_lb * battery_capacity_kwh), imageUpperBound=float(soc_ub * battery_capacity_kwh),
                                               lowerBound=float(total_time), upperBound=float(total_time + duration),
                                               slope=float(charge_rate)))
        total_time += duration

    return PiecewiseLinearFunction.CreatePWL(segments)

def create_charger(battery_capacity_ah: Decimal, battery_capacity_kwh: Decimal, breakpoints: List[Decimal]
                   , currents: List[Decimal], capacity: Optional[int] = None, charger_id: int = 0) -> Charger:
    """
    Battery capacity:
    """
    pwl = create_pwl_function(battery_capacity_ah, battery_capacity_kwh, breakpoints, currents)
    return Charger(capacity if capacity is not None else 1000,
                   charger_id, chargingFunction=pwl, inverseChargingFunction=pwl.inverse(),
                   isBaseCharger=capacity is None)

pelletier_data = {
    'Battery': {
        'battery_capacity_ah': Decimal(40.0),
        'battery_capacity_kwh': Decimal(80.0),
        'min_soc': Decimal(0.05),
        'max_soc': Decimal(0.99),
        'initial_soc': Decimal(0.9)
    },
    'SlowCharger': {
        'breakpoints': [Decimal(0.05), Decimal(0.99)],
        'currents': [Decimal(3.5)],
        'charger_id': 0,
        'capacity': None
    },
    'FastCharger': {
        'breakpoints': [Decimal(0.05), Decimal(0.78), Decimal(0.95), Decimal(0.99)],
        'currents': [Decimal(17.5), Decimal(13.6), Decimal(3.2)],
        'charger_id': 1,
        'capacity': 9  # Placeholder
    }
}

if __name__ == '__main__':
    slow_charger = create_charger(**{key: pelletier_data['Battery'][key] for key in ('battery_capacity_ah', 'battery_capacity_kwh')}, **pelletier_data['SlowCharger'])
    fast_charger = create_charger(**{key: pelletier_data['Battery'][key] for key in ('battery_capacity_ah', 'battery_capacity_kwh')}, **pelletier_data['FastCharger'])

    with open('SlowCharger.json', 'w') as slow_charger_file:
        json.dump(slow_charger.to_dict(), slow_charger_file, indent=2)

    with open('FastCharger.json', 'w') as fast_charger_file:
        json.dump(fast_charger.to_dict(), fast_charger_file, indent=2)

