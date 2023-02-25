# coding=utf-8
import math
import sys
from typing import Callable

import pelletier.models as pelletier_models
import evrpscp.models as evrp_models


def _convert_soc(watts: float, Q: float) -> float:
    assert 0 <= watts <= Q
    return watts / Q


def _convert_min_to_h(minutes: float) -> float:
    return minutes / 60.


def isfinite(val: float):
    return math.isfinite(val) and abs(val) != sys.float_info.max


def _normalize_seg(seg: evrp_models.PiecewiseLinearSegment, x_converter: Callable[[float], float],
                   y_converter: Callable[[float], float]) -> evrp_models.PiecewiseLinearSegment:
    lower_bound = x_converter(seg.lowerBound)
    upper_bound = x_converter(seg.upperBound)
    image_lower_bound = y_converter(seg.imageLowerBound)
    image_upper_bound = y_converter(seg.imageUpperBound)
    if all(map(isfinite, (lower_bound, upper_bound, image_upper_bound, image_lower_bound))):
        slope = (image_upper_bound - image_lower_bound) / (upper_bound - lower_bound)
    else:
        slope = seg.slope

    return evrp_models.PiecewiseLinearSegment(lowerBound=lower_bound, upperBound=upper_bound,
                                              imageLowerBound=image_lower_bound, imageUpperBound=image_upper_bound,
                                              slope=slope)


def _normalize_soc(pwl: evrp_models.PiecewiseLinearFunction, convert_x: Callable[[float], float] = None,
                   convert_y: Callable[[float], float] = None) -> evrp_models.PiecewiseLinearFunction:
    x_converter = lambda val: convert_x(val) if (convert_x and isfinite(val)) else val
    y_converter = lambda val: convert_y(val) if (convert_y and isfinite(val)) else val
    return evrp_models.PiecewiseLinearFunction(
        [_normalize_seg(seg=seg, x_converter=x_converter, y_converter=y_converter) for seg in pwl.segments])


def convert_instance(instance: evrp_models.DiscretizedInstance,
                     max_number_of_charges: int) -> pelletier_models.Instance:
    Q = instance.parameters.battery.capacity
    battery = convert_battery(instance.parameters.battery)
    return pelletier_models.Instance(periods=instance.periods,
                                     routes={k: [convert_tour(t, Q, k) for t in tours]
                                             for k, tours in enumerate(instance.tours)},
                                     chargers=[convert_charger(f, Q, battery.battery_capacity_ah) for f in
                                               instance.chargers],
                                     battery=battery,
                                     max_number_of_charges=max_number_of_charges)


def convert_tour(tour: evrp_models.DiscreteTour, battery_capacity_kwh: float, vehicle: int) -> pelletier_models.Route:
    assert tour.earliest_departure == tour.latest_departure
    return pelletier_models.Route(id=tour.id, arrival_period=tour.earliest_arrival,
                                  departure_period=tour.earliest_departure,
                                  soc_consumption=_convert_soc(tour.consumption, battery_capacity_kwh), vehicle=vehicle)


def convert_charger(charger: evrp_models.Charger, battery_capacity_kwh: float,
                    battery_capacity_amp: float) -> pelletier_models.Charger:
    return pelletier_models.Charger(id=charger.id,
                                    phi=_normalize_soc(charger.chargingFunction, convert_x=_convert_min_to_h,
                                                       convert_y=lambda watts: _convert_soc(watts, battery_capacity_kwh)),
                                    capacity=charger.capacity,
                                    battery_capacity_amps=battery_capacity_amp)


def convert_battery(battery: evrp_models.Battery) -> pelletier_models.Battery:
    assert battery.capacity == 80.0, "Check for new batteries"
    E = battery.capacity
    return pelletier_models.Battery(
        _normalize_soc(battery.wearCostDensityFunction, convert_x=lambda watts: _convert_soc(watts, E), convert_y=None),
        battery.capacity / 2.0, E,
        min_soc=_convert_soc(battery.minimumCharge, E),
        max_soc=_convert_soc(battery.maximumCharge, E),
        initial_soc=_convert_soc(battery.initialCharge, E))
