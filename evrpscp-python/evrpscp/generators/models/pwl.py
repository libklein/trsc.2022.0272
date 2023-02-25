# coding=utf-8
from random import Random

from evrpscp import PiecewiseLinearSegment, PiecewiseLinearFunction


def generate_pwl(ub: float, is_image_ub=True, is_concave=True, min_intervals=1, max_intervals=5, min_slope=0.1,
                 max_slope=0.8, generator: Random = Random()) -> PiecewiseLinearFunction:
    pwl_generator = Random(generator.random())
    # Choose number of intervals
    number_of_intervals = pwl_generator.randint(min_intervals, max_intervals)
    # Choose how much each interval should cover
    weights = [pwl_generator.uniform(0, 1) for _ in range(number_of_intervals)]
    percentage_covered = [interval_weight / sum(weights) for interval_weight in weights]
    percentage_covered[-1] = 1.0 - sum(percentage_covered[:-1])
    assert sum(percentage_covered) == 1
    slopes = sorted((pwl_generator.uniform(min_slope, max_slope) for _ in range(number_of_intervals)),
                    reverse=is_concave)

    prev_ub, prev_img_ub = 0.0, 0.0
    segments = []
    for covered_percentage, slope in zip(percentage_covered, slopes):
        covered_abs = ub * covered_percentage
        if is_image_ub:
            segments.append(PiecewiseLinearSegment(
                lowerBound=prev_ub, imageLowerBound=prev_img_ub,
                upperBound=prev_ub + (covered_abs / slope),
                imageUpperBound=prev_img_ub + covered_abs,
                slope=slope
            ))
        else:
            segments.append(PiecewiseLinearSegment(
                lowerBound=prev_ub, imageLowerBound=prev_img_ub,
                upperBound=prev_ub + covered_abs,
                imageUpperBound=prev_img_ub + (covered_abs * slope),
                slope=slope
            ))
        prev_ub, prev_img_ub = segments[-1].upperBound, segments[-1].imageUpperBound

    if is_image_ub:
        segments[-1].imageUpperBound = ub
    else:
        segments[-1].upperBound = ub

    return PiecewiseLinearFunction.CreatePWL(segments)