# coding=utf-8
from typing import List, Tuple, Union

from evrpscp import Charger, DiscreteTour, PiecewiseLinearFunction, Battery
from funcy import butlast, takewhile, ilen, pairwise
import evspnl

from column_generation.subproblem.network import Arc

try:
    from matplotlib import pyplot as plt
    import matplotlib.axes
except:
    pass

def identify_subpaths(labels_and_arcs: List[Tuple[evspnl.Label, Arc]]) -> List[Tuple[List[Tuple[evspnl.Label, Arc]], float]]:
    subpaths = []
    current_path = [labels_and_arcs[0]]
    for (prev_label, _), (label, arc) in pairwise(labels_and_arcs):
        # End the path when the station switches
        if label.tracked_station is not prev_label.tracked_station:
            subpaths.append((current_path, label.minimum_soc))
            current_path = []
        current_path.append((label, arc))
    # Always end with a SoC of 0 at the root!
    assert labels_and_arcs[-1][0].minimum_soc == 0.0
    subpaths.append((current_path, 0.0))
    return subpaths


def propagate_soc_profile(profile: evspnl.PWLFunction, phi: evspnl.ChargingFunction,
                          charge_time: float) -> evspnl.PWLFunction:
    breakpoints_of_new_profile = []
    calculate_exit_soc = lambda entry_soc: phi.getCharge(entry_soc, charge_time)
    add_bp = lambda exit_soc_at_f, exit_soc_at_end: breakpoints_of_new_profile.append(
        evspnl.PWLSegment(exit_soc_at_f, exit_soc_at_end))
    min_exit_soc, max_exit_soc = calculate_exit_soc(profile.image_lower_bound), min(
        calculate_exit_soc(profile.image_upper_bound), phi.maximum_soc)
    # We have (w_j)'(q_f) = (\Phi)'(\Phi^{-1}(w_i(q_f)) + 30) * (\Phi^{-1})'(w_i(q_f)) * (w_i)'(q_f)
    # Each breakpoint of the previous profile is a breakpoint on the new function: (w_i)'(q_f)
    for bp in profile:
        # BP refers to old profile, i.e., bp.domain gives the exit soc at f
        # and bp.image the corresponding exit soc at the previous node (entry soc for us)
        add_bp(bp.domain, calculate_exit_soc(bp.image))
    # Each breakpoint of Phi (or rather, inverse phi) is a bp
    for bp in phi:
        time, soc = bp.domain, bp.image
        # Two breakpoints:
        # 1) (\Phi)'(\Phi^{-1}(w_i(q_f)) + 30) -> When q_f = w_i^{-1}(\Phi(time - 30))
        # 2) (\Phi^{-1})'(w_i(q_f)) -> When q_f = w_i^{-1}(soc)

        # Case 1:
        if time >= charge_time and profile.image_lower_bound <= (
                prev_exit_soc := phi.soc_after(time - charge_time)) <= profile.image_upper_bound:
            if min_exit_soc <= soc <= max_exit_soc:
                q_f = profile.inverse(prev_exit_soc)
                add_bp(q_f, soc)

        # Case 2:
        if profile.image_lower_bound <= soc <= profile.image_upper_bound:
            q_f = profile.inverse(soc)
            if min_exit_soc <= (exit_soc := calculate_exit_soc(soc)) <= max_exit_soc:
                add_bp(q_f, exit_soc)

    breakpoints_of_new_profile.sort(key=lambda x: x.image)
    breakpoints_of_new_profile.sort(key=lambda x: x.domain)

    # Remove any zero slope breakpoints at the end of the profile
    while len(breakpoints_of_new_profile) > 1 and \
            breakpoints_of_new_profile[-1].image == breakpoints_of_new_profile[-2].image:
        breakpoints_of_new_profile.pop()

    # Should be increasing now
    for prev_bp, bp in pairwise(breakpoints_of_new_profile):
        assert prev_bp.domain <= bp.domain and prev_bp.image <= bp.image, \
            f'Error: Charge profile (w) bps are unsorted: {breakpoints_of_new_profile}, !({prev_bp} <= {bp}), prev profile: {profile}'

    assert all(min_exit_soc <= bp.image <= max_exit_soc for bp in breakpoints_of_new_profile)
    return evspnl.construct_from_breakpoints(breakpoints_of_new_profile, True, True)


def calculate_soc_profile(initial_profile: evspnl.PWLFunction,
                          path: List[Union[Tuple[evspnl.ChargingFunction, float], float]]):
    profile = initial_profile
    for next_stop in path:
        if isinstance(next_stop, float):
            profile = evspnl.clip_image(evspnl.shift_pwl_by(profile, 0.0, -next_stop), 0.0,
                                        profile.image_upper_bound - next_stop)
        else:
            phi, charge_time = next_stop
            profile = propagate_soc_profile(profile, phi, charge_time)
    return profile


def simulate_charge_at_tracked_station(station: evspnl.ChargingFunction,
                                       path: List[Union[evspnl.ChargingFunction, float]], target_soc: float,
                                       max_soc: float, initial_soc: float = 0.0, max_charge_time: float = evspnl.PERIOD_LENGTH) -> float:
    delta_tau = 0.0
    simulation_step_size = 0.1
    while delta_tau < max_charge_time + simulation_step_size:
        delta_tau = min(delta_tau, max_charge_time)
        exit_soc = station.getCharge(initial_soc, delta_tau)
        for next_stop in path:
            if isinstance(next_stop, evspnl.ChargingFunction):
                # exit_soc = min(next_stop.getCharge(exit_soc, evspnl.PERIOD_LENGTH), target_soc, max_soc)
                # TODO This is wrong! We may charge more than required! Perhaps use the cost profile?
                exit_soc = min(next_stop.getCharge(exit_soc, max_charge_time), max_soc)
            else:
                exit_soc -= next_stop
            if evspnl.certainly_lt(exit_soc, 0.0):
                break
        if exit_soc >= target_soc:
            if simulation_step_size <= 1e-7:
                return station.getCharge(initial_soc, delta_tau) - initial_soc
            else:
                delta_tau = max(delta_tau - simulation_step_size, 0.0)
                simulation_step_size = simulation_step_size / 10.0
                continue
        delta_tau += simulation_step_size
    raise RuntimeError(f"Could not find delta_tau such that target_soc of {target_soc} can be reached!")


def calculate_charge_at_tracked_station(station: evspnl.ChargingFunction,
                                        path: List[Union[evspnl.ChargingFunction, float]], target_soc: float,
                                        max_charge_time: float, max_soc: float, initial_soc: float = 0.0) -> float:
    # Initially, the mapping of soc_charged_at_initial_profile -> soc_at_end_of_path is the identity function + initial_soc
    max_reachable_soc_at_station = min(station.getCharge(initial_soc, max_charge_time), max_soc)
    initial_profile = evspnl.PWLFunction([evspnl.PWLSegment(initial_soc, initial_soc),
                                          evspnl.PWLSegment(max_reachable_soc_at_station, max_reachable_soc_at_station,
                                                            1.0)])
    # initial_profile = evspnl.clip(evspnl.PWLFunction([x for x in station]), initial_soc, max_reachable_soc_at_station, initial_soc, max_reachable_soc_at_station)
    # Calculate SoC profile
    soc_at_end = calculate_soc_profile(initial_profile=initial_profile,
                                       path=[(stop, max_charge_time)
                                             if isinstance(stop, evspnl.ChargingFunction) else stop
                                             for stop in path])

    if (evspnl.certainly_lt(target_soc, soc_at_end.image_lower_bound) or
            evspnl.certainly_gt(target_soc, soc_at_end.image_upper_bound)):
        raise AssertionError(f"Could not find delta_soc such that target_soc of {target_soc} can be reached!")

    return soc_at_end.inverse(target_soc) - initial_soc


def to_cpp_pwl_function(pwl: PiecewiseLinearFunction, im_ub=None) -> evspnl.PWLFunction:
    breakpoints = [
        evspnl.PWLSegment(seg.upperBound, seg.imageUpperBound, seg.slope) for seg in butlast(pwl.segments)
    ]
    # LB Segment requires special construction
    breakpoints[0] = evspnl.PWLSegment(pwl.segments[0].upperBound, pwl.segments[0].imageUpperBound)
    return evspnl.PWLFunction(breakpoints)


def to_cpp_charger(charger: Charger, soc_ub=None) -> evspnl.Charger:
    phi = evspnl.ChargingFunction(to_cpp_pwl_function(charger.chargingFunction))
    return evspnl.Charger(phi, charger.id)


def to_cpp_tour(tour: DiscreteTour, id=None) -> evspnl.ServiceOperation:
    id = id if id is not None else tour.id
    if id >= evspnl.MAX_TOUR_ID:
        raise RuntimeError(f"Cannot create Tour with tour id >= MAX_TOUR_ID ({id})")
    return evspnl.ServiceOperation(id, tour.consumption,
                                   int(round(tour.earliest_departure_time / evspnl.PERIOD_LENGTH)),
                                   int(round(tour.latest_departure_time / evspnl.PERIOD_LENGTH)), tour.duration)


def to_cpp_wdf(wdf: PiecewiseLinearFunction) -> evspnl.WearCostDensityFunction:
    return evspnl.WearCostDensityFunction(to_cpp_pwl_function(wdf))

def _plot_breakpoints(breakpoints, *args, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([bp.domain for bp in breakpoints],  [bp.image for bp in breakpoints], *args, linestyle='-', marker='o', **kwargs)
    return ax

def plot_pwl(pwl, *args, breakpoints=None, ax: 'matplotlib.axes.Axes'=None, **kwargs):
    if breakpoints is None:
        breakpoints = list(iter(pwl))
    ax = _plot_breakpoints(*args, breakpoints=breakpoints, ax=ax, **kwargs)
    if isinstance(pwl, evspnl.ChargingFunction):
        ax.set_xlabel('Time')
        ax.set_ylabel('SoC')
    elif isinstance(pwl, evspnl.CostProfile):
        ax.set_xlabel('Cost')
        ax.set_ylabel('SoC')
    elif isinstance(pwl, evspnl.WearCostDensityFunction):
        ax.set_xlabel('SoC')
        ax.set_ylabel('Cost')
    return ax
