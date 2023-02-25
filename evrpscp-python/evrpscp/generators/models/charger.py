# coding=utf-8
from random import Random

from evrpscp import Charger
from evrpscp.generators.models.pwl import generate_pwl


def generate_charger(battery_capacity: float, charger_capacity: int, duration: float = 660, intervals=3,
                     generator: Random = Random(), *args, **kwargs):
    phi = generate_pwl(ub=battery_capacity, generator=generator, min_intervals=intervals, max_intervals=intervals)
    # Scale phi to fit duration
    scaling_factor = phi.upper_bound / duration
    phi = phi.scale_slope(scaling_factor, scale_image=False)
    phi.upper_bound, phi.image_upper_bound = duration, battery_capacity
    assert abs(phi.upper_bound - duration) < 0.01 and abs(phi.image_upper_bound - battery_capacity) < 0.01
    kwargs.setdefault('id', 0)
    kwargs.setdefault('isBaseCharger', charger_capacity <= 0)
    return Charger(chargingFunction=phi, inverseChargingFunction=phi.inverse(), capacity=charger_capacity, *args, **kwargs)

def simulate_charger():
    import pybamm
    import matplotlib.pyplot as plt

    pybamm.set_logging_level("INFO")
    experiment = pybamm.Experiment(
        [
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 W until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour"
        ]
    )
    model = pybamm.lithium_ion.DFN()
    sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
    sim.solve()

    # Plot voltages from the discharge segments only
    fig, axes = plt.subplots(3)
    var_names = (('Terminal voltage [V]', 'Measured battery open circuit voltage [V]'), 'Current [A]', 'Discharge capacity [A.h]')
    # Extract sub solutions
    sol = sim.solution
    for var_name_arr, ax in zip(var_names, axes):
        if not isinstance(var_name_arr, tuple):
            var_name_arr = (var_name_arr,)
        # Extract variables
        for var_name in var_name_arr:
            t = sol["Time [h]"].entries
            var = sol[var_name].entries
            # Plot
            ax.plot(t - t[0], var, label=var_name)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel('['+var_name_arr[0].split('[')[-1])
        ax.set_xlim([11, 13])
        ax.legend()
    plt.show()

    # Show all plots
    #sim.plot()

if __name__ == '__main__':
    simulate_charger()