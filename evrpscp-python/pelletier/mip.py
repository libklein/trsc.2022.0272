# coding=utf-8
from collections import defaultdict
from typing import List, Tuple, Dict

from docplex.mp.conflict_refiner import ConflictRefiner, ConflictRefinerResult
from docplex.mp.model import Model

from column_generation.output import SolveDetails
from evrpscp import DiscretePeriod
from .models import Charger, Instance, Route
from itertools import product

def _name_x(key):
    p, v, (s, bp) = key
    return f"x_{p.begin},{v},{s.id},{s.segments.index(bp)}"

class PelletierMIP:
    def __init__(self, instance: Instance):
        self._instance = instance

        self._periods: List[DiscretePeriod] = instance.periods
        self._chargers: List[Charger] = instance.chargers
        self._routes: List[Route] = [r for v_routes in instance.routes.values() for r in v_routes]
        self._vehicles = sorted(set(x.vehicle for x in self._routes))
        self._battery = instance.battery

        self._delta = self._periods[0].duration/60 # 30 Minutes
        self._Q = self._battery.battery_capacity_ah # Battery capacity in Ah
        self._E = self._battery.battery_capacity_kwh # Battery capacity in kWh
        self._init_soc = self._battery.initial_soc
        self._min_soc = self._battery.min_soc
        self._max_soc = self._battery.max_soc
        self._C = instance.max_number_of_charges

        # Register some helper sets
        # Make sure that it does not contain the first breakpoint
        self._bps_of_charger = {charger: charger.segments for charger in self._chargers}
        self._charger_bps = [(charger, bp) for charger, bps in self._bps_of_charger.items() for bp in bps]
        # Wdf intervals
        self._wdf = self._battery.wdf_segments

        self._arrival_period_of_predecessor_route = {r: p for k, veh_routes in instance.routes.items() for p, r in zip([self._periods[0]] + list(map(lambda r: r.arrival_period, veh_routes)), veh_routes)}

        self._return_periods = {k: [route.arrival_period for route in instance.routes[k]] for k in self._vehicles}

        # TODO Test
        self._uncovered_periods = {k: sorted(set(self._periods) - set(p for r in self._routes if r.vehicle == k for p in r.covered_periods), key=lambda p: p.begin) for k in self._vehicles}
        # Test
        for r in self._routes:
            for p in r.covered_periods:
                assert p not in self._uncovered_periods[r.vehicle]

        # TODO Test
        self._periods_between_tours: Dict[int, List[List[DiscretePeriod]]] = defaultdict(list)
        for k, uncovered_periods in self._uncovered_periods.items():
            cur_uninterrupted_seq = [uncovered_periods[0]]
            for p in uncovered_periods:
                if cur_uninterrupted_seq[-1].end != p.begin:
                    self._periods_between_tours[k].append(cur_uninterrupted_seq)
                    cur_uninterrupted_seq = []
                cur_uninterrupted_seq.append(p)

            self._periods_between_tours[k].append(cur_uninterrupted_seq)

            # Test
            all_added_periods = {p for periods in self._periods_between_tours[k] for p in periods}
            assert all(p in all_added_periods for p in uncovered_periods)


        self._vehicle_and_periods: List[Tuple[DiscretePeriod, int]] = [(p, k) for k in self._vehicles for p in self._periods]

        self._build_model()
        self._model.parameters.threads = 1

    def solve(self):
        sol = self._model.solve()

        if not sol:
            print("Failed to find a feasible solution!")
            conflict_refiner = ConflictRefiner()
            conflicts: ConflictRefinerResult = conflict_refiner.refine_conflict(self._model, display=True)
            for x in conflicts:
                print(x.name, x.element)
            return None

        return sol

    @property
    def solve_details(self) -> SolveDetails:
        details = self._model.solve_details
        return SolveDetails(Runtime=details.time,ObjVal=self._model.objective_value,ObjBound=details.best_bound,MIPGap=details.gap,IterCount=details.nb_iterations,NodeCount=details.nb_nodes_processed)

    @property
    def active_x(self)-> Dict[Tuple[DiscretePeriod, int], Charger]:
        assert self._model.solve_details
        x = {}
        for (p, k, (s, i)), val in self._x.items():
            if not val.to_bool():
                continue
            assert (p, k) not in x

            x[p, k] = s
        return x

    @property
    def soc_values(self) -> Dict[Tuple[DiscretePeriod, int], float]:
        assert self._model.solve_details
        return {key: val.solution_value for key, val in self._soc.items()}

    def _register_variables(self):
        model = self._model
        # soc_pk, soc of vehicle k at begin of period p
        self._soc = model.continuous_var_dict(self._vehicle_and_periods, lb=0.0, name="soc")
        # i_pk, charging current applied to vehicle k in period p
        self._i = model.continuous_var_dict(self._vehicle_and_periods, lb=0.0, name="i")
        # x_pksi
        self._x = model.binary_var_dict(product(self._periods, self._vehicles, self._charger_bps), name=_name_x)
        # z_pk (Eig. überflüssig)
        self._z = model.binary_var_dict(self._vehicle_and_periods, name="z")
        # deg (== soc^+ in paper). Degradation on segment before route r
        self._deg = model.continuous_var_dict(product(self._wdf, self._routes), lb=0.0, name='deg')
        # u, segment active
        self._u = model.binary_var_dict(product(self._wdf, self._routes), name='u')



    def _build_objective(self):
        energy_cost = sum(p.energyPrice * self._E * (self._delta * self._i[p, k])/self._Q for (p, k) in self._vehicle_and_periods)
        deg_cost = sum(self._deg[d, r] * d.cost_per_soc for d, r in product(self._wdf, self._routes))
        self._model.add_kpi(energy_cost, publish_name='Energy cost')
        self._model.add_kpi(deg_cost, publish_name='Deg cost')
        return energy_cost + deg_cost

    def _add_charge_scheduling_constraints(self):
        model = self._model
        # No charging while en-route
        model.add_constraints((sum(self._x[p, route.vehicle, (s, i)] for p in route.covered_periods for (s, i) in self._charger_bps) == 0 for route in self._routes), names="no_en_route_charging")

        # Arrival SoC
        model.add_constraints((self._soc[r.arrival_period, r.vehicle] == self._soc[r.departure_period, r.vehicle] - r.soc_consumption for r in self._routes), names="arrival_soc")

        # Charger capacity limit
        model.add_constraints((sum(self._x[p, k, (s, i)] for k in self._vehicles for i in self._bps_of_charger[s]) <= s.capacity for (p, s) in product(self._periods, self._chargers)), names="charger_capacity")

        # At most one charge
        model.add_constraints((sum(self._x[p, k, (s, i)] for (s, i) in self._charger_bps) <= 1
                               for (k, p) in product(self._vehicles, self._periods)), names="max_1_charge_per_period")

        # Limit charging current lb
        model.add_constraints((0 <= self._i[p, k] for (k, p) in product(self._vehicles, self._periods)), names="limit_charging_current_lb")

        # Limit charging current ub
        model.add_constraints((self._i[p, k] <= sum(i.current * self._x[p, k, (s, i)] for (s, i) in self._charger_bps) for (k, p) in product(self._vehicles, self._periods)), names="limit_charging_current_ub")

        # Limit to one segment a period
        model.add_constraints((self._soc[p.succ, k] <= i.soc_ub + 1 - self._x[p, k, (s, i)]
                               for (p, k) in product(self._periods, self._vehicles) for (s, i) in self._charger_bps if p.succ is not None),
                              names="max_one_segment_per_period_ub")

        # Limit to one segment a period lb
        model.add_constraints((self._soc[p, k] >= i.soc_lb - 1 + self._x[p, k, (s, i)]
                               for (p, k) in product(self._periods, self._vehicles) for (s, i) in self._charger_bps),
                              names="max_one_segment_per_period_lb")

        # Set SoC
        model.add_constraints((self._soc[p, k] == self._soc[p.pred, k] + self._delta*(self._i[p.pred, k]/self._Q) for k in self._vehicles for p in self._periods if p not in self._return_periods[k] if p.pred is not None), names='set_soc')
        # Set initial soc
        model.add_constraints((self._soc[self._periods[0], k] == self._init_soc for k in self._vehicles), names='set_init_soc')

        # Set SoC Bounds
        model.add_constraints((self._min_soc <= self._soc[p, k] for (k, p) in product(self._vehicles, self._periods)), names="soc_lb")
        model.add_constraints((self._soc[p, k] <= self._max_soc for (k, p) in product(self._vehicles, self._periods)), names="soc_ub")

        # Set z in general
        model.add_constraints((self._z[p, k] >= sum(self._x[p, k, (s, i)] for i in self._bps_of_charger[s]) - sum(self._x[p.pred, k, (s, i)] for i in self._bps_of_charger[s]) for (p, k) in product(self._periods, self._vehicles) for s in self._chargers if p.pred is not None), names="set_z")

        # Set z for first period
        model.add_constraints((self._z[self._periods[0], k] >= sum(self._x[self._periods[0], k, (s, i)] for i in self._bps_of_charger[s]) for k in self._vehicles for s in self._chargers), names="set_z_at_start")

        # Limit number of charges
        model.add_constraints(sum(self._z[p, k] for p in periods) <= self._C for k in self._vehicles for periods in self._periods_between_tours[k])

    def _add_degradation_constraints(self):
        model = self._model
        # Total degradation before route r equals total charge
        model.add_constraints(sum(self._deg[d, r] for d in self._wdf) == self._soc[r.departure_period, r.vehicle] - self._soc[self._arrival_period_of_predecessor_route[r], r.vehicle] for r in self._routes)
        # Segment activiation
        model.add_constraints(0 <= self._deg[d, r] for (d, r) in product(self._wdf, self._routes))
        model.add_constraints(self._deg[d, r] <= d.soc_range * self._u[d, r] for (d, r) in product(self._wdf, self._routes))

        # Make sure that high intervals are filled first
        model.add_constraints(self._deg[d, r] <= d.soc_ub - self._soc[self._arrival_period_of_predecessor_route[r], r.vehicle] + 1 - self._u[d, r] for (d, r) in product(self._wdf, self._routes))

    def _build_model(self):
        self._model = model = Model("Pelletier")
        self._register_variables()
        # Register objective
        model.minimize(self._build_objective())

        self._add_charge_scheduling_constraints()
        self._add_degradation_constraints()