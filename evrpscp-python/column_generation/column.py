# coding=utf-8
from itertools import product
from typing import *
from funcy import first, cached_readonly, iterate, nth

from .discretization import DiscretizedInstance
from .parameters import BIG_M_COEF
from evrpscp import DiscretePeriod, DiscreteTour, Charger, is_close, VehicleChargingSchedule, ChargingOperation\
    , VehicleDeparture
from column_generation.debug_mode import extensive_checks


class Column:
    _next_id = 0

    def __init__(self, energy_charged: Dict[DiscretePeriod, float], degradation_cost: Dict[DiscretePeriod, float],
                 charger_usage: Dict[Tuple[DiscretePeriod, Charger], bool], tour_departures: Dict[DiscreteTour, DiscretePeriod],
                 vehicle: int, objective: float, cost: float = None):
        self.energy_charged: Dict[DiscretePeriod, float] = energy_charged
        self.degradation_cost: Dict[DiscretePeriod, float] = degradation_cost
        self.charger_usage: Dict[Tuple[DiscretePeriod, Charger], bool] = charger_usage
        self.tour_departures: Dict[DiscreteTour, DiscretePeriod] = tour_departures
        self.vehicle: int = vehicle
        self.objective: float = objective
        self.id: int = Column._next_id

        self._cost = cost

        self.chargers: List[Charger] = list(set(f for p, f in charger_usage.keys()))
        self.periods: List[DiscretePeriod] = list(energy_charged.keys())

        Column._next_id += 1
        self.__post_init__()

    @property
    def charging_operations(self):
        return ((p, f) for (p, f), used in self.charger_usage.items() if used)

    @cached_readonly
    def deg_cost(self) -> float:
        return sum(self.degradation_cost.values())

    @cached_readonly
    def energy_cost(self) -> float:
        return sum(p.energyPrice * gamma for p, gamma in self.energy_charged.items())

    @cached_readonly
    def tour_cost(self) -> float:
        return 0

    @property
    def reduced_cost(self) -> float:
        return self.objective

    @cached_readonly
    def cost(self):
        if self._cost is not None:
            return self._cost
        cost = self.deg_cost + self.energy_cost + self.tour_cost
        assert cost >= 0.0
        return cost

    @staticmethod
    def DummyColumn(periods: List[DiscretePeriod], chargers: List[Charger], tours: List[DiscreteTour], vehicle: int):
        last_arrival = periods[0]
        departures: Dict[DiscreteTour, DiscretePeriod] = {}
        for pi in tours:
            departures[pi] = last_arrival
            last_arrival = first(period for period in periods if period > last_arrival)
        return Column(
            energy_charged={p: 0.0 for p in periods},
            degradation_cost={p: 0.0 for p in periods},
            charger_usage={(p, f): False for (p, f) in product(periods, chargers)},
            tour_departures=departures,
            vehicle=vehicle,
            objective=BIG_M_COEF
        )

    def _generate_hash(self) -> int:
        return hash(frozenset(self.charger_usage.values()))

    @property
    def is_dummy(self):
        return self.objective == BIG_M_COEF

    def __hash__(self):
        return self.id

    def __eq__(self, other: 'Column'):
        return (self.energy_charged, self.degradation_cost,
                self.charger_usage, self.tour_departures, self.vehicle) == \
               (other.energy_charged, other.degradation_cost, other.charger_usage,
                other.tour_departures, other.vehicle)


    def __post_init__(self):
        if extensive_checks():
            # Check if column is valid
            chargers = set(f for (p, f) in self.charger_usage.keys())
            for p in self.energy_charged.keys():
                active_charger = first(f for f in chargers if self.charger_usage[p, f])
                if not active_charger:
                    assert is_close(self.energy_charged[p], 0), f'V{self.vehicle}: No charger active in {p} but charges {self.energy_charged[p]}'
                    continue
                assert sum(1 for f in chargers if self.charger_usage[p, f] <= 1)
                # No overlap with tours
                for pi, dep_period in self.tour_departures.items():
                    assert p < dep_period or p > dep_period, f'Vehicle {self.vehicle} serves tour {pi} from ' \
                                                             f'{dep_period.begin} to {dep_period.begin + pi.duration * dep_period.duration} ' \
                                                             f'but charges in {p}!'
            # All tours served
            assert all(self.tour_departures.values())

        self.hash = self._generate_hash()

    #def update_objective(self, coverage_dual: float, capacity_duals: Dict[Tuple[DiscretePeriod, Charger], float]):
    #    self.objective = self.cost - coverage_dual - sum(capacity_dual for (p, f), capacity_dual in capacity_duals.items() if self.charger_usage[p, f])

    def __str__(self):
        return f'Column_{self.id}(V{self.vehicle}, RC: {self.reduced_cost:.2f}, Cost: {self.cost:.2f}, ' \
               f'Energy cost: {self.energy_cost:.2f}, Deg cost: {self.deg_cost:.2f}), Tours: ' \
               f'{[f"{str(pi)}: {departure_period.begin}" for pi, departure_period in self.tour_departures.items()]}'

    def __repr__(self):
        if self.is_dummy:
            return f'Column_{self.id} of V{self.vehicle}: <Dummy>'
        used_chargers = [(p, f) for (p, f), is_used in self.charger_usage.items() if is_used]
        used_chargers.sort(key=lambda x: x[0].begin)
        ops = []

        for p, f in used_chargers:
            last_op = ops[-1] if len(ops) > 0 else None
            if last_op and last_op['end'] == p.begin and last_op['charger'] == f:
                last_op['end'] = p.end
                last_op['energy'] += self.energy_charged[p]
            else:
                ops.append(dict(begin=p.begin, end=p.end, charger=f, energy=self.energy_charged[p]))

        for pi, p in self.tour_departures.items():
            ops.append(dict(begin=p.begin, end=p.begin + pi.duration * p.duration, charger=pi, energy=-pi.consumption))

        ops.sort(key=lambda x: x['begin'])

        return f'Column_{self.id} of V{self.vehicle} (' \
                   f'Reduced cost: {self.reduced_cost:.2f}, Cost: {self.cost:.2f} ' \
                   f'Charge: {sum(self.energy_charged.values()):.2f} [{self.energy_cost:.2f}€], ' \
                   f'Deg-cost: {self.deg_cost:.2f}' \
               f')\n\t' + \
               ', '.join((f'|{x["begin"]}-{x["end"]}: {x["charger"]} / {x["energy"]:.2f}|' for x in ops))

    def __format__(self, format_spec) -> str:
        if format_spec is None or format_spec == '':
            repr = str(self)
        elif format_spec == 't':
            def format_period_activity(p: DiscretePeriod) -> str:
                if (f := first(f for f in self.chargers if self.charger_usage[p, f])) is not None:
                    return str(f.id)
                elif (t := first(tour for tour, departure_period in self.tour_departures.items() if departure_period.begin <= p.begin <= nth(tour.duration, departure_period).begin)):
                    return '#'
                else:
                    return '.'
            repr = ''.join(format_period_activity(p) for p in self.periods)
        else:
            raise NotImplementedError(f'No such format: {format_spec}')
        return repr

    def iter_operations(self, initial_charge: float):
        periods_with_departures = {p: pi for pi, p in self.tour_departures.items()}

        entry_soc = initial_charge
        for p in self.periods:
            if (next_tour := periods_with_departures.get(p)) is not None:
                # We depart in p, hence we arrive in p + dur, because if dur = 0 then we arrive in the same period
                yield VehicleDeparture(begin=p, end=nth(next_tour.duration, iter(p)), entrySoC=entry_soc,
                                                   exitSoC=entry_soc - next_tour.consumption, isFeasible=True, tour=next_tour)
                entry_soc -= next_tour.consumption
                continue

            active_charger: Optional[Charger] = first(f for f in self.chargers if self.charger_usage[p, f])
            if not active_charger:
                continue

            assert entry_soc > -0.01
            entry_soc = max(0.0, entry_soc)
            exit_soc = entry_soc + self.energy_charged[p]
            # Skip if no charge
            if is_close(entry_soc, exit_soc):
                continue

            charge_duration = active_charger.duration(entry_soc, exit_soc)
            assert charge_duration <= p.duration + 0.01, f'Charging from {entry_soc:.2f} to {exit_soc:.2f} ({exit_soc - entry_soc:.2f}) takes {charge_duration:.2f} but period is only {p.duration} min'
            yield ChargingOperation(begin=p, end=p, entrySoC=entry_soc, exitSoC=exit_soc,
                                                chargeDuration=charge_duration, stationID=active_charger.id,
                                                chargeRate=(exit_soc-entry_soc)/charge_duration,
                                                stationNetworkID=active_charger.id, charger=active_charger)

            entry_soc = exit_soc

    def create_vehicle_schedule(self, instance: DiscretizedInstance) -> VehicleChargingSchedule:
        operations = list(self.iter_operations(instance.parameters.battery.initialCharge))
        schedule = VehicleChargingSchedule(cost=0, isFeasible=True, operations=operations, vehicleID=self.vehicle)
        schedule.calculate_cost(instance.parameters.battery)
        return schedule
