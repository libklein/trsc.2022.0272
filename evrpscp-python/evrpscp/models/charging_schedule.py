import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Union, Iterable
from dataclasses_json import dataclass_json, config
from . import FleetTourPlan, Tour, Charger, Period, PiecewiseLinearSegment, DiscreteTour, DiscretePeriod, Battery
from evrpscp import is_close
from funcy import cached_readonly, pairwise


@dataclass_json
@dataclass
class Operation:
    begin: DiscretePeriod
    end: DiscretePeriod
    entrySoC: float
    exitSoC: float

    def __post_init__(self):
        assert isinstance(self.begin, DiscretePeriod)
        assert isinstance(self.end, DiscretePeriod)

    @property
    def soc_delta(self):
        return self.exitSoC - self.entrySoC

    @cached_readonly
    def periods(self) -> List[Period]:
        covers = [self.begin]
        while covers[-1] is not self.end:
            covers.append(covers[-1].succ)
        return covers

    @property
    def duration(self) -> int:
        return len(self.periods)

    @property
    def duration_time(self) -> float:
        return self.duration * self.begin.duration

    @staticmethod
    def FromDict(dict) -> Union['VehicleDeparture', 'ChargingOperation']:
        if dict['type'] == 'tour':
            return VehicleDeparture.FromDict(dict)
        elif dict['type'] == 'rechargingOperation':
            return ChargingOperation.FromDict(dict)
        raise NotImplementedError(f'No data class for label type {dict["type"]} implemented!')

    def __post_init__(self):
        assert self.begin <= self.end

@dataclass_json
@dataclass
class VehicleDeparture(Operation):
    isFeasible: bool = field(metadata=config(field_name='feasible'))
    tour: Optional[Union[Tour, DiscreteTour]] = None

    @property
    def cost(self):
        assert self.tour is not None
        return self.tour.cost

    def __post_init__(self):
        if self.tour is not None:
            assert is_close(self.soc_delta, -1 * self.tour.consumption)
            assert self.tour.earliest_departure <= self.begin <= self.tour.latest_departure
            assert self.tour.earliest_arrival <= self.end <= self.tour.latest_arrival

    @property
    def duration(self) -> int:
        return self.tour.duration

    def __repr__(self):
        return f'[{self.begin}-{self.end} ({self.duration} min)/{self.entrySoC:.2f}-{self.exitSoC:.2f} ({self.soc_delta:.2f} kWh), {self.tour}]'

@dataclass_json
@dataclass
class ChargingOperation(Operation):
    chargeDuration: Optional[float] = None
    chargeRate: Optional[float] = None # in kWh per min
    stationID: Optional[int] = None
    stationNetworkID: Optional[int] = None
    charger: Optional[Charger] = None

    def __post_init__(self):
        if self.charger:
            if self.chargeDuration is None:
                self.chargeDuration = self.charger.duration(self.entrySoC, self.exitSoC)
            if self.stationID is None:
                self.stationID = self.charger.id
            if self.stationNetworkID is None:
                self.stationNetworkID = self.charger.id

        try:
            assert is_close(self.chargeDuration, self.charger.duration(self.entrySoC, self.exitSoC)), \
                f'Error: Charge duration of operation \n\t{self}\nis {round(self.chargeDuration, 2)} but charging from {round(self.entrySoC, 2)} ' \
                f'to {round(self.exitSoC, 2)} takes {round(self.charger.duration(self.entrySoC, self.exitSoC), 2)}.'
            assert self.chargeDuration < self.duration_time or is_close(self.chargeDuration, self.duration_time)
            if self.soc_delta > 0:
                calculated_charge_rate = self.soc_delta / self.chargeDuration
                if not is_close(calculated_charge_rate, self.chargeRate):
                    raise AssertionError(f'Calculated charge rate ({calculated_charge_rate:.2f}) does not match '
                                         f'charge rate ({self.chargeRate:.2f})')
        except AssertionError as e:
            print(f'Invalid Charging Operation! Span: {self.begin}-{self.end}, SoC: {self.entrySoC}-{self.exitSoC}. '
                  f'Charger: {self.charger}')
            raise e

    def __repr__(self):
        return f'[{self.begin.begin:.2f}-{self.end.end:.2f} ({self.duration_time:.2f} min)/{self.entrySoC:.2f}-{self.exitSoC:.2f} ({self.soc_delta:.2f} kWh), {self.charger} {self.chargeDuration:.2f}min @ {self.begin.energyPrice:.2f}€ ({self.chargeRate:.2f} kWh/t)]'

    @property
    def degradation_cost(self):
        return self._degradation_cost

    @property
    def energy_cost(self):
        return self._energy_cost

    def calculate_cost(self, battery: Battery) -> float:
        assert (self.duration == 1) # Only implemented for single-period charging (otherwise period would be a list)
        self._degradation_cost = battery.wearCost(self.entrySoC, self.exitSoC)
        self._energy_cost = self.soc_delta * self.begin.energyPrice
        return self._degradation_cost + self._energy_cost

    @staticmethod
    def FromChargerPeriod(charger: Charger, begin: DiscretePeriod, entry_soc: float, soc_delta: float):
        assert soc_delta >= 0
        charge_duration = round(charger.duration(entry_soc, entry_soc + soc_delta), 2)
        charging_rate = soc_delta / charge_duration
        end = begin
        while end.end < begin.begin + charge_duration:
            end = end.succ
        assert begin == end, f'Charging for {charge_duration} is not supported!'
        return ChargingOperation(begin=begin, end=end,
                                 entrySoC=entry_soc, exitSoC=entry_soc+soc_delta,
                                 chargeDuration=charge_duration, chargeRate=charging_rate,
                                 stationID=charger.id, stationNetworkID=charger.id, charger=charger)

@dataclass_json
@dataclass
class VehicleChargingSchedule:
    cost: float
    isFeasible: bool = field(metadata=config(field_name='feasible'))
    operations: List[Operation] = field(metadata=config(field_name='schedule'))
    vehicleID: int

    def __post_init__(self):
        self.operations.sort(key=lambda op: op.begin)

        # Operations must not overlap!
        for i, j in pairwise(self.operations):
            if isinstance(i, VehicleDeparture):
                if i.end.begin == j.begin.begin:
                    print(f'[Error][Overlap]: Operations {i} and {j} overlap.', file=sys.stderr)
                    break


    def __iter__(self) -> Iterable[Operation]:
        return iter(self.operations)

    def __repr__(self):
        output = f'Charging schedule of vehicle {self.vehicleID} (cost: {self.cost}):\n'
        for op in self.operations:
            output += '\t'+repr(op)+'\n'
        return output[:-1]  # Discard last newline

    def __getitem__(self, item) -> Operation:
        return self.operations[item]

    def __len__(self) -> int:
        return len(self.operations)

    @property
    def is_feasible(self) -> bool:
        return all(op.exitSoC >= -0.01 for op in self.operations)

    @property
    def number_of_departures(self) -> int:
        return sum(1 if isinstance(op, VehicleDeparture) else 0 for op in self.operations)

    @property
    def degradation_cost(self) -> float:
        return sum(op.degradation_cost if not isinstance(op, VehicleDeparture) else 0.0 for op in self.operations)

    @property
    def tour_cost(self) -> float:
        return sum(op.cost if isinstance(op, VehicleDeparture) else 0.0 for op in self.operations)

    @property
    def energy_cost(self) -> float:
        return sum(op.energy_cost if not isinstance(op, VehicleDeparture) else 0.0 for op in self.operations)

    @property
    def total_charge(self) -> float:
        return sum(max(0.0, x.exitSoC - x.entrySoC) for x in self.operations)

    def calculate_cost(self, battery: Battery) -> float:
        self.cost = 0.0
        if len(self.operations) == 0:
            return self.cost
        next_period_id = 0
        for op in self.operations:
            if isinstance(op, VehicleDeparture):
                if op.tour:
                    self.cost += op.tour.cost
            else:
                self.cost += op.calculate_cost(battery)
        return self.cost

@dataclass_json
@dataclass
class FleetChargingSchedule:
    vehicleSchedules: List[VehicleChargingSchedule] = field(metadata=config(field_name='schedules'))
    isFeasible: Optional[bool] = field(metadata=config(field_name='feasible'), default=None)
    cost: Optional[float] = None

    def __str__(self):
        output = f'Charging schedules for {len(self.vehicleSchedules)} vehicles (total cost: {self.cost}):\n'
        for schedule in self.vehicleSchedules:
            output += str(schedule)
            output += '\n'
        return output[:-1]  # Discard last newline

    def __repr__(self):
        return str(self)

    @property
    def is_feasible(self) -> bool:
        period_charging_ops = defaultdict(list)
        for schedule in self.vehicleSchedules:
            for op in schedule:
                if isinstance(op, ChargingOperation):
                    period_charging_ops[op.begin, op.charger].append(op)
        for (p, f), operations in period_charging_ops.items():
            if len(operations) > f.capacity:
                return False
        return all(schedule.is_feasible for schedule in self.vehicleSchedules)

    @property
    def number_of_departures(self) -> int:
        return sum(x.number_of_departures for x in self.vehicleSchedules)

    @property
    def tour_cost(self) -> float:
        return 0
        return sum(x.tour_cost for x in self.vehicleSchedules)

    @property
    def degradation_cost(self) -> float:
        return sum(x.degradation_cost for x in self.vehicleSchedules)

    @property
    def energy_cost(self) -> float:
        return sum(x.energy_cost for x in self.vehicleSchedules)

    @property
    def total_charge(self) -> float:
        return sum(x.total_charge for x in self.vehicleSchedules)

    def __post_init__(self):
        self.vehicleSchedules.sort(key=lambda x: x.vehicleID)
        self.cost = sum((x.cost for x in self.vehicleSchedules)) \
            if self.cost is None else self.cost
        self.isFeasible = all((x.isFeasible for x in self.vehicleSchedules)) \
            if self.isFeasible is None else self.isFeasible

    def __iter__(self) -> Iterable[VehicleChargingSchedule]:
        return iter(self.vehicleSchedules)

    def calculate_cost(self, battery: Battery) -> float:
        self.cost = sum(x.calculate_cost(battery) for x in self.vehicleSchedules)
        return self.cost

@dataclass_json
@dataclass
class FleetSchedule:
    tourPlan: FleetTourPlan = field(metadata=config(field_name='tours'))
    chargingSchedule: FleetChargingSchedule = field(metadata=config(field_name='schedules'))

    def __iter__(self):
        return zip(self.chargingSchedule, self.tourPlan)
