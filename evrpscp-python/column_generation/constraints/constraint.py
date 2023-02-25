# coding=utf-8
from evrpscp import Charger, DiscretePeriod
from column_generation import Column


class Constraint:

    def is_violated(self, column: Column) -> bool:
        raise NotImplementedError

    def applies_to_vehicle(self, vehicle: int) -> bool:
        raise NotImplementedError

    def conflicts(self, other: 'Constraint') -> bool:
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class BlockChargerConstraint(Constraint):
    def __init__(self, vehicle: int, charger: Charger, period: DiscretePeriod, allocate: bool):
        self._vehicle = vehicle
        self._charger = charger
        self._period = period
        self._allocate = allocate
        # It does not make sense to forbid a base charger from being used
        assert not self._charger.isBaseCharger

    def __eq__(self, other: 'BlockChargerConstraint'):
        return (self._vehicle, self._charger, self._period, self._allocate) == (other._vehicle, other._charger, other._period, other._allocate)

    @property
    def vehicle(self) -> int:
        return self._vehicle

    @property
    def charger(self) -> Charger:
        return self._charger

    @property
    def period(self) -> DiscretePeriod:
        return self._period

    @property
    def force_usage(self) -> bool:
        return self._allocate

    def conflicts(self, other):
        if isinstance(other, BlockChargerConstraint):
            return self.vehicle == other.vehicle and \
                   self.period == other.period and \
                   self.charger == other.charger and \
                   self.force_usage != other.force_usage
        else:
            raise NotImplementedError

    def is_violated(self, column: Column) -> bool:
        return self.applies_to_vehicle(column.vehicle) and column.charger_usage[
            self._period, self._charger] != self._allocate

    def applies_to_vehicle(self, vehicle: int) -> bool:
        return self._vehicle == vehicle

    def __hash__(self):
        return hash((self._vehicle, self._charger, self._period, self._allocate))

    def __str__(self):
        if self._allocate:
            f'{self._vehicle} uses {self._charger} in {self._period}'
        else:
            f'{self._vehicle} blocked {self._charger} in {self._period}'

        return f'v[{self._vehicle}, {self._period}, {self._charger}] == {int(self._allocate)}'

    def __repr__(self):
        return f'v[{self._vehicle}, {self._period}, {self._charger}] == {int(self._allocate)}'
