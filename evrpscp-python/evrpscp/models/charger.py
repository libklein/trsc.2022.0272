from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement
from dataclasses_json import dataclass_json
from . import PiecewiseLinearFunction, PiecewiseLinearSegment
from typing import Iterator, List


@dataclass_json
@dataclass
class Charger:
    capacity: int
    id: int
    chargingFunction: PiecewiseLinearFunction # Time -> kWh (kwh/time)
    inverseChargingFunction: PiecewiseLinearFunction # kWh -> Time (time/kwh)
    isBaseCharger: bool

    def __post_init__(self):
        for seg, inv_seg in zip(self.chargingFunction, self.inverseChargingFunction):
            assert seg == inv_seg.inverse()
            assert seg.inverse() == inv_seg

    def __hash__(self):
        return self.id

    def equals(self, other: 'Charger'):
        return self.to_dict() == other.to_dict()

    def __iter__(self) -> Iterator[PiecewiseLinearSegment]:
        return iter(self.chargingFunction)

    def __getitem__(self, item):
        return self.chargingFunction[item]

    def __str__(self):
        return f"C{self.id}"

    def __repr__(self):
        return str(self)

    @property
    def breakpoints(self) -> Iterator[float]:
        return zip(self.chargingFunction.breakpoints, self.chargingFunction.image_breakpoints)

    @property
    def max_charging_rate(self) -> float:
        assert self.chargingFunction.is_concave()
        return self.chargingFunction.segments[1].slope

    @property
    def fullChargeDuration(self):
        return self.chargingFunction.segments[-2].upperBound - self.chargingFunction.segments[1].lowerBound

    def duration(self, kwh_begin: float, kwh_end: float):
        return self.inverseChargingFunction(kwh_end) - self.inverseChargingFunction(kwh_begin)

    def charge(self, time_begin: float, time_end: float):
        return self.chargingFunction(time_end) - self.chargingFunction(time_begin)

    def charge_for(self, kwh_begin: float, time: float) -> float:
        return self.chargingFunction(self.inverseChargingFunction(kwh_begin) + time)

    def dump(self) -> str:
        header = [f'Charger {self.id}', 'Dom. Bounds (min)', 'Slope (kWh/min)', 'Img. Bounds (kWh)', 'Inv. slope (min/kWh)']
        return self.chargingFunction.dump(header=header)

    def toXML(self) -> Element:
        xml_charger = Element('Charger')

        xml_capacity = SubElement(xml_charger, 'Capacity')
        xml_capacity.text = f'{self.capacity}'

        xml_id = SubElement(xml_charger, 'ID')
        xml_id.text = f'{self.id}'

        xml_plf = self.chargingFunction.toXML()
        xml_plf.tag = 'ChargingFunction'
        xml_charger.append(xml_plf)

        return xml_charger

