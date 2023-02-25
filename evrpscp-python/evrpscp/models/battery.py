from typing import Iterator
from dataclasses import dataclass, field
from xml.etree.ElementTree import Element, SubElement
from dataclasses_json import dataclass_json, config
from . import PiecewiseLinearFunction, PiecewiseLinearSegment

@dataclass_json
@dataclass
class Battery:
    capacity: float
    initialCharge: float = field(metadata=config(field_name='initial'))
    maximumCharge: float = field(metadata=config(field_name='maximum'))
    minimumCharge: float = field(metadata=config(field_name='minimum'))
    wearCostDensityFunction: PiecewiseLinearFunction = field(metadata=config(field_name='wearDensityFunction'))

    def __post_init__(self):
        if not self.minimumCharge <= self.initialCharge <= self.maximumCharge <= self.capacity:
            raise ValueError('Invalid battery values')
        # TODO Check WDF

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Battery: [Min {self.minimumCharge}, Max {self.maximumCharge}, init {self.initialCharge}, capacity {self.capacity}]. WDF: {self.wearCostDensityFunction}'


    def __iter__(self) -> Iterator[PiecewiseLinearSegment]:
        return iter(self.wearCostDensityFunction)

    def wearCost(self, soc_begin: float, soc_end: float) -> float:
        return self.wearCostDensityFunction(soc_end) - self.wearCostDensityFunction(soc_begin)

    def toXML(self) -> Element:
        xml_battery = Element('Battery')

        xml_capacity = SubElement(xml_battery, 'Capacity')
        xml_capacity.text = f'{self.capacity}'

        xml_minimum = SubElement(xml_battery, 'MinimumCharge')
        xml_minimum.text = f'{self.minimumCharge}'

        xml_maximum = SubElement(xml_battery, 'MaximumCharge')
        xml_maximum.text = f'{self.maximumCharge}'

        xml_initialCharge = SubElement(xml_battery, 'InitialCharge')
        xml_initialCharge.text = f'{self.initialCharge}'

        xml_plf = self.wearCostDensityFunction.toXML()
        xml_plf.tag = 'WearCostDensityFunction'
        xml_battery.append(xml_plf)

        return xml_battery
