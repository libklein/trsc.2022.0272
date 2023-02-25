from dataclasses import dataclass, field, InitVar
from dataclasses_json import dataclass_json, config, DataClassJsonMixin
from xml.etree.ElementTree import Element, SubElement
from typing import List, Union, Optional
from marshmallow import fields
import json

@dataclass_json
@dataclass(eq=False)
class Period:
    begin: float = field(metadata=config(mm_field=fields.Float(), encoder=lambda x: round(float(x), 2)))
    end: float = field(metadata=config(mm_field=fields.Float(), encoder=lambda x: round(float(x), 2)))
    energyPrice: float = field(metadata=config(mm_field=fields.Float(), encoder=lambda x: round(float(x), 2)))
    succ: Optional['DiscretePeriod'] = field(default=None, metadata=config(exclude=lambda _: True))
    pred: Optional['DiscretePeriod'] = field(default=None, metadata=config(exclude=lambda _: True))

    def __post_init__(self):
        if self.energyPrice < 0:
            raise ValueError("Negative energy price!")

    def __iter__(self):
        return PeriodIterator(self)

    @property
    def duration(self):
        return self.end - self.begin

    def offset(self, offset: float) -> 'Period':
        self.begin += offset
        self.end += offset
        return self

    def __str__(self):
        return f"P[{self.begin}-{self.end}]"

    def __repr__(self):
        return str(self)

    def __gt__(self, other: 'Period') -> bool:
        return self.begin >= other.end

    def toXML(self) -> Element:
        xml_period = Element('Period')

        xml_begin = SubElement(xml_period, 'Begin')
        xml_begin.text = f'{self.begin:.2f}'

        xml_end = SubElement(xml_period, 'End')
        xml_end.text = f'{self.end:.2f}'

        xml_energy = SubElement(xml_period, 'EnergyPrice')
        xml_energy.text = f'{self.energyPrice:.2f}'

        return xml_period


@dataclass(eq=False)
class DiscretePeriod:
    # Time at which this period begins (i.e. minutes)
    begin: int
    # Time at which this period ends (i.e. minutes)
    end: int
    energyPrice: float
    succ: InitVar[Optional['DiscretePeriod']] = None
    pred: InitVar[Optional['DiscretePeriod']] = None

    def to_dict(self, *args, **kwargs):
        return dict(begin=self.begin, end=self.end, energyPrice=self.energyPrice)

    def to_json(self, *args, **kwargs):
        return json.dumps(self.to_dict())

    def __post_init__(self, succ: Optional['DiscretePeriod'] = None, pred: Optional['DiscretePeriod'] = None):
        self.succ = succ
        self.pred = pred
        assert self.end > self.begin and (self.end - self.begin) >= 1
        if self.energyPrice < 0:
            raise ValueError("Negative energy price!")

    def __iter__(self):
        return PeriodIterator(self)

    @property
    def duration(self):
        return self.end - self.begin

    def offset(self, offset: int) -> 'DiscretePeriod':
        self.begin += offset
        self.end += offset
        return self

    def __str__(self):
        return f"P[{self.begin}-{self.end} | {self.energyPrice}€]"

    def __repr__(self):
        return str(self)

    def __contains__(self, item) -> bool:
        return self.begin < item <= self.end

    def __gt__(self, other: 'DiscretePeriod') -> bool:
        if isinstance(other, DiscretePeriod):
            return self.begin > other.begin
        elif isinstance(other, float) or isinstance(other, int):
            return self.begin > other
        else:
            raise NotImplementedError()

    def __ge__(self, other: Union['DiscretePeriod', float, int]) -> bool:
        if isinstance(other, DiscretePeriod):
            return self.begin >= other.begin
        elif isinstance(other, float) or isinstance(other, int):
            return self.begin >= other
        else:
            raise NotImplementedError()


class PeriodVertex:
    def __init__(self, strID, x, y, demand, service_time, customer = True, station = False, depot = False, period=0):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json):
        return cls('_'.join(json['strID'].split("_")[:-1]), None, None, float(json['demand']), float(json['serviceTime']),
                   customer=json['type'] == 'customer', station=json['type'] == 'station', depot=json['type'] == 'depot')


    @property
    def strID(self):
        if self.customer:
            return f"{self._strID}_{self.period}"
        else:
            return self._strID

    def __repr__(self):
        return f"{self._strID}:{self.period}"

    def __hash__(self):
        return hash(self.strID)

    def toXML(self) -> Element:
        xml_rep = super().getXMLRepr()
        xml_period = SubElement(xml_rep, 'Period')
        xml_period.text = f'{self.period}'

        return xml_rep

class PeriodIterator:
    def __init__(self, p):
        self.next = p

    def __iter__(self):
        return self

    def __next__(self):
        if self.next is None:
            raise StopIteration
        _next = self.next
        self.next = _next.succ
        return _next

def find_intersecting_period(t: float, periods: List[Union[Period, DiscretePeriod]]) -> int:
    """
    Find the index of the period which contains time point t or -1 if none contains t
    :param t: A point in time
    :param periods: List of periods
    :return: The index of the first period p with p.begin <= t < t or -1 if no such p exists
    """
    for i, p in enumerate(periods):
        if p.begin <= t < p.end:
            return i
    return -1

def is_within_period(timepoint: float, p: Union[Period, DiscretePeriod]):
    return p.begin <= timepoint < p.end