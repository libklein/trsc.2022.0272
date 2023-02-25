from dataclasses import dataclass, field
from xml.etree.ElementTree import Element, SubElement
from typing import List, Optional, Union, Tuple, Iterable, Iterator, Dict
from dataclasses_json import dataclass_json
from math import ceil
from . import Period, DiscretePeriod, Route, is_within_period
from itertools import cycle
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
from funcy import cached_readonly, some, nth, dropwhile, first


@dataclass_json
@dataclass
class Tour:
    id: int
    duration_time: float
    earliest_departure_time: float
    latest_departure_time: float
    consumption: float
    cost: float

    def __str__(self):
        return f'T{self.id}'

    def __post_init__(self):
        self.cost = float(self.cost)
        self.consumption = float(self.consumption)
        self.earliest_departure_time = float(self.earliest_departure_time)
        self.latest_departure_time = float(self.latest_departure_time)

    @property
    def earliest_arrival_time(self) -> float:
        return self.earliest_departure_time + self.duration_time

    @property
    def latest_arrival_time(self) -> float:
        return self.latest_departure_time + self.duration_time

    @property
    def departure_time_window_length(self) -> float:
        return self.latest_departure_time - self.earliest_departure_time

    def __hash__(self):
        return id(self)

    def equals(self, other: 'Tour') -> bool:
        return self.to_dict() == other.to_dict()


@dataclass
class DiscreteTour:
    id: int
    duration: int
    """
    First period in which the vehicle can leave the depot (i.e. first time at which the vehicle becomes unavailable)
    """
    earliest_departure: DiscretePeriod
    """
    Last period in which the vehicle can leave the depot (i.e. last time at which the vehicle becomes unavailable)
    """
    latest_departure: DiscretePeriod
    consumption: float
    cost: float

    def __post_init__(self):
        assert isinstance(self.earliest_departure, DiscretePeriod), f"DiscreteTour earliest departure is not a DiscretePeriod: {self.earliest_departure}"
        assert isinstance(self.latest_departure, DiscretePeriod), f"DiscreteTour latest departure is not a DiscretePeriod: {self.latest_departure}"
        assert isinstance(self.duration, int), f"Duration is not integer: {self.duration}"
        assert self.earliest_arrival is not None
        assert self.latest_arrival is not None
        assert self.earliest_arrival_time - self.earliest_departure_time == self.duration_time
        assert self.latest_arrival_time - self.latest_departure_time == self.duration_time
        assert self.earliest_departure <= self.latest_departure

    def __hash__(self):
        return id(self)

    @staticmethod
    def from_dict(dict_repr: Dict, periods: List[DiscretePeriod]):
        earliest_departure = dict_repr.pop('earliest_departure')
        latest_departure = dict_repr.pop('latest_departure')
        earliest_departure = first(p for p in periods if p.begin == earliest_departure['begin'])
        latest_departure = first(p for p in periods if p.end == latest_departure['end'])
        return DiscreteTour(earliest_departure=earliest_departure, latest_departure=latest_departure, **dict_repr)

    # Time windows - dynamic tours
    @property
    def earliest_departure_time(self) -> float:
        """
        Earliest point in time at which we could possibly depart
        i.e. begin of earliest departure period
        """
        return self.earliest_departure.begin

    @property
    def latest_departure_time(self) -> float:
        """
        Latest point in time at which we could possibly depart
        i.e. end of latest departure period
        """
        return self.latest_departure.end

    @cached_readonly
    def earliest_arrival(self) -> DiscretePeriod:
        """
        Earliest period during which the vehicle may arrive at the depot. It is still blocked during this period
        """
        earliest_arrival = nth(self.duration, self.earliest_departure)
        return earliest_arrival

    @property
    def earliest_arrival_time(self) -> float:
        return self.earliest_arrival.begin

    @cached_readonly
    def latest_arrival(self) -> DiscretePeriod:
        """
        Latest period during which the vehicle may arrive at the depot. It is still blocked during this period
        """
        latest_arrival = nth(self.duration, self.latest_departure)
        return latest_arrival

    @property
    def latest_arrival_time(self) -> float:
        return self.latest_arrival.end

    @cached_readonly
    def departure_time_window_length(self) -> int:
        length = next(dropwhile(lambda p: p[1] is not self.latest_departure, enumerate(self.earliest_departure)))[0]
        assert nth(length, self.earliest_departure) is self.latest_departure
        return length

    @property
    def duration_time(self) -> float:
        # Number of periods * period length
        return self.duration * self.earliest_departure.duration

    def __str__(self):
        return f'T{self.id}'

    def __repr__(self) -> str:
        return f'T{self.id}[{self.earliest_departure_time}-{self.latest_departure_time}, {self.earliest_arrival_time}-{self.latest_arrival_time}, SoC: {self.consumption}, Cost: {self.cost}]'

    def equals(self, other: 'DiscreteTour') -> bool:
        return (self.cost, self.consumption, self.duration, self.earliest_departure, self.latest_departure) \
               == (other.cost, other.consumption, other.duration, other.earliest_departure, other.latest_departure)

    @staticmethod
    def FromTour(tour: Tour, periods: List[DiscretePeriod]) -> 'DiscreteTour':
        if tour.earliest_departure_time == 0:
            earliest_departure_period = periods[0]
            assert periods[0].begin == tour.earliest_departure_time
        else:
            earliest_departure_period = first(p for p in periods if (p.begin >= tour.earliest_departure_time))
        if tour.latest_departure_time == 0:
            latest_departure_period = periods[0]
            assert periods[0].begin == tour.latest_departure_time
        else:
            latest_departure_period = first(p for p in periods if tour.latest_departure_time <= p.end)

        if not earliest_departure_period or not latest_departure_period:
            print(f"Could not discretize tour {tour}: Departure periods could not be determined")
            raise ValueError
        if not latest_departure_period >= earliest_departure_period:
            print(f"Could not discretize tour {tour}: Latest departure is not greater than or equal to earliest departure")
            raise ValueError
        duration = ceil(tour.duration_time / periods[0].duration)

        return DiscreteTour(
            id=tour.id, duration=duration, earliest_departure=earliest_departure_period,
            latest_departure=latest_departure_period, consumption=tour.consumption, cost=tour.cost
        )

    def ToContinuousTour(self) -> Tour:
        return Tour(id=self.id, duration_time=self.duration_time, earliest_departure_time=self.earliest_departure_time,
                    latest_departure_time=self.latest_departure_time, consumption=self.consumption, cost=self.cost)


@dataclass_json
@dataclass
class TourPlan:
    tours: List[Tour] = field(default_factory=list)
    vehicleID: Optional[int] = None

    def __iter__(self) -> Iterator[Tour]:
        return iter(self.tours)

    def __len__(self) -> int:
        return len(self.tours)

    def __getitem__(self, item):
        return self.tours.__getitem__(item)

    def __eq__(self, other):
        return len(self) == len(other) and all(a.equals(b) for a,b in zip(self.tours, other.tours))

    def toXML(self) -> Element:
        xml_tour_plan = Element('VehicleTourPlan')

        for x in self.tours:
            xml_tour_plan.append(x.toXML())

        return xml_tour_plan


@dataclass_json
@dataclass
class FleetTourPlan:
    schedules: List[TourPlan] = field(default_factory=list)
    fleetSize: int = 0

    def plot(self, begin=0, end=None, discrete=False):
        fig, ax = plt.subplots()
        if len(self.schedules) == 0:
            return fig, ax
        y_ticks = []
        major_x_ticks = set()
        for vehicle_index, vehicle_schedule in enumerate(self.schedules):
            tours_to_plot = list(filter(lambda tour: tour.departure_time >= begin and (not end or tour.arrival_time <= end), vehicle_schedule))
            rectangles = [(tour.departure_time, tour.duration) for tour in tours_to_plot]
            y_offset = 10*(vehicle_index+1)
            y_span = 8
            ax.broken_barh(rectangles, (y_offset, y_span), label=vehicle_index, facecolors=('royalblue', 'cornflowerblue'))
            # Add labels
            annotation_y = cycle(range(2, y_span, 2))
            for (x_beg, width), tour, next_y_span in zip(rectangles, tours_to_plot, annotation_y):
                major_x_ticks.union({x_beg, x_beg+width})
                ax.annotate(str(tour.id), (x_beg + width/2, y_offset + next_y_span), color='black', ha='center', va='center')

            y_ticks.append(y_offset + y_span/2)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(f'Vehicle {v_id}' for v_id in range(len(self.schedules)))
        if end:
            ax.set_xlim(begin, end)
        ax.set_ylim(0, y_ticks[-1] + 4 + 10)

        ax.set_xticks(list(major_x_ticks), minor=False)
        ax.grid(True, which='major')

        if discrete:
            ax.set_xticks(list(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]))), minor=True)
            ax.grid(True, which='minor')

        return fig, ax

    def __post_init__(self):
        self.schedules.sort(key=lambda tp: tp.vehicleID)
        if self.fleetSize != len(self.schedules):
            self.fleetSize = len(self.schedules)

    def __getitem__(self, *args, **kwargs):
        return self.schedules.__getitem__(*args, **kwargs)

    def __iter__(self) -> Iterator[TourPlan]:
        return iter(self.schedules)

    def __len__(self):
        return len(self.schedules)

    def toXML(self) -> Element:
        xml_tours = Element('TourPlans')
        for x in self.schedules:
            xml_tours.append(x.toXML())
        return xml_tours
