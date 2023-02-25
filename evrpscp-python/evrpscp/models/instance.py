from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement
from typing import List, Optional, Dict, Tuple, Set, Union, Any
from dataclasses import field
from dataclasses_json import dataclass_json, config
from terminaltables import AsciiTable

from . import Period, DiscretePeriod, Charger, Parameters, FleetTourPlan, PeriodVertex


@dataclass_json
@dataclass
class Instance:
    periods: Union[List[Period], List[DiscretePeriod]] = field(metadata={'dataclasses_json': {
        # Default to Period repr
        'decoder': lambda encoded: Period.schema().load(encoded, many=True)
    }})
    chargers: List[Charger]
    param: Parameters

    def __post_init__(self):
        self.chargers.sort(key=lambda x: x.id)
        self.periods.sort(key=lambda x: x.begin)

        assert self.param.battery.minimumCharge == 0
        for f in self.chargers:
            assert f.chargingFunction.image_upper_bound == self.param.battery.maximumCharge
            assert f.chargingFunction.image_lower_bound == self.param.battery.minimumCharge

    def getPeriod(self, t: float) -> Period:
        return next((x for x in self.periods if x.begin <= t < x.end), None)

    def getCharger(self, stationID: int) -> Charger:
        return next((x for x in self.chargers if x.id == stationID), None)

    def dump_to_console(self):
        print("--------------------------Periods------------------------------------")
        print(plot_periods(self.periods))
        print("--------------------------Chargers-----------------------------------")
        for x in self.chargers:
            print(x.dump())
        print("------------------Wearcost Density Function--------------------------")
        print(self.param.battery.wearCostDensityFunction.dump())
        print("--------------------------Parameters---------------------------------")
        print(self.param)


@dataclass_json
@dataclass
class SchedulingInstance(Instance):
    tourPlans: Optional[FleetTourPlan] = None

    def dump_to_console(self):
        super().dump_to_console()
        print("---------------------------Tour Plans--------------------------------")
        print(plot_fleet_tours(self.tourPlans))
        print("---------------------------------------------------------------------")

def format_time(t) -> str:
    return f'{int((round(t, 2)//60) % 24):02}:{int(round(t,2) % 60):02}+{int(round(t,2) // (60*24))} ({round(t, 2)})'

def plot_fleet_tours(tours) -> str:
    max_tours_assigned = max(map(lambda x: len(x), tours))
    tour_table_rows = [['Vehicle']]
    tour_table_rows[0].extend(
        ['Tour', 'Earliest Departure', 'Latest Departure', 'Earliest Arrival', 'Latest Arrival', 'Duration', 'Consumption', 'Cost'] * max_tours_assigned)
    for vehicle_id, vehicle_tours in enumerate(tours):
        tour_table_rows.append([vehicle_id])
        for tour in vehicle_tours:
            tour_table_rows[-1].extend([tour,
                                        format_time(tour.earliest_departure_time),
                                        format_time(tour.latest_departure_time),
                                        format_time(tour.earliest_arrival_time),
                                        format_time(tour.latest_arrival_time),
                                        format_time(tour.duration_time),
                                        round(tour.consumption, 2),
                                        round(tour.cost, 2)])
    return AsciiTable(tour_table_rows).table

def plot_periods(periods) -> str:
    rows = [('Begin', 'End', 'Energy Price')]
    rows += [(format_time(p.begin), format_time(p.end), f'{p.energyPrice}$') for p in periods]
    print(AsciiTable(rows).table)
