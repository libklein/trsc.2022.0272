from .piecewise_linear_function import PiecewiseLinearSegment, PiecewiseLinearFunction
from .charger import Charger
from .battery import Battery
from .period import Period, DiscretePeriod, PeriodVertex, find_intersecting_period, is_within_period
from .parameters import Parameters, DiscreteParameters
from .routing_solution import Route, PeriodRoutingSolution, RoutingSolution
from .tours import Tour, DiscreteTour, TourPlan, FleetTourPlan
from .instance import SchedulingInstance
from .charging_schedule import Operation, ChargingOperation, VehicleDeparture,\
    FleetSchedule, FleetChargingSchedule, VehicleChargingSchedule
from .discretization import DiscretizedInstance

from .serialization import serialize_discrete_instance, deserialize_discrete_instance

Vehicle = int