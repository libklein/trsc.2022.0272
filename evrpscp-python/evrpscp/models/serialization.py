from functools import singledispatch
from typing import List
from funcy import takewhile

from .charger import Charger
from .discretization import DiscretizedInstance, DiscretePeriod, DiscreteTour, DiscreteParameters, decode_periods_from_json
from json import JSONDecoder, JSONEncoder, dumps, loads


def serialize_discrete_instance(instance: DiscretizedInstance) -> str:
    return instance.to_json()


def deserialize_discrete_instance(instance_json: str) -> DiscretizedInstance:
    instance_dict = loads(instance_json)
    instance_dict['periods'] = decode_periods_from_json(instance_dict['periods'])
    instance_dict['chargers'] = [Charger.from_dict(x) for x in instance_dict['chargers']]
    # Tours
    instance_dict['tours'] = [[DiscreteTour.from_dict(t, periods=instance_dict['periods']) for t in veh_tours]
                              for veh_tours in instance_dict['tours']]
    # Parameters
    instance_dict['parameters'] = DiscreteParameters.from_dict(instance_dict['parameters'])
    return DiscretizedInstance(**instance_dict)
