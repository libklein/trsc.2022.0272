# coding=utf-8
import dataclasses
from copy import copy, deepcopy
from dataclasses import dataclass, field, InitVar
from typing import List, Iterator, Dict, Optional, Any, ClassVar, Callable
from enum import Enum

from graphviz import Digraph

from .nodes import Node, NodeType
from evrpscp import DiscretizedInstance, Charger, DiscreteTour, Vehicle, DiscretePeriod, Battery

from funcy import *

class ArcType(Enum):
    Source = 'SourceArc'
    Charging = 'ChargingArc'
    Idle = 'IdleArc'
    Service = 'ServiceArc'

@dataclass
class Arc:
    _next_id: ClassVar[int] = 0
    origin: Node
    target: Node
    consumption: float
    arc_type: ArcType
    cost: InitVar[float]
    id: int = field(default=-1, init=False, compare=False)

    _is_active: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        self._is_active = value

    def __post_init__(self, cost: float):
        assert self.origin is not None and self.target is not None
        self.cost = cost
        if self.id < 0:
            self.id = Arc._next_id
            Arc._next_id += 1

    def check_solution(self):
        pass

    def __str__(self):
        return f'({self.arc_type.value}: {self.origin} -> {self.target} | {-self.cost}$ | {-self.consumption}q)'

    __repr__ = __str__

    def __hash__(self):
        return hash(id(self))

    def _remove(self):
        self.origin._remove_arc(self)
        self.target._remove_arc(self)

    def render(self, graph: Digraph, *args, **kwargs):
        defaults = {
            'label': f'pi: {self.cost}, q: {-self.consumption}'
        }
        defaults.update(kwargs)

        head_name = self.target.render(graph, *args, **kwargs)
        if head_name is None:
            return None
        tail_name = self.origin.render(graph, *args, **kwargs)
        if tail_name is None:
            return None
        return graph.edge(tail_name=tail_name, head_name=head_name, **defaults)

    def update_references(self, new_references: Dict[int, 'Node']):
        self.origin = new_references[id(self.origin)]
        self.target = new_references[id(self.target)]

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError
        carbon_copy = copy(self)
        assert id(self) not in memodict
        memodict[id(self)] = carbon_copy
        if id(self.origin) not in memodict:
            deepcopy(self.origin, memodict)
        carbon_copy.origin = memodict[id(self.origin)]
        if id(self.target) not in memodict:
            deepcopy(self.target, memodict)
        carbon_copy.target = memodict[id(self.target)]
        return carbon_copy

@dataclass
class SourceArc(Arc):
    def __post_init__(self, *args, **kwargs):
        assert self.origin.node_type == NodeType.Source
        super().__post_init__(*args, **kwargs)

    __hash__ = Arc.__hash__

@dataclass
class ChargingArc(Arc):
    @property
    def charger(self) -> Charger:
        return self.origin.charger

    def check_solution(self):
        super(ChargingArc, self).check_solution()
        pass

    def __post_init__(self, *args, **kwargs):
        assert self.origin.node_type == NodeType.Station
        super().__post_init__(*args, **kwargs)

    __hash__ = Arc.__hash__

@dataclass
class IdleArc(Arc):
    def __post_init__(self, *args, **kwargs):
        assert self.origin.node_type == NodeType.Idle
        super().__post_init__(*args, **kwargs)

    __hash__ = Arc.__hash__

@dataclass
class ServiceArc(Arc):
    tour: DiscreteTour

    def __post_init__(self, *args, **kwargs):
        assert self.tour.earliest_departure <= self.origin.period
        assert self.tour.latest_departure >= self.origin.period
        assert nth(self.tour.duration, iter(self.origin.period)) == self.target.period, \
            f'Trying to create service arc from period {self.origin.period} to {self.target.period} but duration of tour' \
            f' is {self.tour.duration}'
        super().__post_init__(*args, **kwargs)

    def __str__(self):
        return f'{str(self.tour)} | ({self.origin} -> {self.target} | {-self.cost}$ | {-self.consumption}q)'

    __repr__ = __str__

    def render(self, graph: Digraph, *args, **kwargs):
        head_name = self.target.render(graph)
        if head_name is None:
            return None
        tail_name = self.origin.render(graph)
        if tail_name is None:
            return None
        return graph.edge(tail_name=tail_name, head_name=head_name, label=f'pi: {self.cost}, q: {-self.consumption}', style='dashed')

    __hash__ = Arc.__hash__
