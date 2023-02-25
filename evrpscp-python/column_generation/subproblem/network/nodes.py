# coding=utf-8
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import List, Iterator, Dict, Optional, Any, ClassVar, Callable
from enum import Enum

from column_generation import EPS
from funcy import *
from graphviz import Digraph

from evrpscp import DiscretizedInstance, Charger, DiscreteTour, Vehicle, DiscretePeriod, Battery

class NodeType(Enum):
    Source = 'SourceNode'
    Sink = 'SinkNode'
    Station = 'StationNode'
    Idle = 'IdleNode'

@dataclass
class Node:
    period: Optional[DiscretePeriod]
    charger: Optional[Charger]
    node_type: NodeType
    id: int = field(default=-1, init=False, compare=False)
    outgoing_arcs: List['Arc'] = field(default_factory=list, init=False, compare=False)
    incoming_arcs: List['Arc'] = field(default_factory=list, init=False, compare=False)
    _next_id: ClassVar[int] = 0

    _entry_soc: float = field(default=0.0, init=False, repr=False, compare=False)
    _deg_cost: float = field(default=0.0, init=False, repr=False, compare=False)
    _charge: float = field(default=0.0, init=False, repr=False, compare=False)

    def __eq__(self, other):
        return self.period == other.period and self.charger == other.charger

    def __post_init__(self):
        assert self.outgoing_arcs is not None
        if self.id < 0:
            self.id = Node._next_id
            Node._next_id += 1

    def __hash__(self):
        return hash(id(self))

    def __iter__(self) -> Iterator['Arc']:
        return iter(self.outgoing_arcs)

    @property
    def is_idle_node(self):
        return self.node_type == NodeType.Idle

    @property
    def is_source_node(self):
        return self.node_type == NodeType.Source

    @property
    def is_sink_node(self):
        return self.node_type == NodeType.Sink

    @property
    def is_station_node(self):
        return self.node_type == NodeType.Station

    @property
    def exit_soc(self):
        return self.entry_soc + self.charge

    @property
    def entry_soc(self) -> float:
        return self._entry_soc

    @entry_soc.setter
    def entry_soc(self, value: float):
        self._entry_soc = value

    @property
    def deg_cost(self):
        return self._deg_cost

    @deg_cost.setter
    def deg_cost(self, value: float):
        self._deg_cost = value

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value: float):
        self._charge = value

    @property
    def neighbors(self) -> Iterator['Node']:
        return (x.target for x in self.outgoing_arcs)

    def get_arc_to(self, node: 'Node') -> 'Arc':
        for arc in self.outgoing_arcs:
            if arc.target is node:
                return arc
        raise ValueError(f'Node {self} is not connected to {node}')

    def add_incomming_arc(self, arc: 'Arc') -> 'Arc':
        self.incoming_arcs.append(arc)
        return arc

    def add_outgoing_arc(self, arc: 'Arc') -> 'Arc':
        self.outgoing_arcs.append(arc)
        return arc

    def _remove_arc(self, arc: 'Arc'):
        if arc.origin is self:
            self.outgoing_arcs.remove(arc)
        else:
            self.incoming_arcs.remove(arc)

    def _remove(self):
        for arc in self.outgoing_arcs[:]:
            arc._remove()
        assert len(self.outgoing_arcs) == 0

        for arc in self.incoming_arcs[:]:
            arc._remove()
        assert len(self.incoming_arcs) == 0


    def check_solution(self):
        assert self.entry_soc >= -EPS
        assert sum(x.is_active for x in self.outgoing_arcs) <= 1, f'Vertex {self} has more than 1 outgoing arc!'
        if self.charge > 0:
            assert self.deg_cost > 0
            assert self.charger is not None
            assert self.charger.charge_for(self.entry_soc, self.period.duration) >= self.exit_soc - EPS
        else:
            assert self.deg_cost <= EPS

    def render(self, graph: Digraph, *args, **kwargs):
        node_name = str(self.id)

        defaults = {
            'label': str(self)
        }
        defaults.update(kwargs)

        attrs = {}
        if self.model.solution is not None:
            attrs['label'] = f'{str(self)}: beta={self.entry_soc}, rho={self.deg_cost}, gamma={self.charge}'
        attrs.update(kwargs)
        return super(MIPNode, self).render(graph, *args, **attrs)

        graph.node(name=node_name, **defaults)
        return node_name

    def update_references(self, new_references: Dict[int, 'Arc']):
        self.incoming_arcs = [new_references[id(x)] for x in self.incoming_arcs]
        self.outgoing_arcs = [new_references[id(x)] for x in self.outgoing_arcs]

    def __deepcopy__(self, memodict={}):
        assert id(self) not in memodict
        carbon_copy = copy(self)
        memodict[id(self)] = carbon_copy
        carbon_copy.outgoing_arcs = []
        for arc in self.outgoing_arcs:
            if id(arc) not in memodict:
                deepcopy(arc, memodict)
            carbon_copy.outgoing_arcs.append(memodict[id(arc)])
        carbon_copy.incoming_arcs = []
        for arc in self.incoming_arcs:
            if id(arc) not in memodict:
                deepcopy(arc, memodict)
            carbon_copy.incoming_arcs.append(memodict[id(arc)])
        return carbon_copy

    def __str__(self) -> str:
        return f'{self.period} ({self.charger})'

    def __repr__(self) -> str:
        return str(self)

@dataclass
class SourceNode(Node):

    def __init__(self, *args, **kwargs):
        super(SourceNode, self).__init__(*args, node_type=NodeType.Source, **kwargs)

    def __hash__(self):
        return super().__hash__()

    def __str__(self) -> str:
        return 'Source'

    __repr__ = __str__

    def check_solution(self):
        assert sum(x.is_active for x in self.outgoing_arcs) <= 1, f'Source vertex has no outgoing arc!'

@dataclass
class SinkNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, node_type=NodeType.Sink, **kwargs)

    def __hash__(self):
        return super().__hash__()

    def __str__(self) -> str:
        return 'Sink'

    __repr__ = __str__

@dataclass
class StationNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, node_type=NodeType.Station, **kwargs)

    def __hash__(self):
        return super().__hash__()

    def __str__(self) -> str:
        return f'{self.period} ({self.charger})'

    __repr__ = __str__

@dataclass
class IdleNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, node_type=NodeType.Idle, **kwargs)

    def __hash__(self):
        return super().__hash__()

    def __str__(self) -> str:
        return f'{str(self.period)} (Idle)'

    __repr__ = __str__

