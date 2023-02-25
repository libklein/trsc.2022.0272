# coding=utf-8
from __future__ import annotations
from enum import Enum
from itertools import product
from typing import List, Optional, Set, Iterable
from math import inf

from column_generation.constraints import Constraint
from evrpscp import EPS


class Node:
    def __init__(self, parent: Optional[Node], constraints: Set[Constraint]):
        self._parent: Optional[Node] = parent
        self._local_constraints: Set[Constraint] = constraints
        self._children: List[Node] = []
        self._lower_bound: float = -inf

        # Initialize the lower bound with the lower bound of the parent.
        # We always add constraints when decending the tree, hence any parent LB will be a valid LB for the child.
        if not self.is_root_node:
            self._lower_bound = self.parent.lower_bound

        for c1, c2 in product(self.global_constraints, self.global_constraints):
            if (c1 is not c2) and c1.conflicts(c2):
                raise ValueError(f"Created node with conflicting constraints: {c1}, {c2}")

    def add_child(self, child: Node):
        self._children.append(child)

    @property
    def children(self) -> Iterable[Node]:
        return iter(self._children)

    @property
    def id(self) -> str:
        if self.is_root_node:
            return '1'
        else:
            return f'{self.parent.id}.{self.parent._children.index(self)}'

    def __str__(self) -> str:
        return self.id

    @property
    def is_root_node(self) -> bool:
        return self._parent is None

    @property
    def is_leaf_node(self) -> bool:
        return len(self._children) == 0

    @property
    def parent(self) -> Node:
        if self._parent is None:
            raise ValueError('Node has no parent!')
        return self._parent

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, new_lb: float):
        if self._lower_bound <= new_lb + EPS:
            self._lower_bound = new_lb

    @property
    def local_constraints(self) -> Set[Constraint]:
        return self._local_constraints

    @property
    def global_constraints(self) -> Set[Constraint]:
        if self._parent is None:
            return self._local_constraints
        return self.parent.global_constraints | self._local_constraints