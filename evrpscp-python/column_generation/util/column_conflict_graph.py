# coding=utf-8
import itertools
from collections import defaultdict
from itertools import product
from typing import List, Dict, Tuple, Union, Set
from graphviz import Graph

from column_generation.solution import Solution, Column
from evrpscp import Charger, DiscretePeriod


def _compute_convexity_constraint_edges(sol: Solution) -> List[Tuple[Tuple[Column, Column], int]]:
    conflicts = []
    for i in sol.columns:
        for j in sol.columns:
            if i.id >= j.id:
                continue
            if i.vehicle == j.vehicle:
                conflicts.append(((i, j), i.vehicle))
    return conflicts


def _compute_capacity_constraint_edges(sol: Solution, periods: List[DiscretePeriod], chargers: List[Charger]):
    conflicts: Dict[Tuple[Column, Column], List[Tuple[DiscretePeriod, Charger]]] = defaultdict(list)
    conflict_counts: Dict[Tuple[DiscretePeriod, Charger], Set[int]] = defaultdict(set)
    for i in sol:
        for j in sol:
            if i.id >= j.id:
                continue
            for p, f in product(periods, chargers):
                if i.charger_usage[p, f] and j.charger_usage[p, f]:
                    conflicts[i, j].append((p, f))
                    conflict_counts[p, f].update((i.id, j.id))

    # Identify periods/chargers where conflicts occur
    conflicts = {key: [(p, f) for (p, f) in confs if len(conflict_counts[p, f]) >= f.capacity]
                 for key, confs in conflicts.items()}
    return conflicts


def draw_column_conflict_graph(sol: Solution, vehicles: List[int],
                               periods: List[DiscretePeriod],
                               chargers: List[Charger]) -> Graph:
    if sol.integral:
        return
    assert all(not x.is_dummy for x in sol)

    edges: Dict[Tuple[Column, Column], List[Union[Tuple[DiscretePeriod, Charger], int]]] = defaultdict(list)
    # Get edges from convexity constraints
    for edge, veh in _compute_convexity_constraint_edges(sol):
        edges[edge].append(veh)

    # Get capacity constraint edges
    for edge, conflicts in _compute_capacity_constraint_edges(sol, periods, chargers).items():
        edges[edge].extend(conflicts)

    g = Graph(name='Conflict Graph')

    for col, x in sol.items():
        # Draw nodes
        g.node(name=str(col.id), label=f'{col.id}: {x:.4f}')

    color_mapper = {x: 'black' for x in vehicles}
    COLORS = itertools.cycle(['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'brown', 'pink'])

    for edge, constraints in edges.items():
        for constr in constraints:
            if (color := color_mapper.get(constr)) is None:
                color_mapper[constr] = color = next(COLORS)
            g.edge(tail_name=str(edge[0].id), head_name=str(edge[1].id), color=color)

    return g