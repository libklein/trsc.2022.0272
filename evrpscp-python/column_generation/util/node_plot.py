# coding=utf-8
from typing import List, Dict, Iterable
import graphviz
from column_generation.node import Node


class BranchAndBoundTreePlotter:
    # STATUS_TO_STYLE = {
    #     NodeSolveStatus.UNSOLVED: {'style': 'dashed'},
    #     NodeSolveStatus.INFEASIBLE: {'color': 'red', 'style': 'filled', 'shape': 'box'},
    #     NodeSolveStatus.OPTIMAL: {'color': 'green', 'shape': 'box'},
    #     NodeSolveStatus.PRUNED: {'style': 'dotted', 'shape': 'polygon', 'color': 'red'},
    #     NodeSolveStatus.NON_INTEGRAL: {}
    # }

    def __init__(self, global_constraints=False, best_objective=None):
        self.dot = graphviz.Digraph(name='BnB-Tree')
        self.node_names: Dict[Node, str] = {}
        self.global_constraints = global_constraints
        self.best_objective = best_objective

    def _get_children(self, node: Node) -> Iterable[Node]:
        return node.children

    def _add_node(self, node: Node):
        new_name = self.node_names[node] = f'Node {node.id}' if not node.is_root_node else f'Root'
        return new_name

    def _process_node(self, node: Node) -> str:
        name = self._add_node(node)
        self.dot.node(name, label=f'{name} | LB: {node.lower_bound:.2f}'
                                  f' | Constrs: {node.global_constraints if self.global_constraints else node.local_constraints}')
        # self.dot.node(name, label=f'{name} | LB: {node.lower_bound:.2f} | Constrs: {node.global_constraints if self.global_constraints else node.local_constraints}',
        #               bgcolor='green' if node.objective_value == self.best_objective else 'white',
        #               **BranchAndBoundTreePlotter.STATUS_TO_STYLE[node.status])

        for child in self._get_children(node):
            self.dot.edge(name, self._process_node(child))

        return name

    def __call__(self, node: Node) -> graphviz.Digraph:
        self._process_node(node)
        return self.dot

    def plot(self, *args, view=True, **kwargs):
        self.dot.render(*args, view=view, **kwargs)