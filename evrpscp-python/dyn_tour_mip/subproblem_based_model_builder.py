# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from itertools import product, chain
from operator import attrgetter
from typing import List, Union, Set, Dict, Tuple, Iterable, Optional, Callable

from bidict import bidict
from docplex.mp.linear import Var, LinearExpr, ConstantExpr
from docplex.mp.model import Model
from evrpscp import Charger, DiscreteTour, DiscretePeriod, DiscretizedInstance, PiecewiseLinearSegment, \
    FleetChargingSchedule
from funcy import flatten, cached_readonly, group_by
from graphviz import dot

from .models import PWLBreakpoint, Vehicle
from column_generation.subproblem.mip.subproblem import SubProblem as MIPSubproblem

def _add_to_model(model: Model, other_model: Model, rename_func: Optional[Callable]):
    """
    Adds all variables, constraints, etc. in `other_model` to `model`.
    """
    assert other_model.number_of_linear_constraints == other_model.number_of_constraints
    assert other_model.number_of_sos1 == 0
    # Variables
    var_mapper = bidict({})
    for var in other_model.iter_variables():
        var_mapper[var] = model.var(var.vartype, var.lb, var.ub, var.name if rename_func is None else rename_func(var))

    def translate_expr(expr):
        new_expr = model.sum(var_mapper[var] * coef for var, coef in expr.iter_terms())
        if isinstance(expr, LinearExpr) or (isinstance(expr, ConstantExpr) and expr.number_of_variables() == 0):
            new_expr += expr.constant
        return new_expr

    # Constraints
    constr_mapper = bidict({})
    for constr in other_model.iter_constraints():
        constr_mapper[constr] = model.add_constraint(
            model.linear_constraint(
                lhs=translate_expr(constr.get_left_expr()),
                ctsense=constr.type,
                rhs=translate_expr(constr.get_right_expr())
            ), ctname=constr.name if not rename_func else rename_func(constr))
    # SOS2 sets
    for sos2 in other_model.iter_sos2():
        model.add_sos2([var_mapper[v] for v in sos2.iter_variables()])

    # Add to objective
    model.set_objective(model.objective_sense, model.objective_expr + translate_expr(other_model.objective_expr))

    return model, var_mapper, constr_mapper

class DynamicTourModel:

    def __init__(self, instance: DiscretizedInstance, **cli_args):
        self.instance = instance
        self.vehicles = instance.vehicles[:1]
        self.tours = instance.tours
        self.params = instance.parameters
        self.battery = self.params.battery
        self.chargers = instance.chargers
        self.periods = instance.periods
        self._cli_args = cli_args

        self.model = Model(f'Dynamic Tour EVRP-SCP')
        self.model.set_objective('min', self.model.linear_expr())

        # Create subproblems
        self._subproblems: Dict[Vehicle, MIPSubproblem] = \
            {k: MIPSubproblem(instance=instance, vehicle=k, **cli_args) for k in self.vehicles}

        # Merge models
        # Maps [k, old_var] to var in new model
        self.vehicle_var_map = bidict({})
        # Maps [k, old_constr] to constr in new model
        self.vehicle_constr_map = bidict({})
        # Maps [k, p, f] to x in new model
        self._x_map = bidict({})
        for k, subproblem in self._subproblems.items():
            _, var_map, constr_map = _add_to_model(self.model, subproblem.model, rename_func=lambda x: f'v{k}-{x.name}')
            self.vehicle_var_map.update(((k, key), value) for key, value in var_map.items())
            self.vehicle_constr_map.update(((k, key), value) for key, value in constr_map.items())
            for arc in subproblem.network.arcs:
                self._x_map[k, arc.origin.period, arc.origin.charger] = var_map[arc.x]

        self._create_capacity_constraits()

    @property
    def cplex(self) -> Model:
        return self.model

    def solve(self):
        pass

    def set_parameters(self, **kwargs):
        self.cplex.display = 1
        self.cplex.log_output = True
        self.cplex.parameters.threads = kwargs.get('threads', 1)
        self.cplex.time_limit = kwargs.get('time_limit')
        self.cplex.parameters.mip.polishafter.time = self.model.cplex.time_limit - 60

    def render_solution(self, vehicle: int = 0) -> dot.Digraph:
        raise NotImplementedError

    def display_solution(self, *args, **kwargs):
        self.render_solution(*args, **kwargs).render(view=True)

    def _create_capacity_constraits(self):
        for p in self.periods:
            for f in self.chargers:
                capacity_sum = self.model.sum(self._x_map.get((k, p, f), 0) for k in self.vehicles)
                self.model.add_constraint(capacity_sum <= f.capacity, ctname=f"Capacity-{p}-{f}")

    def create_fleet_charging_schedule(self) -> FleetChargingSchedule:
        #operations = [
        #    [_create_operation(arc) for arc in subproblem.network.arcs for ]
        #    for k, subproblem in self._subproblems.items()
        #]
        raise NotImplementedError

    def validate_solution(self):
        # TODO Validate each network individually

        # TODO Validate capacity constraint
        raise NotImplementedError