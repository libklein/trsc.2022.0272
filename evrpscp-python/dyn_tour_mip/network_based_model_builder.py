# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass, InitVar, field
from functools import singledispatch, singledispatchmethod
from itertools import product, chain
from operator import attrgetter
from typing import List, Union, Set, Dict, Tuple, Iterable, Optional, Callable

from bidict import bidict
from column_generation import Column
from docplex.mp.linear import Var, LinearExpr, ConstantExpr
from docplex.mp.model import Model
from evrpscp import Charger, DiscreteTour, DiscretePeriod, DiscretizedInstance, \
    PiecewiseLinearSegment, FleetChargingSchedule, DiscreteParameters
from column_generation.subproblem.network import *
from funcy import *
from graphviz import dot

from .models import PWLBreakpoint, Vehicle

PERIOD_LENGTH = 30.0

@dataclass
class PhiBreakpoint:
    time: float
    soc: float

    def __hash__(self):
        return id(self)

@dataclass
class WDFBreakpoint:
    soc: float
    cost: float

    def __hash__(self):
        return id(self)


class DynamicTourModel:

    def __init__(self, instance: DiscretizedInstance, **cli_args):
        self.instance = instance
        self.vehicles = instance.vehicles
        self.tours = instance.tours
        self.params = instance.parameters
        self.battery = self.params.battery
        self.chargers = instance.chargers
        self.periods = instance.periods
        self._cli_args = cli_args

        self.breakpoints_per_charger = {f: [PhiBreakpoint(time=time, soc=soc) for (time, soc) in f.breakpoints]
                                        for f in self.chargers}
        self.wdf_breakpoints = [WDFBreakpoint(soc=soc, cost=cost) for (soc, cost) in
                                zip(self.battery.wearCostDensityFunction.breakpoints,
                                    self.battery.wearCostDensityFunction.image_breakpoints)]

        self.model = Model(f'Dynamic Tour EVRP-SCP')

        self._networks: Dict[Vehicle, TimeExpandedNetwork] = \
            {k: TimeExpandedNetwork(instance=instance, vehicle=k) for k in self.vehicles}
        self._x: Dict[Vehicle, Dict[Arc, Var]] = {}
        self._beta: Dict[Vehicle, Dict[Node, Var]] = {}
        self._gamma: Dict[Vehicle, Dict[StationNode, Var]] = {}
        self._rho: Dict[Vehicle, Dict[StationNode, Var]] = {}
        self._lambda_entry: Dict[Vehicle, Dict[StationNode, Dict[PhiBreakpoint, Var]]] = {}
        self._lambda_exit: Dict[Vehicle, Dict[StationNode, Dict[PhiBreakpoint, Var]]] = {}
        self._mu_entry: Dict[Vehicle, Dict[StationNode, Dict[WDFBreakpoint, Var]]] = {}
        self._mu_exit: Dict[Vehicle, Dict[StationNode, Dict[WDFBreakpoint, Var]]] = {}

        for k, network in self._networks.items():
            self._apply_optimizations(network=network, tours=self.tours[k])
            self._compute_network_bounds(network=network, tours=self.tours[k])
            x, beta, gamma, rho, lambda_entry, lambda_exit, mu_entry, mu_exit = \
                self._build_model_from_network(network=network, vehicle=k)
            for target_dict, child in zip((self._x, self._beta, self._gamma, self._rho, self._lambda_entry, self._lambda_exit, self._mu_entry, self._mu_exit), (x, beta, gamma, rho, lambda_entry, lambda_exit, mu_entry, mu_exit)):
                target_dict[k] = child

        self._create_capacity_constraits()
        self.model.set_objective('min', self._create_objective())

        # Valid inequalities
        self._add_single_node_per_period_valid_ineq()

    def _apply_optimizations(self, network, tours):
        optimizations = [
            partial(remove_covered_periods, network),
            partial(remove_superfluous_sink_periods, network, tours),
            partial(remove_dead_ends, network),
            partial(remove_idle_chains, network)
        ]
        removal_stats = apply_optimizations(*optimizations)
        print(f'Optimizer stats: {removal_stats}, total removal: {sum(removal_stats.values())}')
        return removal_stats

    @property
    def cplex(self) -> Model:
        return self.model

    def _set_network_solutions(self):
        for k, network in self._networks.items():
            for node in network.nodes:
                node.entry_soc = self._beta[k][node].solution_value
                if isinstance(node, StationNode):
                    node.charge = self._gamma[k][node].solution_value
                    node.deg_cost = self._rho[k][node].solution_value
            for arc in network.arcs:
                arc.is_active = self._x[k][arc].to_bool()

    def solve(self):
        sol = self.model.solve()

        if sol is not None:
            self._set_network_solutions()
        return sol

    def set_parameters(self, **kwargs):
        self.cplex.display = 1
        self.cplex.log_output = True
        self.cplex.parameters.threads = kwargs.get('threads', 1)
        self.cplex.time_limit = kwargs.get('time_limit')
        self.cplex.parameters.mip.polishafter.time = self.cplex.time_limit - 60
        self.cplex.parameters.mip.limits.solutions = kwargs.get('solution_limit', 1000000)

    def render_solution(self, vehicle: int = 0) -> dot.Digraph:
        return self._networks[vehicle].render()

    def display_solution(self, *args, **kwargs):
        self.render_solution(*args, **kwargs).render(view=True)

    def _create_column(self, k: Vehicle) -> Column:
        return self._networks[k].create_column(self.chargers, k, self.model.kpi_value_by_name(f'ObjV{k}'))

    def _create_columns(self) -> Dict[Vehicle, Column]:
        return {k: self._create_column(k) for k in self.vehicles}


    def create_fleet_charging_schedule(self) -> FleetChargingSchedule:
        #operations = [
        #    [_create_operation(arc) for arc in subproblem.network.arcs for ]
        #    for k, subproblem in self._subproblems.items()
        #]
        columns = self._create_columns()
        for k, col in columns.items():
            print(k)
            print(repr(col))

        schedules = [col.create_vehicle_schedule(self.instance) for col in columns.values()]
        schedules.sort(key=lambda x: x.vehicleID)
        fleet_schedule = FleetChargingSchedule(vehicleSchedules=schedules)
        fleet_schedule.calculate_cost(self.instance.parameters.battery)
        return fleet_schedule


    def validate_solution(self):
        # Validate each network individually
        for k, network in self._networks.items():
            network._check_solution()
        # TODO Validate capacity constraint

    def _add_single_node_per_period_valid_ineq(self):
        for k, network in self._networks.items():
            self.model.add_constraints(self.model.sum(self._x[k][arc] for node in nodes for arc in node.outgoing_arcs) <= 1 for nodes in network.nodes_by_period.values())

    def _compute_network_bounds(self, network, tours):
        # Compute lower bounds etc.
        compute_required_service(network=network, operations=tours)
        compute_lower_bound(network=network, wdf=self.battery.wearCostDensityFunction)

    def _create_capacity_constraits(self):
        charger_usage_vars: Dict[Tuple[DiscretePeriod, Charger], List[Var]] = defaultdict(list)
        for k, network in self._networks.items():
            for arc in network.arcs_by_type(ChargingArc):
                charger_usage_vars[arc.origin.period, arc.origin.charger].append(self._x[k][arc])
        self.model.add_constraints(
            (self.model.sum(usage_vars) <= f.capacity for (_, f), usage_vars in charger_usage_vars.items()),
            names=(f'capacity_{f}_in_{p}' for (p, f) in charger_usage_vars.keys())
        )

    def _create_objective(self):
        veh_objs = []
        for k, network in self._networks.items():
            veh_obj = self.model.linear_expr()
            for node in network.nodes_by_type(StationNode):
                veh_obj += self._rho[k][node] + (self._gamma[k][node] * node.period.energyPrice)
            self.model.add_kpi(veh_obj, publish_name=f'ObjV{k}')
            veh_objs.append(veh_obj)
        return self.model.sum(veh_objs)


    def _build_model_from_network(self, network: TimeExpandedNetwork, vehicle: Vehicle):
        x: Dict[Arc, Var] = self.model.binary_var_dict(network.arcs, name=f'v{vehicle}-x')

        beta: Dict[Node, Var] = self.model.continuous_var_dict(network.nodes, lb=0.0, ub=self.params.battery.maximumCharge, name=f'v{vehicle}-beta')
        max_charge_per_period = max(f.charge_for(0.0, PERIOD_LENGTH) for f in self.chargers)
        gamma: Dict[StationNode, Var] = self.model.continuous_var_dict(network.nodes_by_type(StationNode), ub=max_charge_per_period, name=f'v{vehicle}-beta')
        rho: Dict[StationNode, Var] = self.model.continuous_var_dict(network.nodes_by_type(StationNode),
                                                                     ub=self.battery.wearCostDensityFunction(self.battery.maximumCharge)
                                                                        - self.battery.wearCostDensityFunction(self.battery.maximumCharge - max_charge_per_period), name=f'v{vehicle}-beta')

        # Phi(s)
        lambda_entry = self._add_pwl_constr(nodes=network.nodes_by_type(StationNode), get_node_bps=lambda node: self.breakpoints_per_charger[node.charger], name=f'v{vehicle}-phi_in')
        lambda_exit = self._add_pwl_constr(nodes=network.nodes_by_type(StationNode), get_node_bps=lambda node: self.breakpoints_per_charger[node.charger], name=f'v{vehicle}-phi_out')
        # Link entry/exit to beta and gamma
        self._link_pwl_constr(entry_vars=lambda_entry, exit_vars=lambda_exit, beta=beta, gamma=gamma, name=f'pwl')
        # Bind time
        self.model.add_constraints(
            self.model.sum(var * bp.time for bp, var in lambda_exit[node].items())
            - self.model.sum(var * bp.time for bp, var in lambda_entry[node].items())
            <= PERIOD_LENGTH
            * self.model.sum(x[arc] for arc in node.outgoing_arcs if arc.arc_type is ArcType.Charging) for node in lambda_entry)

        # WDF
        mu_entry = self._add_pwl_constr(nodes=network.nodes_by_type(StationNode), get_node_bps=lambda node: self.wdf_breakpoints, name=f'v{vehicle}-wdf_in')
        mu_exit = self._add_pwl_constr(nodes=network.nodes_by_type(StationNode), get_node_bps=lambda node: self.wdf_breakpoints, name=f'v{vehicle}-wdf_out')
        # Link entry/exit to beta and gamma
        self._link_pwl_constr(entry_vars=mu_entry, exit_vars=mu_exit, beta=beta, gamma=gamma, name=f'mu')
        # Bind to rho
        self.model.add_constraints(
            self.model.sum(var * bp.cost for bp, var in mu_exit[node].items())
            - self.model.sum(var * bp.cost for bp, var in mu_entry[node].items())
            == rho[node] for node in mu_entry)

        # Leave source on one arc
        self.model.add_constraint(self.model.sum(x[var] for var in network.source.outgoing_arcs) == 1)
        # Each operation is serviced
        ops_by_tour = group_by(lambda arc: arc.tour, network.arcs_by_type(ServiceArc))
        self.model.add_constraints(self.model.sum(x[arc] for arc in service_arcs_of_tour) == 1
                                   for service_arcs_of_tour in ops_by_tour.values())
        # Flow conservation
        self.model.add_constraints(self.model.sum(x[arc] for arc in node.outgoing_arcs) - self.model.sum(x[arc] for arc in node.incoming_arcs) == 0 for node in network.nodes if node not in (network.source, network.sink))
        # Charge propagation
        self.model.add_constraints(beta[arc.origin] - arc.consumption + gamma.get(arc.origin, 0)
                                  >= beta[arc.target] - (1 - x[arc]) * self.params.battery.maximumCharge
                                  for arc in network.arcs)
        # Soc Bounds
        self.model.add_constraints(self.params.battery.minimumCharge <= beta[node] for node in network.nodes)
        self.model.add_constraints(beta[node] <= self.params.battery.maximumCharge for node in network.nodes)
        # Initial SoC
        self.model.add_constraint(beta[network.source] == 0)

        return x, beta, gamma, rho, lambda_entry, lambda_exit, mu_entry, mu_exit

    def _add_pwl_constr(self, nodes: Iterable[StationNode],
                        get_node_bps: Callable[[StationNode], Union[List[PhiBreakpoint], List[WDFBreakpoint]]],
                        name: str) -> Dict[StationNode, Dict[Union[WDFBreakpoint, PhiBreakpoint], Var]]:
        # Create variable
        vars = self.model.continuous_var_dict(((node, bp) for node in nodes for bp in get_node_bps(node)), name=name)
        vars_by_node: Dict[StationNode, Dict[Union[PhiBreakpoint, WDFBreakpoint], Var]] = defaultdict(dict)
        for (node, bp), var in vars.items():
            vars_by_node[node][bp] = var
        # Create convexity constraint
        self.model.add_constraints(self.model.sum(node_vars.values()) == 1 for node_vars in vars_by_node.values())
        # Add SOS2
        for node, node_vars in vars_by_node.items():
            self.model.add_sos2(list(node_vars.values()))
        return vars_by_node


    def _link_pwl_constr(self, entry_vars: Dict[StationNode, Dict[Union[WDFBreakpoint, PhiBreakpoint], Var]],
                         exit_vars: Dict[StationNode, Dict[Union[WDFBreakpoint, PhiBreakpoint], Var]],
                         beta: Dict[Node, Var],
                         gamma: Dict[StationNode, Var], name):
        self.model.add_constraints(
            (self.model.sum(var * b.soc for b, var in node_entry_vars.items()) == beta[node] for node, node_entry_vars in
             entry_vars.items()),
            names=(f'bind_{name}_entry_to_beta_{node}' for node in entry_vars))

        self.model.add_constraints((
            self.model.sum(var * b.soc for b, var in exit_vars[node].items())
            - self.model.sum(var * b.soc for b, var in entry_vars[node].items())
            == gamma[node] for node in entry_vars),
            names=(f'bind_{name}_entry_to_beta_{node}' for node in entry_vars))
