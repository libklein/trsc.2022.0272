# coding=utf-8
from copy import deepcopy
from itertools import chain
from typing import Dict, Tuple

import pytest

from column_generation.subproblem.network import *
from column_generation.column import Column
from evrpscp.models import DiscretePeriod, Charger, DiscreteTour, DiscretizedInstance
from funcy import nth


@pytest.fixture
def network(instance):
    return TimeExpandedNetwork(instance, vehicle=0)

@pytest.fixture
def column(instance):
    energy_charged: Dict[DiscretePeriod, float] = {}
    degradation_cost: Dict[DiscretePeriod, float] = {}
    charger_usage: Dict[Tuple[DiscretePeriod, Charger], bool] = {}
    tour_departures: Dict[DiscreteTour, DiscretePeriod] = {}
    vehicle = 0
    tours = sorted(instance.tours[vehicle], key=lambda pi: pi.earliest_departure)

    def get_fastest_charger(entry_soc: float, dur: float):
        return max(((f, f.charge_for(entry_soc, dur)) for f in instance.chargers), key=lambda tpl: tpl[1])

    battery = instance.parameters.battery
    cur_soc = battery.initialCharge
    # Iterate over periods, charging as much as possible to traverse the next tour
    next_tour = instance.tours[vehicle][0]
    return_period = None
    charge_required = next_tour.consumption
    total_cost = 0
    for p in instance.periods:
        delta_soc = deg_cost = 0
        f = None
        if cur_soc < charge_required:
            f, max_charge = get_fastest_charger(entry_soc=cur_soc, dur=p.duration)
            delta_soc = min(charge_required - cur_soc, max_charge)
            deg_cost = battery.wearCost(cur_soc, cur_soc + delta_soc)
        cur_soc += delta_soc
        energy_charged[p] = delta_soc
        degradation_cost[p] = deg_cost
        for f_prime in instance.chargers:
            charger_usage[p, f_prime] = (f is f_prime)

        if return_period is not None:
            if return_period is p:
                return_period = None
                next_tour = tours[tours.index(next_tour)+1] if next_tour is not tours[-1] else None
                charge_required = next_tour.consumption if next_tour is not None else 0.0
            continue
        if next_tour is not None and p >= next_tour.earliest_departure and cur_soc >= charge_required:
            assert p <= next_tour.latest_departure
            tour_departures[next_tour] = p
            return_period = nth(next_tour.duration, iter(p))

    assert all(pi in tour_departures for pi in tours)

    total_cost = sum(degradation_cost[p] + p.energyPrice * energy_charged[p] for p in instance.periods)

    return Column(energy_charged=energy_charged, degradation_cost=degradation_cost, charger_usage=charger_usage,
                  tour_departures=tour_departures, vehicle=vehicle, objective=total_cost)

class MockNode(Node):
    @staticmethod
    def create_node(type, *args, **kwargs):
        return MockNode(*args, node_type=type, **kwargs)

    def __post_init__(self, *args, **kwargs):
        self._charge = 0
        self._entry_soc = 0
        self._deg_cost = 0
        super(MockNode, self).__post_init__(*args, **kwargs)

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

class MockArc(Arc):
    def __post_init__(self, *args, **kwargs):
        self._is_active = False
        super(MockArc, self).__post_init__(*args, **kwargs)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool):
        print(f'Setting arc {self} active!')
        self._is_active = value

    @staticmethod
    def create_arc(type, *args, tour=None, **kwargs):
        return MockArc(*args, arc_type=type, **kwargs)

def test_set_solution_with_reset(instance: DiscretizedInstance, column: Column):
    network = TimeExpandedNetwork(instance=instance, vehicle=column.vehicle, arc_factory=MockArc.create_arc,
                                  node_factory=MockNode.create_node)
    print(column)
    network.set_solution(column=column, reset=True)
    assert network.create_column(chargers=column.chargers, vehicle=column.vehicle, objective=column.objective)\
           == column
    # TODO Get second different column, and apply solution. Should match

def test_set_solution_without_reset(instance: DiscretizedInstance, column: Column):
    network = TimeExpandedNetwork(instance=instance, vehicle=column.vehicle, arc_factory=MockArc.create_arc,
                                  node_factory=MockNode.create_node)
    print(column)
    network.set_solution(column=column, reset=False)
    assert network.create_column(chargers=column.chargers, vehicle=column.vehicle, objective=column.objective) \
           == column
    # TODO Get second column, and apply solution. Should not match


def test_deepcopy(network: TimeExpandedNetwork):
    network.display()
    copied_network = deepcopy(network)
    # Check that the network is indeed a copy
    def eq_but_not_same(lhs, rhs):
        return lhs == rhs and lhs is not rhs
    assert eq_but_not_same(copied_network.source, network.source)
    assert eq_but_not_same(copied_network.sink, network.sink)
    assert eq_but_not_same(copied_network.station_nodes, network.station_nodes)
    assert eq_but_not_same(copied_network.idle_nodes, network.idle_nodes)
    assert eq_but_not_same(copied_network.nodes_by_period, network.nodes_by_period)
    # Ensure that nodes have been deepcopied
    original_ids = {id(x) for x in chain(network.arcs, network.nodes)}
    for copy_node, original_node in zip(copied_network.nodes, network.nodes):
        assert id(copy_node) != id(original_node)
        assert all(id(x) not in original_ids for x in copy_node.outgoing_arcs)
        assert all(copy_node is arc.origin for arc in copy_node.outgoing_arcs)
        assert all(id(x) not in original_ids for x in copy_node.incoming_arcs)
        assert all(copy_node is arc.target for arc in copy_node.incoming_arcs)
    for copy_arc, original_arc in zip(copied_network.arcs, network.arcs):
        assert id(copy_arc) != id(original_arc)
        assert id(copy_arc.origin) != id(original_arc.origin)
        assert id(copy_arc.target) != id(original_arc.target)
        assert copy_arc in copy_arc.origin.outgoing_arcs
