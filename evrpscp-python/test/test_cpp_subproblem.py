# coding=utf-8
from copy import deepcopy
from itertools import product

import pytest
from column_generation.subproblem.frvcp_cpp.network import *
from column_generation.subproblem.frvcp_cpp import CPPSubproblem
import docplex.mp.basic
import docplex.mp.model
import docplex.mp.utils
from funcy import ilen


@pytest.fixture
def cpp_subproblem(instance):
    return CPPSubproblem(instance=instance, vehicle=0)

def test_deepcopy(cpp_subproblem: CPPSubproblem):
    original = cpp_subproblem
    clone: CPPSubproblem = deepcopy(original)
    assert original is not clone

    # Factories should be updated
    assert original.network.arc_factory is not clone.network.arc_factory
    assert original.network.node_factory is not clone.network.node_factory

    # Instance data should not change
    for clone_instance_data, original_instance_data in zip(
            (clone._instance, clone.vehicle, clone.tours,
            clone.params, clone.battery, clone.chargers, clone.periods),
            (original._instance, original.vehicle, original.tours,
            original.params, original.battery, original.chargers, original.periods)):
        assert clone_instance_data is original_instance_data

    # Bindings should refer to the same objects
    assert clone._cpp_wdf is original._cpp_wdf
    for clone_binding, original_binding in zip(
            (clone._cpp_charger_mapping, clone._cpp_tour_mapping),
            (original._cpp_charger_mapping, original._cpp_tour_mapping)):
        for clone_cpp_obj, original_cpp_obj in zip(clone_binding.values(), original_binding.values()):
            assert clone_cpp_obj is original_cpp_obj

        for clone_cpp_obj, original_cpp_obj in zip(clone_binding.keys(), original_binding.keys()):
            assert clone_cpp_obj is original_cpp_obj

    # cpp network should have been cloned
    cloned_cpp_network, original_cpp_network = clone._cpp_network, original._cpp_network
    assert cloned_cpp_network is not original_cpp_network

    # Check number of arcs and ops
    assert cloned_cpp_network.number_of_vertices == original_cpp_network.number_of_vertices
    assert cloned_cpp_network.number_of_arcs == original_cpp_network.number_of_arcs
    assert cloned_cpp_network.number_of_operations == original_cpp_network.number_of_operations

    # Check vertices and neighborhood structure
    for original_vid in original_cpp_network.vertices:
        has_copy = False
        for clone_vid in cloned_cpp_network.vertices:
            cloned_vertex, original_vertex = cloned_cpp_network.get_vertex(clone_vid), original_cpp_network.get_vertex(original_vid)
            assert cloned_vertex is not original_vertex
            if cloned_vertex == original_vertex:
                # Check outgoing arcs
                num_equal_arcs = 0
                for oa in map(lambda x: original_cpp_network.get_arc(x), original_cpp_network.get_outgoing_arcs(original_vid)):
                    has_arc_copy = False
                    for ca in map(lambda x: cloned_cpp_network.get_arc(x), cloned_cpp_network.get_outgoing_arcs(clone_vid)):
                        if oa == ca:
                            has_arc_copy = True
                            break
                    num_equal_arcs += has_arc_copy
                has_copy = num_equal_arcs == ilen(original_cpp_network.get_outgoing_arcs(original_vid))
                if has_copy:
                    break
        assert has_copy, f'Vertex {original_vid} has no copy in clone!'

    # Check node/arc mappings
    assert clone._cpp_vertex_mapping is not original._cpp_vertex_mapping
    assert clone._cpp_arc_mapping is not original._cpp_arc_mapping

    for vertex, cpp_vid in clone._cpp_vertex_mapping.items():
        assert vertex.vertex_id == cpp_vid
    for arc, cpp_arc_id in clone._cpp_arc_mapping.items():
        assert arc.arc_id == cpp_arc_id

    # TODO Network should have been recreated

    # TODO Vertex / Arc bindings should be valid
