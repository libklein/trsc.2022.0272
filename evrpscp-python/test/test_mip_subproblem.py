# coding=utf-8
from copy import deepcopy
from itertools import product

import pytest
from column_generation.subproblem.mip.network import *
from column_generation.subproblem.mip import SubProblem as MIPSubproblem
import docplex.mp.basic
import docplex.mp.model
import docplex.mp.utils

@pytest.fixture
def mip_subproblem(instance):
    return MIPSubproblem(instance=instance, vehicle=0)

def assert_members_belong_to_model(obj, model: Model):
    for attr_name in dir(obj):
        try:
            attr = getattr(obj, attr_name)
        except docplex.mp.utils.DOcplexException:
            continue
        if isinstance(attr, docplex.mp.basic.IndexableObject):
            assert attr.model is model
        elif isinstance(attr, docplex.mp.model.Model):
            assert attr is model

def test_deepcopy(mip_subproblem: MIPSubproblem):
    subproblem_clone = deepcopy(mip_subproblem)
    original_model = mip_subproblem.model
    cloned_model = subproblem_clone.model
    assert original_model is not cloned_model
    # Test the network
    for node in subproblem_clone.network.nodes:
        assert node.model is cloned_model
        assert node.beta.model is cloned_model
        if node.node_type == NodeType.Station:
            assert node.gamma.model is cloned_model
            assert node.rho.model is cloned_model
        assert_members_belong_to_model(node, model=cloned_model)
    for arc in subproblem_clone.network.arcs:
        assert arc.model is cloned_model
        assert arc.x.model is cloned_model
        assert_members_belong_to_model(arc, model=cloned_model)

    for model in (original_model, cloned_model):
        # Force deterministic parallel mode, otherwise results will likely differ even for the same model
        model.parameters.parallel = 1

    original_sol = original_model.solve()
    cloned_sol = cloned_model.solve()
    assert original_sol.objective_value == cloned_sol.objective_value

    original_sol.clear()
    cloned_sol.clear()

    capacity_duals = {
        (p,f): 2.0 for p,f in product(mip_subproblem.periods, mip_subproblem.chargers)
    }
    assert mip_subproblem.generate_column(coverage_dual=10.0, capacity_dual=capacity_duals) == subproblem_clone.generate_column(coverage_dual=10.0, capacity_dual=capacity_duals)