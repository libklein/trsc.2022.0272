# coding=utf-8
from contextlib import suppress

class SubProblemInfeasible(Exception):
    pass

with suppress(ModuleNotFoundError):
    from .mip import SubProblem as MIPSubproblem

from .frvcp_cpp import CPPSubproblem