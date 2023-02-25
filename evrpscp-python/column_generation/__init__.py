# coding=utf-8
from contextlib import suppress

from .parameters import EPS, BIG_M_COEF, INTEGRALITY_TOLERANCE, INTEGRALITY_PRECISION
from .discretization import DiscretizedInstance
from .column import Column
from .constraints import Constraint
from .node import Node
from .master_problem import MasterProblem
from .subproblem import SubProblemInfeasible

with suppress(ModuleNotFoundError, ImportError):
    from .subproblem import MIPSubproblem

with suppress(ModuleNotFoundError, ImportError):
    from .subproblem import CPPSubproblem

from .cli import run_from_command_line
from .solver import Solver