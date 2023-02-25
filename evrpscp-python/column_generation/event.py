# coding=utf-8
from events import Events


GlobalSolverEvents = SolverEvents

def reset_global_eh():
    for x in GlobalSolverEvents.__events__:
        getattr(GlobalSolverEvents, x).targets.clear()