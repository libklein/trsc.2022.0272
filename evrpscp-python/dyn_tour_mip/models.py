# coding=utf-8
from dataclasses import dataclass

Vehicle = int

@dataclass
class PWLBreakpoint:
    time: float
    soc: float

    def __str__(self):
        return f'({self.time:.2f}, {self.soc:.2f})'

    def __hash__(self):
        return hash((self.time, self.soc))
