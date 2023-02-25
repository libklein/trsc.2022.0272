# coding=utf-8
from copy import deepcopy

from evrpscp.parsing.pelletier import parseBattery, parseCharger, parseRates
from pathlib import Path

__SlowCharger = parseCharger(Path(__file__).parent / 'SlowCharger.json')
__FastCharger = parseCharger(Path(__file__).parent / 'FastCharger.json')
__SummerRates = parseRates(Path(__file__).parent / 'SummerRates-TOU-EV-4.json')
__WinterRates = parseRates(Path(__file__).parent / 'WinterRates-TOU-EV-4.json')
__Battery = parseBattery(Path(__file__).parent / 'Battery.json')


def __getattr__(name: str):
    if name == 'SlowCharger':
        return deepcopy(__SlowCharger)
    elif name == 'FastCharger':
        return deepcopy(__FastCharger)
    elif name == 'SummerRates':
        return deepcopy(__SummerRates)
    elif name == 'WinterRates':
        return deepcopy(__WinterRates)
    elif name == 'Battery':
        return deepcopy(__Battery)

    raise AttributeError(f'Module {__name__} has no attribute \"{name}\"')
