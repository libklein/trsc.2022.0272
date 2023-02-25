#!/usr/bin/python3
from typing import *
from json import load as load_json
from pathlib import Path
from xml.etree.ElementTree import tostring as xml_tostring
from xml.dom.minidom import parseString as parseXMLString
from copy import copy
import toml
import json

from evrpscp import SchedulingInstance, FleetChargingSchedule
from evrpscp.models import Period, Charger, Battery, FleetSchedule, PeriodRoutingSolution
from evrpscp.pool import RankedPeriodSolution, PeriodSolutionPool, SolutionPool


def __parseFromJSON(path: Path, cls):
    with open(path) as f:
        return cls.from_json(f.read())

def __parseFromDict(data, cls):
    return cls.from_dict(data)


def parseInstance(path: Path) -> Union[SchedulingInstance]:
    with open(path) as inst_file:
        inst_dict = json.load(inst_file)
    try:
        return __parseFromDict(inst_dict, SchedulingInstance)
    except Exception as e:
        from traceback import print_tb
        print_tb(e.__traceback__)
        raise e


def parseSolution(path: Path) -> FleetSchedule:
    return __parseFromJSON(path, FleetSchedule)

def parseSchedule(path: Path) -> FleetChargingSchedule:
    return __parseFromJSON(path, FleetChargingSchedule)


def parseRates(path: Path, days: int = 1, day_len: float = 60*24) -> List[Period]:
    _rates = []
    with open(path, "r") as f:
        data = Period.schema().loads(f.read(), many=True)
        for d in range(days):
            offset = d * day_len
            for p in data:
                _rates.append(copy(p).offset(offset))

    # Join adjacent
    rates = [_rates[0]]
    for i in range(1, len(_rates)):
        if rates[-1].energyPrice == _rates[i].energyPrice:
            rates[-1].end = _rates[i].end
        else:
            rates.append(_rates[i])
    return rates


def parseCharger(path: Path) -> Charger:
    return __parseFromJSON(path, Charger)


def parseBattery(path: Path) -> Battery:
    return __parseFromJSON(path, Battery)

def parseSolutionPool(files: List[Path]) -> SolutionPool:
    solution_pools = []
    for period, pool_file in enumerate(files):
        print(f'Parsing solution pool {period + 1}/{len(files)}...', end='\r', flush=True)
        pool = __parseFromJSON(pool_file, PeriodSolutionPool)
        pool.period = period
        solution_pools.append(pool)
    print(
        f"Loaded {len(solution_pools)} solution pools with a total of {sum(x.size for x in solution_pools)} solutions.")
    return SolutionPool(solution_pools)


def WriteXML(obj, fobj, mode="text"):
    if not hasattr(fobj,"write") or not callable(fobj.write):
        raise ValueError("Out object has no write method!")

    pretty_print = lambda xml_root: parseXMLString(xml_tostring(xml_root)).toprettyxml()

    xml_root = obj.toXML()

    if mode == "text":
        fobj.write(pretty_print(xml_root))
    elif mode == "binary":
        fobj.write(pretty_print(xml_root).encode())
    fobj.close()


