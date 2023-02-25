# coding=utf-8
try:
    from gurobipy import GRB
    import gurobipy as grb
except:
    pass
from typing import Dict
import json
import logging
from pathlib import Path
from zlib import compress
import random
import ijson
from re import compile

DEPTH_MSG_REGEX = compile(r'\s*(?P<explored_nodes>\d+)\s*(?P<unexplored_nodes>\d+)\s*(?P<current_obj>\d+\.?\d*)\s*'
                          r'(?P<depth>\d+)\s*(?P<violated_integrals>\d+)\s*(?P<best_integral>(?:\d+\.?\d*)|-)\s*'
                          r'(?P<best_bound>\d+\.?\d*)\s*(?P<gap>(?:\d+\.?\d*)|-)%?\s*'
                          r'(?P<iters_per_node>(?:\d+\.?\d*)|-)\s*(?P<time>\d+)s')


class JSONFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)

    def format(self, record):
        return json.dumps(record.msg)


def create_node_logger(filename: Path):
    logger = logging.getLogger(str(filename))
    sink = logging.FileHandler(filename, mode='w')
    sink.setFormatter(JSONFormatter())
    sink.setLevel(logging.INFO)
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    return lambda model, where, variables: log_node(logger, where, model=model, variables=variables)


def log_node(logger, where, model, variables):
    if where == GRB.Callback.MESSAGE:
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        # Parse msg and set depth
        match = DEPTH_MSG_REGEX.search(msg)
        if (match):
            model._current_est_depth = int(match.group('depth'))
        return
    elif where != GRB.Callback.MIPNODE or model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return

    nodecnt = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
    if nodecnt % 100 != 1:
        return
    # Decide which variables are non integral
    INT_FEAS_THRESHOLD = model.Params.IntFeasTol
    integral_vals = {}
    non_integral_vals = {}
    for key, variable in variables.items():
        value = model.cbGetNodeRel(variable)
        if value < INT_FEAS_THRESHOLD or value >= 1.0 - INT_FEAS_THRESHOLD:
            integral_vals[variable.VarName] = value
        else:
            non_integral_vals[variable.VarName] = value

    # Push to disk
    serialized_node = {
        "objbst": model.cbGet(GRB.Callback.MIPNODE_OBJBST),
        "nodecnt": model.cbGet(GRB.Callback.MIPNODE_NODCNT),
        "depth": model._current_est_depth,
        "numIntegral": len(integral_vals),
        "numCont": len(non_integral_vals),
        "integralVariables": integral_vals,
        "contVariables": non_integral_vals
    }

    logger.info(serialized_node)

def node_log_stream(node_log_file: Path):
    # Node log is a list of JSON objects, not formatted as a JSON array
    with open(node_log_file, 'r') as node_fs:
        node_stream = ijson.items(node_fs, '', multiple_values=True)
        for node in node_stream:
            yield node
