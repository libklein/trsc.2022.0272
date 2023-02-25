# coding=utf-8
from typing import Callable, Optional, Sequence, Union, Tuple, Iterable, TypeVar
from pathlib import Path
from argparse import ArgumentTypeError
from os import access, W_OK
from re import search as matches_regex
from contextlib import contextmanager
from itertools import islice, zip_longest


def skip(iterable: Iterable, skip: int) -> Iterable:
    if skip < 0:
        raise ValueError
    return islice(iterable, skip, None)

def parsePath(arg, mode='r', type='d', regex=None):
    p = Path(arg)
    if not p.exists():
        if type == 'd':
            if mode == 'r':
                raise ArgumentTypeError("Path {} does not exist".format(p))
            else:
                # Try creating the directory
                try:
                    p.mkdir(parents=True)
                except:
                    raise ArgumentTypeError("Cannot write to {} (Does not exist)".format(p))
        elif type == 'f' and mode == 'r':
            raise ArgumentTypeError("Path {} does not exist".format(p))

    if type == 'd':
        if not p.is_dir():
            raise ArgumentTypeError("Path {} is not a directory".format(p))
        if mode == 'w' and not access(p, W_OK):
            raise ArgumentTypeError('Path {} is not writeable'.format(p))
    if regex and not matches_regex(regex, str(p)):
        raise ArgumentTypeError('Path {} does not match re {}'.format(p, regex))
    return p


def PathParser(mode='r', type='d', regex=None):
    return lambda x: parsePath(x, mode=mode, type=type, regex=regex)


@contextmanager
def FunctionContext(entry_function: Callable, exit_function: Callable, *args, **kwargs):
    entry_function(*args, **kwargs)
    yield
    exit_function(*args, **kwargs)

@contextmanager
def SilenceGurobiContext(model, pred: Optional[Callable] = None):
    silence = True if pred is None else pred()
    _prev_setting = model.Params.LogToConsole
    if silence:
        model.Params.LogToConsole = 0
    yield
    if silence:
        model.Params.LogToConsole = _prev_setting = model.Params.LogToConsole

"""
Takes a sorted sequence of items "seq" and returns a new list containing the unique items of "seq". 

>>> remove_duplicates_from_sorted([1,1,2,4,4,5])
[1, 2, 4, 5]
>>> remove_duplicates_from_sorted([[1],[1],[2, 3],[2],[2],[4],[5],[5]], key = lambda x: x[0])
[[1], [2], [4], [5]]
>>> remove_duplicates_from_sorted([1])
[1]
"""
def remove_duplicates_from_sorted(seq: Sequence, key=lambda x: x) -> Sequence:
    return [x_prev for x, x_prev in zip_longest(islice(seq, 1, None), seq, fillvalue=None)
            if key(x) != key(x_prev)]


T = TypeVar('T')
def create_list_iterator(first: T, end: T, get_next: Callable[[T],Optional[T]] = lambda x: x.successor, inclusive=False) -> Iterable[T]:
    p = first
    while p != (end if not inclusive else get_next(end)):
        yield p
        p = get_next(end)

