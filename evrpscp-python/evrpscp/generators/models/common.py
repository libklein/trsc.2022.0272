# coding=utf-8
import random
from inspect import signature
from typing import TypeVar, Tuple, Optional, Union, Any, Generic
from json import JSONEncoder
from funcy import is_tuple

import funcy

T = TypeVar('T')
Range = Tuple[T, T]


class Parameter(Generic[T]):
    _lower_bound: T
    _upper_bound: T
    _value = None

    def __init__(self, lb: Union[T, Tuple[T, T]], ub: Optional[T] = None):
        if is_tuple(lb):
            self._lower_bound, self._upper_bound = lb
        else:
            self._lower_bound = lb
            self._upper_bound = ub if ub is not None else self._lower_bound
        assert self._lower_bound <= self._upper_bound

    def __eq__(self, other: 'Parameter'):
        return (self.min, self.max) == (other.min, other.max)

    def __str__(self):
        if self.min != self.max:
            return f'{self.min}-{self.max}'
        return str(self.min)

    @property
    def max(self) -> T:
        return self._upper_bound

    @property
    def min(self) -> T:
        return self._lower_bound

    def value(self, rand_gen: Optional[random.Random] = None) -> T:
        if self._value is None:
            self._value = self.generate(rand_gen=rand_gen)
        return self._value

    def generate(self, rand_gen: Optional[random.Random] = None) -> T:
        if isinstance(self.min, int):
            val = rand_gen.randint(self.min, self.max) if rand_gen is not None else random.randint(self.min, self.max)
        else:
            val = rand_gen.uniform(self.min, self.max) if rand_gen is not None else random.uniform(self.min, self.max)
        self._value = val
        return val

    class JSONEncoder(JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, Parameter):
                return (o.min, o.max)
            return JSONEncoder.default(self, o)

def parameterized(func):
    def decorated_func(*args, **kwargs):
        sig = signature(func)
        params = sig.bind(*args, **kwargs)
        for param_name, param_value in params.arguments.items():
            if sig.parameters[param_name].annotation is not Parameter:
                continue
            if isinstance(param_value, Parameter):
                continue
            # Cast to parameter
            params.arguments[param_name] = Parameter(param_value)
        return func(*params.args, **params.kwargs)
    return decorated_func
