from . import vector
from . import tensor 
from . import value
from typing import TypeVar, Any, overload
import copapy as cp

U = TypeVar("U", int, float)


@overload
def relu(x: U) -> U: ...
@overload
def relu(x: value[U]) -> value[U]: ...
@overload
def relu(x: vector[U]) -> vector[U]: ...
@overload
def relu(x: tensor[U]) -> tensor[U]: ...
def relu(x: U | value[U] | vector[U] | tensor[U]) -> Any:
    """Returns x for x > 0 and otherwise 0."""
    ret = x * (x > 0)
    return ret


@overload
def sigmoid(x: U) -> float: ...
@overload
def sigmoid(x: value[U]) -> value[float]: ...
@overload
def sigmoid(x: vector[U]) -> vector[float]: ...
@overload
def sigmoid(x: tensor[U]) -> tensor[float]: ...
def sigmoid(x: U | value[U] | vector[U] | tensor[U]) -> Any:
    """Sigmoid function to map any value to the range (0, 1)."""
    return 1 / (1 + cp.exp(-x))
