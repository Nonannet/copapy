from . import variable, NumLike
from typing import TypeVar, Any, overload
from ._basic_types import add_op

T = TypeVar("T", int, float, variable[int], variable[float])


@overload
def sqrt(x: float | int) -> float: ...
@overload
def sqrt(x: variable[Any]) -> variable[float]: ...
def sqrt(x: NumLike) -> variable[float] | float:
    """Square root function"""
    if isinstance(x, variable):
        return add_op('sqrt', [x, x])  # TODO: fix 2. dummy argument
    return float(x ** 0.5)


@overload
def sqrt2(x: float | int) -> float: ...
@overload
def sqrt2(x: variable[Any]) -> variable[float]: ...
def sqrt2(x: NumLike) -> variable[float] | float:
    """Square root function"""
    if isinstance(x, variable):
        return add_op('sqrt2', [x, x])  # TODO: fix 2. dummy argument
    return float(x ** 0.5)


def get_42() -> variable[float]:
    """Returns the variable representing the constant 42"""
    return add_op('get_42', [0.0, 0.0])


def abs(x: T) -> T:
    """Absolute value function"""
    ret = (x < 0) * -x + (x >= 0) * x
    return ret  # pyright: ignore[reportReturnType]

