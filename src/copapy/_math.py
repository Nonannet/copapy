from . import variable, NumLike
from typing import TypeVar, Any, overload
from ._basic_types import add_op
import math

T = TypeVar("T", int, float, variable[int], variable[float])


@overload
def sqrt(x: float | int) -> float: ...
@overload
def sqrt(x: variable[Any]) -> variable[float]: ...
def sqrt(x: NumLike) -> variable[float] | float:
    """Square root function
    
    Arguments:
        x: Input value
        
    Returns:
        Square root of x
    """
    if isinstance(x, variable):
        return add_op('sqrt', [x, x])  # TODO: fix 2. dummy argument
    return float(x ** 0.5)


@overload
def sin(x: float | int) -> float: ...
@overload
def sin(x: variable[Any]) -> variable[float]: ...
def sin(x: NumLike) -> variable[float] | float:
    """Sine function
    
    Arguments:
        x: Input value
        
    Returns:
        Square root of x
    """
    if isinstance(x, variable):
        return add_op('sin', [x, x])  # TODO: fix 2. dummy argument
    return math.sin(x)

@overload
def cos(x: float | int) -> float: ...
@overload
def cos(x: variable[Any]) -> variable[float]: ...
def cos(x: NumLike) -> variable[float] | float:
    """Cosine function
    
    Arguments:
        x: Input value
        
    Returns:
        Cosine of x
    """
    if isinstance(x, variable):
        return add_op('cos', [x, x])  # TODO: fix 2. dummy argument
    return math.cos(x)

@overload
def tan(x: float | int) -> float: ...
@overload
def tan(x: variable[Any]) -> variable[float]: ...
def tan(x: NumLike) -> variable[float] | float:
    """Tangent function
    
    Arguments:
        x: Input value
        
    Returns:
        Tangent of x
    """
    if isinstance(x, variable):
        return add_op('tan', [x, x])  # TODO: fix 2. dummy argument
    return math.tan(x)


def get_42() -> variable[float]:
    """Returns the variable representing the constant 42"""
    return add_op('get_42', [0.0, 0.0])


def abs(x: T) -> T:
    """Absolute value function
    
    Arguments:
        x: Input value
        
    Returns:
        Absolute value of x
    """
    ret = (x < 0) * -x + (x >= 0) * x
    return ret  # pyright: ignore[reportReturnType]

