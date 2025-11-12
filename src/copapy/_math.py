from . import variable, NumLike
from typing import TypeVar, Any, overload
from ._basic_types import add_op
import math

T = TypeVar("T", int, float, variable[int], variable[float])


@overload
def exp(x: float | int) -> float: ...
@overload
def exp(x: variable[Any]) -> variable[float]: ...
def exp(x: NumLike) -> variable[float] | float:
    """Exponential function to basis e

    Arguments:
        x: Input value

    Returns:
        result of e**x
    """
    if isinstance(x, variable):
        return add_op('exp', [x, x])  # TODO: fix 2. dummy argument
    return float(math.exp(x))


@overload
def log(x: float | int) -> float: ...
@overload
def log(x: variable[Any]) -> variable[float]: ...
def log(x: NumLike) -> variable[float] | float:
    """Logarithm to basis e

    Arguments:
        x: Input value

    Returns:
        result of ln(x)
    """
    if isinstance(x, variable):
        return add_op('log', [x, x])  # TODO: fix 2. dummy argument
    return float(math.log(x))


@overload
def pow(x: float | int, y: float | int) -> float: ...
@overload
def pow(x: variable[Any], y: NumLike) -> variable[float]: ...
@overload
def pow(x: NumLike, y: variable[Any]) -> variable[float]: ...
def pow(x: NumLike, y: NumLike) -> NumLike:
    """x to the power of y

    Arguments:
        x: Input value

    Returns:
        result of x**y
    """
    if isinstance(y, int) and 0 <= y < 8:
        if y == 0:
            return 1
        m = x
        for _ in range(y - 1):
            m *= x
        return m
    if y == -1:
        return 1 / x
    if isinstance(x, variable) or isinstance(y, variable):
        return add_op('pow', [x, y])
    else:
        return float(x ** y)


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
    return float(math.sqrt(x))


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


@overload
def atan(x: float | int) -> float: ...
@overload
def atan(x: variable[Any]) -> variable[float]: ...
def atan(x: NumLike) -> variable[float] | float:
    """Inverse tangent function

    Arguments:
        x: Input value

    Returns:
        Inverse tangent of x
    """
    if isinstance(x, variable):
        return add_op('atan', [x, x])  # TODO: fix 2. dummy argument
    return math.atan(x)


@overload
def atan2(x: float | int, y: float | int) -> float: ...
@overload
def atan2(x: variable[Any], y: variable[Any]) -> variable[float]: ...
def atan2(x: NumLike, y: NumLike) -> variable[float] | float:
    """2-argument arctangent

    Arguments:
        x: Input value
        y: Input value

    Returns:
        Result in radian
    """
    if isinstance(x, variable) or isinstance(y, variable):
        return add_op('atan2', [x, y])  # TODO: fix 2. dummy argument
    return math.atan2(x, y)


@overload
def asin(x: float | int) -> float: ...
@overload
def asin(x: variable[Any]) -> variable[float]: ...
def asin(x: NumLike) -> variable[float] | float:
    """Inverse sine function

    Arguments:
        x: Input value

    Returns:
        Inverse sine of x
    """
    if isinstance(x, variable):
        return add_op('asin', [x, x])  # TODO: fix 2. dummy argument
    return math.asin(x)


@overload
def acos(x: float | int) -> float: ...
@overload
def acos(x: variable[Any]) -> variable[float]: ...
def acos(x: NumLike) -> variable[float] | float:
    """Inverse cosine function

    Arguments:
        x: Input value

    Returns:
        Inverse cosine of x
    """
    return math.pi / 2 - asin(x)


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
