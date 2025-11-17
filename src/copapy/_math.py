from . import vector
from ._vectors import VecNumLike
from . import variable, NumLike
from typing import TypeVar, Any, overload, Callable
from ._basic_types import add_op
import math

T = TypeVar("T", int, float, variable[int], variable[float])
U = TypeVar("U", int, float)

@overload
def exp(x: float | int) -> float: ...
@overload
def exp(x: variable[Any]) -> variable[float]: ...
@overload
def exp(x: vector[Any]) -> vector[float]: ...
def exp(x: Any) -> Any:
    """Exponential function to basis e

    Arguments:
        x: Input value

    Returns:
        result of e**x
    """
    if isinstance(x, variable):
        return add_op('exp', [x])
    if isinstance(x, vector):
        return x.map(exp)
    return float(math.exp(x))


@overload
def log(x: float | int) -> float: ...
@overload
def log(x: variable[Any]) -> variable[float]: ...
@overload
def log(x: vector[Any]) -> vector[float]: ...
def log(x: Any) -> Any:
    """Logarithm to basis e

    Arguments:
        x: Input value

    Returns:
        result of ln(x)
    """
    if isinstance(x, variable):
        return add_op('log', [x])
    if isinstance(x, vector):
        return x.map(log)
    return float(math.log(x))


@overload
def pow(x: float | int, y: float | int) -> float: ...
@overload
def pow(x: variable[Any], y: NumLike) -> variable[float]: ...
@overload
def pow(x: NumLike, y: variable[Any]) -> variable[float]: ...
@overload
def pow(x: vector[Any], y: Any) -> vector[float]: ...
def pow(x: VecNumLike, y: VecNumLike) -> Any:
    """x to the power of y

    Arguments:
        x: Input value

    Returns:
        result of x**y
    """
    if isinstance(x, vector) or isinstance(y, vector):
        return map2(x, y, pow)
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
@overload
def sqrt(x: vector[Any]) -> vector[float]: ...
def sqrt(x: Any) -> Any:
    """Square root function

    Arguments:
        x: Input value

    Returns:
        Square root of x
    """
    if isinstance(x, variable):
        return add_op('sqrt', [x])
    if isinstance(x, vector):
        return x.map(sqrt)
    return float(math.sqrt(x))


@overload
def sin(x: float | int) -> float: ...
@overload
def sin(x: variable[Any]) -> variable[float]: ...
@overload
def sin(x: vector[Any]) -> vector[float]: ...
def sin(x: Any) -> Any:
    """Sine function

    Arguments:
        x: Input value

    Returns:
        Square root of x
    """
    if isinstance(x, variable):
        return add_op('sin', [x])
    if isinstance(x, vector):
        return x.map(sin)
    return math.sin(x)


@overload
def cos(x: float | int) -> float: ...
@overload
def cos(x: variable[Any]) -> variable[float]: ...
@overload
def cos(x: vector[Any]) -> vector[float]: ...
def cos(x: Any) -> Any:
    """Cosine function

    Arguments:
        x: Input value

    Returns:
        Cosine of x
    """
    if isinstance(x, variable):
        return add_op('cos', [x])
    if isinstance(x, vector):
        return x.map(cos)
    return math.cos(x)


@overload
def tan(x: float | int) -> float: ...
@overload
def tan(x: variable[Any]) -> variable[float]: ...
@overload
def tan(x: vector[Any]) -> vector[float]: ...
def tan(x: Any) -> Any:
    """Tangent function

    Arguments:
        x: Input value

    Returns:
        Tangent of x
    """
    if isinstance(x, variable):
        return add_op('tan', [x])
    if isinstance(x, vector):
        #return x.map(tan)
        return x.map(tan)
    return math.tan(x)


@overload
def atan(x: float | int) -> float: ...
@overload
def atan(x: variable[Any]) -> variable[float]: ...
@overload
def atan(x: vector[Any]) -> vector[float]: ...
def atan(x: Any) -> Any:
    """Inverse tangent function

    Arguments:
        x: Input value

    Returns:
        Inverse tangent of x
    """
    if isinstance(x, variable):
        return add_op('atan', [x])
    if isinstance(x, vector):
        return x.map(atan)
    return math.atan(x)


@overload
def atan2(x: float | int, y: float | int) -> float: ...
@overload
def atan2(x: variable[Any], y: NumLike) -> variable[float]: ...
@overload
def atan2(x: NumLike, y: variable[Any]) -> variable[float]: ...
@overload
def atan2(x: vector[float], y: VecNumLike) -> vector[float]: ...
@overload
def atan2(x: VecNumLike, y: vector[float]) -> vector[float]: ...
def atan2(x: VecNumLike, y: VecNumLike) -> Any:
    """2-argument arctangent

    Arguments:
        x: Input value
        y: Input value

    Returns:
        Result in radian
    """
    if isinstance(x, vector) or isinstance(y, vector):
        return map2(x, y, atan2)
    if isinstance(x, variable) or isinstance(y, variable):
        return add_op('atan2', [x, y])
    return math.atan2(x, y)


@overload
def asin(x: float | int) -> float: ...
@overload
def asin(x: variable[Any]) -> variable[float]: ...
@overload
def asin(x: vector[Any]) -> vector[float]: ...
def asin(x: Any) -> Any:
    """Inverse sine function

    Arguments:
        x: Input value

    Returns:
        Inverse sine of x
    """
    if isinstance(x, variable):
        return add_op('asin', [x])
    if isinstance(x, vector):
        return x.map(asin)
    return math.asin(x)


@overload
def acos(x: float | int) -> float: ...
@overload
def acos(x: variable[Any]) -> variable[float]: ...
@overload
def acos(x: vector[Any]) -> vector[float]: ...
def acos(x: Any) -> Any:
    """Inverse cosine function

    Arguments:
        x: Input value

    Returns:
        Inverse cosine of x
    """
    if isinstance(x, variable):
        return add_op('acos', [x])
    if isinstance(x, vector):
        return x.map(acos)
    return math.asin(x)


@overload
def get_42(x: float | int) -> float: ...
@overload
def get_42(x: variable[Any]) -> variable[float]: ...
def get_42(x: NumLike) -> variable[float] | float:
    """Returns the variable representing the constant 42"""
    if isinstance(x, variable):
        return add_op('get_42', [x, x])
    return float((int(x) * 3.0 + 42.0) * 5.0 + 21.0)


def abs(x: T) -> T:
    """Absolute value function

    Arguments:
        x: Input value

    Returns:
        Absolute value of x
    """
    ret = (x < 0) * -x + (x >= 0) * x
    return ret  # pyright: ignore[reportReturnType]


def map2(self: VecNumLike, other: VecNumLike, func: Callable[[Any, Any], variable[U] | U]) -> vector[U]:
    """Applies a function to each element of the vector and a second vector or scalar."""
    if isinstance(self, vector) and isinstance(other, vector):
        return vector(func(x, y) for x, y in zip(self.values, other.values))
    elif isinstance(self, vector):
        return vector(func(x, other) for x in self.values)
    elif isinstance(other, vector):
        return vector(func(self, x) for x in other.values)
    else:
        return vector([func(self, other)])
