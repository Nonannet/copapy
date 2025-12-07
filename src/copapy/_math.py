from . import vector
from ._vectors import VecNumLike
from . import value, NumLike
from typing import TypeVar, Any, overload, Callable
from ._basic_types import add_op, unifloat
import math

T = TypeVar("T", int, float, value[int], value[float])
U = TypeVar("U", int, float)


@overload
def exp(x: float | int) -> float: ...
@overload
def exp(x: value[Any]) -> value[float]: ...
@overload
def exp(x: vector[Any]) -> vector[float]: ...
def exp(x: Any) -> Any:
    """Exponential function to basis e

    Arguments:
        x: Input value

    Returns:
        result of e**x
    """
    if isinstance(x, value):
        return add_op('exp', [x])
    if isinstance(x, vector):
        return x.map(exp)
    return float(math.exp(x))


@overload
def log(x: float | int) -> float: ...
@overload
def log(x: value[Any]) -> value[float]: ...
@overload
def log(x: vector[Any]) -> vector[float]: ...
def log(x: Any) -> Any:
    """Logarithm to basis e

    Arguments:
        x: Input value

    Returns:
        result of ln(x)
    """
    if isinstance(x, value):
        return add_op('log', [x])
    if isinstance(x, vector):
        return x.map(log)
    return float(math.log(x))


@overload
def pow(x: float | int, y: float | int) -> float: ...
@overload
def pow(x: value[Any], y: NumLike) -> value[float]: ...
@overload
def pow(x: NumLike, y: value[Any]) -> value[float]: ...
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
        return _map2(x, y, pow)
    if isinstance(y, int) and 0 <= y < 8:
        if y == 0:
            return 1
        m = x
        for _ in range(y - 1):
            m *= x
        return m
    if y == -1:
        return 1 / x
    if isinstance(x, value) or isinstance(y, value):
        return add_op('pow', [x, y])
    else:
        return float(x ** y)


@overload
def sqrt(x: float | int) -> float: ...
@overload
def sqrt(x: value[Any]) -> value[float]: ...
@overload
def sqrt(x: vector[Any]) -> vector[float]: ...
def sqrt(x: Any) -> Any:
    """Square root function

    Arguments:
        x: Input value

    Returns:
        Square root of x
    """
    if isinstance(x, value):
        return add_op('sqrt', [x])
    if isinstance(x, vector):
        return x.map(sqrt)
    return float(math.sqrt(x))


@overload
def sin(x: float | int) -> float: ...
@overload
def sin(x: value[Any]) -> value[float]: ...
@overload
def sin(x: vector[Any]) -> vector[float]: ...
def sin(x: Any) -> Any:
    """Sine function

    Arguments:
        x: Input value

    Returns:
        Square root of x
    """
    if isinstance(x, value):
        return add_op('sin', [x])
    if isinstance(x, vector):
        return x.map(sin)
    return math.sin(x)


@overload
def cos(x: float | int) -> float: ...
@overload
def cos(x: value[Any]) -> value[float]: ...
@overload
def cos(x: vector[Any]) -> vector[float]: ...
def cos(x: Any) -> Any:
    """Cosine function

    Arguments:
        x: Input value

    Returns:
        Cosine of x
    """
    if isinstance(x, value):
        return add_op('cos', [x])
    if isinstance(x, vector):
        return x.map(cos)
    return math.cos(x)


@overload
def tan(x: float | int) -> float: ...
@overload
def tan(x: value[Any]) -> value[float]: ...
@overload
def tan(x: vector[Any]) -> vector[float]: ...
def tan(x: Any) -> Any:
    """Tangent function

    Arguments:
        x: Input value

    Returns:
        Tangent of x
    """
    if isinstance(x, value):
        return add_op('tan', [x])
    if isinstance(x, vector):
        #return x.map(tan)
        return x.map(tan)
    return math.tan(x)


@overload
def atan(x: float | int) -> float: ...
@overload
def atan(x: value[Any]) -> value[float]: ...
@overload
def atan(x: vector[Any]) -> vector[float]: ...
def atan(x: Any) -> Any:
    """Inverse tangent function

    Arguments:
        x: Input value

    Returns:
        Inverse tangent of x
    """
    if isinstance(x, value):
        return add_op('atan', [x])
    if isinstance(x, vector):
        return x.map(atan)
    return math.atan(x)


@overload
def atan2(x: float | int, y: float | int) -> float: ...
@overload
def atan2(x: value[Any], y: NumLike) -> value[float]: ...
@overload
def atan2(x: NumLike, y: value[Any]) -> value[float]: ...
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
        return _map2(x, y, atan2)
    if isinstance(x, value) or isinstance(y, value):
        return add_op('atan2', [x, y])
    return math.atan2(x, y)


@overload
def asin(x: float | int) -> float: ...
@overload
def asin(x: value[Any]) -> value[float]: ...
@overload
def asin(x: vector[Any]) -> vector[float]: ...
def asin(x: Any) -> Any:
    """Inverse sine function

    Arguments:
        x: Input value

    Returns:
        Inverse sine of x
    """
    if isinstance(x, value):
        return add_op('asin', [x])
    if isinstance(x, vector):
        return x.map(asin)
    return math.asin(x)


@overload
def acos(x: float | int) -> float: ...
@overload
def acos(x: value[Any]) -> value[float]: ...
@overload
def acos(x: vector[Any]) -> vector[float]: ...
def acos(x: Any) -> Any:
    """Inverse cosine function

    Arguments:
        x: Input value

    Returns:
        Inverse cosine of x
    """
    if isinstance(x, value):
        return add_op('acos', [x])
    if isinstance(x, vector):
        return x.map(acos)
    return math.asin(x)


@overload
def get_42(x: float | int) -> float: ...
@overload
def get_42(x: value[Any]) -> value[float]: ...
def get_42(x: NumLike) -> value[float] | float:
    """Returns the value representing the constant 42"""
    if isinstance(x, value):
        return add_op('get_42', [x, x])
    return float((int(x) * 3.0 + 42.0) * 5.0 + 21.0)


#TODO: Add vector support
@overload
def abs(x: U) -> U: ...
@overload
def abs(x: value[U]) -> value[U]: ...
@overload
def abs(x: vector[U]) -> vector[U]: ...
def abs(x: U | value[U] | vector[U]) -> Any:
    """Absolute value function

    Arguments:
        x: Input value

    Returns:
        Absolute value of x
    """
    #tt = -x * (x < 0)
    ret = (x < 0) * -x + (x >= 0) * x
    return ret  # REMpyright: ignore[reportReturnType]


@overload
def sign(x: U) -> U: ...
@overload
def sign(x: value[U]) -> value[U]: ...
@overload
def sign(x: vector[U]) -> vector[U]: ...
def sign(x: U | value[U] | vector[U]) -> Any:
    """Return 1 for positive numbers and -1 for negative numbers.
    For an input of 0 the return value is 0.

    Arguments:
        x: Input value

    Returns:
        -1, 0 or 1
    """
    ret = (x > 0) - (x < 0)
    return ret


@overload
def clamp(x: value[U], min_value: U | value[U], max_value: U | value[U]) -> value[U]: ...
@overload
def clamp(x: U | value[U], min_value: value[U], max_value: U | value[U]) -> value[U]: ...
@overload
def clamp(x: U | value[U], min_value: U | value[U], max_value: value[U]) -> value[U]: ...
@overload
def clamp(x: U, min_value: U, max_value: U) -> U: ...
@overload
def clamp(x: vector[U], min_value: 'U | value[U]', max_value: 'U | value[U]') -> vector[U]: ...
def clamp(x: U | value[U] | vector[U], min_value: U | value[U], max_value:  U | value[U]) -> Any:
    """Clamp function to limit a value between a minimum and maximum.

    Arguments:
        x: Input value
        min_value: Minimum limit
        max_value: Maximum limit

    Returns:
        Clamped value of x
    """
    if isinstance(x, vector):
        return vector(clamp(comp, min_value, max_value) for comp in x.values)

    return (x < min_value) * min_value + \
          (x > max_value) * max_value + \
          ((x >= min_value) & (x <= max_value)) * x


@overload
def min(x: value[U], y: U | value[U]) -> value[U]: ...
@overload
def min(x: U | value[U], y: value[U]) -> value[U]: ...
@overload
def min(x: U, y: U) -> U: ...
def min(x: U | value[U], y: U | value[U]) -> Any:
    """Minimum function to get the smaller of two values.

    Arguments:
        x: First value
        y: Second value

    Returns:
        Minimum of x and y
    """
    return (x < y) * x + (x >= y) * y


@overload
def max(x: value[U], y: U | value[U]) -> value[U]: ...
@overload
def max(x: U | value[U], y: value[U]) -> value[U]: ...
@overload
def max(x: U, y: U) -> U: ...
def max(x: U | value[U], y: U | value[U]) -> Any:
    """Maximum function to get the larger of two values.

    Arguments:
        x: First value
        y: Second value

    Returns:
        Maximum of x and y
    """
    return (x > y) * x + (x <= y) * y


@overload
def lerp(v1: value[U], v2: U | value[U], t: unifloat) -> value[U]: ...
@overload
def lerp(v1: U | value[U], v2: value[U], t: unifloat) -> value[U]: ...
@overload
def lerp(v1: U | value[U], v2: U | value[U], t: value[float]) -> value[U]: ...
@overload
def lerp(v1: U, v2: U, t: float) -> U: ...
@overload
def lerp(v1: vector[U], v2: vector[U], t: unifloat) -> vector[U]: ...
def lerp(v1: U | value[U] | vector[U], v2: U | value[U] | vector[U], t:  unifloat) -> Any:
    """Linearly interpolate between two values or vectors v1 and v2 by a factor t."""
    if isinstance(v1, vector) or isinstance(v2, vector):
        assert isinstance(v1, vector) and isinstance(v2, vector), "None or both v1 and v2 must be vectors."
        assert len(v1.values) == len(v2.values), "Vectors must be of the same length."
        return vector(lerp(vv1, vv2, t) for vv1, vv2 in zip(v1.values, v2.values))
    return v1 * (1 - t) + v2 * t


@overload
def relu(x: U) -> U: ...
@overload
def relu(x: value[U]) -> value[U]: ...
@overload
def relu(x: vector[U]) -> vector[U]: ...
def relu(x: U | value[U] | vector[U]) -> Any:
    """Returns x for x > 0 and otherwise 0."""
    ret = (x > 0) * x
    return ret


def _map2(self: VecNumLike, other: VecNumLike, func: Callable[[Any, Any], value[U] | U]) -> vector[U]:
    """Applies a function to each element of the vector and a second vector or scalar."""
    if isinstance(self, vector) and isinstance(other, vector):
        return vector(func(x, y) for x, y in zip(self.values, other.values))
    elif isinstance(self, vector):
        return vector(func(x, other) for x in self.values)
    elif isinstance(other, vector):
        return vector(func(self, x) for x in other.values)
    else:
        return vector([func(self, other)])
