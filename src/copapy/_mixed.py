
from . import value
from typing import TypeVar, Iterable, Any, overload

T = TypeVar("T", int, float)


@overload
def mixed_sum(scalars: Iterable[float | value[float]]) -> float | value[float]: ...
@overload
def mixed_sum(scalars: Iterable[int | value[int]]) -> int | value[int]: ...
@overload
def mixed_sum(scalars: Iterable[T | value[T]]) -> T | value[T]: ...
def mixed_sum(scalars: Iterable[int | float | value[Any]]) -> Any:
    sl = list(scalars)
    return sum(a for a in sl if not isinstance(a, value)) +\
           sum(a for a in sl if isinstance(a, value))


def mixed_homogenize(scalars: Iterable[T | value[T]]) -> Iterable[T] | Iterable[value[T]]:
    if any(isinstance(val, value) for val in scalars):
        return (value(val) if not isinstance(val, value) else val for val in scalars)
    else:
        return (val for val in scalars if not isinstance(val, value))
