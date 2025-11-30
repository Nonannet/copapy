
from . import variable
from typing import TypeVar, Iterable, Any, overload

T = TypeVar("T", int, float)


@overload
def mixed_sum(scalars: Iterable[float | variable[float]]) -> float | variable[float]: ...
@overload
def mixed_sum(scalars: Iterable[int | variable[int]]) -> int | variable[int]: ...
@overload
def mixed_sum(scalars: Iterable[T | variable[T]]) -> T | variable[T]: ...
def mixed_sum(scalars: Iterable[int | float | variable[Any]]) -> Any:
    sl = list(scalars)
    return sum(a for a in sl if not isinstance(a, variable)) +\
           sum(a for a in sl if isinstance(a, variable))


def mixed_homogenize(scalars: Iterable[T | variable[T]]) -> Iterable[T] | Iterable[variable[T]]:
    if any(isinstance(val, variable) for val in scalars):
        return (variable(val) if not isinstance(val, variable) else val for val in scalars)
    else:
        return (val for val in scalars if not isinstance(val, variable))
