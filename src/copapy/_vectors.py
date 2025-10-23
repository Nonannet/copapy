from numpy import isin
from copapy import NumLike, CPNumber, variable
from typing import Generic, TypeVar, Iterable, Any, overload

from copapy._basic_types import TNum

T = TypeVar("T", int, float, bool)
T2 = TypeVar("T2", bound=CPNumber)

class vector(Generic[T]):
    def __init__(self, values: Iterable[T | variable[T]]):
        #self.values: tuple[variable[T], ...] = tuple(v if isinstance(v, variable) else variable(v) for v in values)
        self.values: tuple[variable[T] | T, ...] = tuple(values)

    @overload
    def __add__(self, other: 'vector[float] | variable[float] | float') -> 'vector[float]':
        ...

    @overload
    def __add__(self: 'vector[T]', other: 'vector[int] | variable[int] | int') -> 'vector[T]':
        ...

    def __add__(self, other: 'vector[Any] | variable[Any] | float | int') -> Any:
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a + b for a, b in zip(self.values, other.values))
        else:
            return vector(a + other for a in self.values)
    
    #@overload
    #def sum(self: 'vector[float]') -> variable[float]:
    #    ...

    #@overload
    #def sum(self: 'vector[int]') -> variable[int]:
    #    ...

    #def sum(self: 'vector[T]') -> variable[T] | T:
    #    comp_time = sum(v for v in self.values if not isinstance(v, variable))
    #    run_time = sum(v for v in self.values if isinstance(v, variable))
    #    if isinstance(run_time, variable):
    #        return comp_time + run_time  # type: ignore
    #    else:
    #        return comp_time
    