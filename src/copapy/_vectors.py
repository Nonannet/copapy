from copapy import NumLike, CPNumber, cpint, cpfloat, cpbool
from typing import Generic, TypeVar, Iterable, Any

T = TypeVar("T", bound=CPNumber)
T2 = TypeVar("T2", bound=CPNumber)

class cpvector(Generic[T]):
    def __init__(self, value: Iterable[T]):
        self.value = tuple(value)

    def __add__(self, other: 'cpvector[Any]') -> 'cpvector[CPNumber]':
        assert len(self.value) == len(other.value)
        tup = (a + b for a, b in zip(self.value, other.value))
        return cpvector(*(v for v in tup if isinstance(v, CPNumber)))