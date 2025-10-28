from . import variable
from typing import Generic, TypeVar, Iterable, Any, overload, TypeAlias
from ._math import sqrt

VecNumLike: TypeAlias = 'vector[int] | vector[float] | variable[int] | variable[float] | int | float'
VecIntLike: TypeAlias = 'vector[int] | variable[int] | int'
VecFloatLike: TypeAlias = 'vector[float] | variable[float] | float'
T = TypeVar("T", int, float)

epsilon = 1e-20


class vector(Generic[T]):
    """Mathematical vector class supporting basic operations and interactions with variables.
    """
    def __init__(self, values: Iterable[T | variable[T]]):
        """Create a vector with given values and variables.

        Args:
            values: iterable of constant values and variables
        """
        self.values: tuple[variable[T] | T, ...] = tuple(values)

    def __repr__(self) -> str:
        return f"vector({self.values})"

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> variable[T] | T:
        return self.values[index]

    @overload
    def __add__(self: 'vector[int]', other: VecFloatLike) -> 'vector[float]': ...
    @overload
    def __add__(self: 'vector[int]', other: VecIntLike) -> 'vector[int]': ...
    @overload
    def __add__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
    @overload
    def __add__(self, other: VecNumLike) -> 'vector[int] | vector[float]': ...
    def __add__(self, other: VecNumLike) -> Any:
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a + b for a, b in zip(self.values, other.values))
        return vector(a + other for a in self.values)

    @overload
    def __radd__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
    @overload
    def __radd__(self: 'vector[int]', other: variable[int] | int) -> 'vector[int]': ...
    def __radd__(self, other: Any) -> Any:
        return self + other

    @overload
    def __sub__(self: 'vector[int]', other: VecFloatLike) -> 'vector[float]': ...
    @overload
    def __sub__(self: 'vector[int]', other: VecIntLike) -> 'vector[int]': ...
    @overload
    def __sub__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
    @overload
    def __sub__(self, other: VecNumLike) -> 'vector[int] | vector[float]': ...
    def __sub__(self, other: VecNumLike) -> Any:
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a - b for a, b in zip(self.values, other.values))
        return vector(a - other for a in self.values)

    @overload
    def __rsub__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
    @overload
    def __rsub__(self: 'vector[int]', other: variable[int] | int) -> 'vector[int]': ...
    def __rsub__(self, other: VecNumLike) -> Any:
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(b - a for a, b in zip(self.values, other.values))
        return vector(other - a for a in self.values)

    @overload
    def __mul__(self: 'vector[int]', other: VecFloatLike) -> 'vector[float]': ...
    @overload
    def __mul__(self: 'vector[int]', other: VecIntLike) -> 'vector[int]': ...
    @overload
    def __mul__(self: 'vector[float]', other: 'vector[int] | float | int | variable[int]') -> 'vector[float]': ...
    @overload
    def __mul__(self, other: VecNumLike) -> 'vector[int] | vector[float]': ...
    def __mul__(self, other: VecNumLike) -> Any:
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a * b for a, b in zip(self.values, other.values))
        return vector(a * other for a in self.values)

    @overload
    def __rmul__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
    @overload
    def __rmul__(self: 'vector[int]', other: variable[int] | int) -> 'vector[int]': ...
    def __rmul__(self, other: VecNumLike) -> Any:
        return self * other

    def __truediv__(self, other: VecNumLike) -> 'vector[float]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a / b for a, b in zip(self.values, other.values))
        return vector(a / other for a in self.values)

    def __rtruediv__(self, other: VecNumLike) -> 'vector[float]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(b / a for a, b in zip(self.values, other.values))
        return vector(other / a for a in self.values)

    @overload
    def dot(self: 'vector[int]', other: 'vector[int]') -> int | variable[int]: ...
    @overload
    def dot(self, other: 'vector[float]') -> float | variable[float]: ...
    @overload
    def dot(self: 'vector[float]', other: 'vector[int] | vector[float]') -> float | variable[float]: ...
    @overload
    def dot(self, other: 'vector[int] | vector[float]') -> float | int | variable[float] | variable[int]: ...
    def dot(self, other: 'vector[int] | vector[float]') -> Any:
        assert len(self.values) == len(other.values)
        return sum(a * b for a, b in zip(self.values, other.values))

    # @ operator
    @overload
    def __matmul__(self: 'vector[int]', other: 'vector[int]') -> int | variable[int]: ...
    @overload
    def __matmul__(self, other: 'vector[float]') -> float | variable[float]: ...
    @overload
    def __matmul__(self: 'vector[float]', other: 'vector[int] | vector[float]') -> float | variable[float]: ...
    @overload
    def __matmul__(self, other: 'vector[int] | vector[float]') -> float | int | variable[float] | variable[int]: ...
    def __matmul__(self, other: 'vector[int] | vector[float]') -> Any:
        return self.dot(other)

    def cross(self: 'vector[float]', other: 'vector[float]') -> 'vector[float]':
        """3D cross product"""
        assert len(self.values) == 3 and len(other.values) == 3
        a1, a2, a3 = self.values
        b1, b2, b3 = other.values
        return vector([
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ])

    @overload
    def sum(self: 'vector[int]') -> int | variable[int]: ...
    @overload
    def sum(self: 'vector[float]') -> float | variable[float]: ...
    def sum(self) -> Any:
        """Sum of all vector elements."""
        return sum(a for a in self.values if isinstance(a, variable)) +\
               sum(a for a in self.values if not isinstance(a, variable))

    def magnitude(self) -> 'float | variable[float]':
        """Magnitude (length) of the vector."""
        s = sum(a * a for a in self.values)
        return sqrt(s) if isinstance(s, variable) else sqrt(s)

    def normalize(self) -> 'vector[float]':
        """Returns a normalized (unit length) version of the vector."""
        mag = self.magnitude() + epsilon
        return self / mag

    def __iter__(self) -> Iterable[variable[T] | T]:
        return iter(self.values)
