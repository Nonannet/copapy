from . import variable
from ._mixed import mixed_sum, mixed_homogenize
from typing import TypeVar, Iterable, Any, overload, TypeAlias, Callable, Iterator, Generic
import copapy as cp
from ._helper_types import TNum

#VecNumLike: TypeAlias = 'vector[int] | vector[float] | variable[int] | variable[float] | int | float | bool'
VecNumLike: TypeAlias = 'vector[Any] | variable[Any] | int | float | bool'
VecIntLike: TypeAlias = 'vector[int] | variable[int] | int'
VecFloatLike: TypeAlias = 'vector[float] | variable[float] | float'
U = TypeVar("U", int, float)

epsilon = 1e-20


class vector(Generic[TNum]):
    """Mathematical vector class supporting basic operations and interactions with variables.
    """
    def __init__(self, values: Iterable[TNum | variable[TNum]]):
        """Create a vector with given values and variables.

        Args:
            values: iterable of constant values and variables
        """
        self.values: tuple[variable[TNum] | TNum, ...] = tuple(values)

    def __repr__(self) -> str:
        return f"vector({self.values})"

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> variable[TNum] | TNum:
        return self.values[index]

    def __neg__(self) -> 'vector[TNum]':
        return vector(-a for a in self.values)

    def __iter__(self) -> Iterator[variable[TNum] | TNum]:
        return iter(self.values)

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
    @overload
    def __radd__(self, other: VecNumLike) -> 'vector[Any]': ...
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
    @overload
    def __rsub__(self, other: VecNumLike) -> 'vector[Any]': ...
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
    def __mul__(self: 'vector[float]', other: VecNumLike) -> 'vector[float]': ...
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
    @overload
    def __rmul__(self, other: VecNumLike) -> 'vector[Any]': ...
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
        assert len(self.values) == len(other.values), "Vectors must be of same length."
        return mixed_sum(a * b for a, b in zip(self.values, other.values))

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
        assert len(self.values) == 3 and len(other.values) == 3, "Both vectors must be 3-dimensional."
        a1, a2, a3 = self.values
        b1, b2, b3 = other.values
        return vector([
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ])
    
    def __gt__(self, other: VecNumLike) -> 'vector[int]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a > b for a, b in zip(self.values, other.values))
        return vector(a > other for a in self.values)

    def __lt__(self, other: VecNumLike) -> 'vector[int]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a < b for a, b in zip(self.values, other.values))
        return vector(a < other for a in self.values)

    def __ge__(self, other: VecNumLike) -> 'vector[int]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a >= b for a, b in zip(self.values, other.values))
        return vector(a >= other for a in self.values)

    def __le__(self, other: VecNumLike) -> 'vector[int]':
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a <= b for a, b in zip(self.values, other.values))
        return vector(a <= other for a in self.values)

    def __eq__(self, other: VecNumLike) -> 'vector[int]':  # type: ignore
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a == b for a, b in zip(self.values, other.values))
        return vector(a == other for a in self.values)

    def __ne__(self, other: VecNumLike) -> 'vector[int]':  # type: ignore
        if isinstance(other, vector):
            assert len(self.values) == len(other.values)
            return vector(a != b for a, b in zip(self.values, other.values))
        return vector(a != other for a in self.values)

    @overload
    def sum(self: 'vector[int]') -> int | variable[int]: ...
    @overload
    def sum(self: 'vector[float]') -> float | variable[float]: ...
    def sum(self) -> Any:
        """Sum of all vector elements."""
        return mixed_sum(self.values)

    def magnitude(self) -> 'float | variable[float]':
        """Magnitude (length) of the vector."""
        s = mixed_sum(a * a for a in self.values)
        return cp.sqrt(s)

    def normalize(self) -> 'vector[float]':
        """Returns a normalized (unit length) version of the vector."""
        mag = self.magnitude() + epsilon
        return self / mag

    def homogenize(self) -> 'vector[TNum]':
        if any(isinstance(val, variable) for val in self.values):
            return vector(mixed_homogenize(self))
        else:
            return self

    def map(self, func: Callable[[Any], variable[U] | U]) -> 'vector[U]':
        """Applies a function to each element of the vector and returns a new vector."""
        return vector(func(x) for x in self.values)


# Utility functions for 3D vectors with two arguments

def cross_product(v1: vector[float], v2: vector[float]) -> vector[float]:
    """Calculate the cross product of two 3D vectors."""
    return v1.cross(v2)


def dot_product(v1: vector[float], v2: vector[float]) -> 'float | variable[float]':
    """Calculate the dot product of two vectors."""
    return v1.dot(v2)


def distance(v1: vector[float], v2: vector[float]) -> 'float | variable[float]':
    """Calculate the Euclidean distance between two vectors."""
    diff = v1 - v2
    return diff.magnitude()


def scalar_projection(v1: vector[float], v2: vector[float]) -> 'float | variable[float]':
    """Calculate the scalar projection of v1 onto v2."""
    dot_prod = v1.dot(v2)
    mag_v2 = v2.magnitude() + epsilon
    return dot_prod / mag_v2


def vector_projection(v1: vector[float], v2: vector[float]) -> vector[float]:
    """Calculate the vector projection of v1 onto v2."""
    dot_prod = v1.dot(v2)
    mag_v2_squared = v2.magnitude() ** 2 + epsilon
    scalar_proj = dot_prod / mag_v2_squared
    return v2 * scalar_proj


def angle_between(v1: vector[float], v2: vector[float]) -> 'float | variable[float]':
    """Calculate the angle in radians between two vectors."""
    dot_prod = v1.dot(v2)
    mag_v1 = v1.magnitude()
    mag_v2 = v2.magnitude()
    cos_angle = dot_prod / (mag_v1 * mag_v2 + epsilon)
    return cp.acos(cos_angle)


def rotate_vector(v: vector[float], axis: vector[float], angle: 'float | variable[float]') -> vector[float]:
    """Rotate vector v around a given axis by a specified angle using Rodrigues' rotation formula."""
    k = axis.normalize()
    cos_angle = cp.cos(angle)
    sin_angle = cp.sin(angle)
    term1 = v * cos_angle
    term2 = k.cross(v) * sin_angle
    term3 = k * (k.dot(v)) * (1 - cos_angle)
    return term1 + term2 + term3
