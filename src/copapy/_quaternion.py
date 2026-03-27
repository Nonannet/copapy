from typing import overload, Iterable, Callable, Any
from ._vectors import vector
from ._tensors import tensor
import copapy as cp
from ._basic_types import NumLike, value, unifloat, ArrayType
from ._mixed import mixed_sum


class quaternion(ArrayType[float]):
    """Mathematical quaternion class for representing 3D rotations.

    Attributes:
        values (tuple[unifloat, ...]): Internal storage of the (w, x, y, z) components.
        w, x, y, z (unifloat): Property accessors to individual components.
    """
    def __init__(
        self,
        w: unifloat | Iterable[unifloat] = 1.0,
        x: unifloat = 0.0,
        y: unifloat = 0.0,
        z: unifloat = 0.0):
        """Create a quaternion with given components.

        Arguments:
            w: w component, or an iterable of 4 components.
            x: x component (ignored if w is an iterable).
            y: y component (ignored if w is an iterable).
            z: z component (ignored if w is an iterable).
        """
        self.shape = (4,)
        if isinstance(w, Iterable):
            self.values = tuple(v for v in w)
            assert len(self.values) == 4, "Sequence must have exactly 4 elements for quaternion initialization."
        else:
            self.values = (w, x, y, z)


    @classmethod
    def from_euler(cls, roll: NumLike, pitch: NumLike, yaw: NumLike) -> 'quaternion':
        """Create a quaternion from Euler angles (roll, pitch, yaw).

        Arguments:
            roll: Rotation around the x-axis in radians.
            pitch: Rotation around the y-axis in radians.
            yaw: Rotation around the z-axis in radians.

        Returns:
            A quaternion representing the rotation.
        """
        cy = cp.cos(yaw * 0.5)
        sy = cp.sin(yaw * 0.5)
        ci = cp.cos(pitch * 0.5)
        sp = cp.sin(pitch * 0.5)
        cr = cp.cos(roll * 0.5)
        sr = cp.sin(roll * 0.5)

        w = cr * ci * cy + sr * sp * sy
        x = sr * ci * cy - cr * sp * sy
        y = cr * sp * cy + sr * ci * sy
        z = cr * ci * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def identity(cls) -> 'quaternion':
        """Return the identity quaternion (no rotation).

        Returns:
            The identity quaternion (x=0, y=0, z=0, w=1).
        """
        return cls(w=1.0, x=0.0, y=0.0, z=0.0)

    @property
    def x(self) -> unifloat:
        return self.values[1]

    @property
    def y(self) -> unifloat:
        return self.values[2]

    @property
    def z(self) -> unifloat:
        return self.values[3]

    @property
    def w(self) -> unifloat:
        return self.values[0]


    def normalize(self) -> 'quaternion':
        """Normalize the quaternion to unit length.

        Returns:
            A normalized (unit) quaternion. Returns identity if the norm is zero.
        """
        n = self.norm()
        if not isinstance(n, value) and n == 0:
            return quaternion.identity()
        return quaternion(v / n for v in self.values)

    def toRotationMatrix(self) -> tensor[float]:
        """Convert the quaternion to a 4x4 rotation matrix.

        Returns:
            A 4x4 tensor representing the rotation matrix.
        """
        w, x, y, z = self.values
        x2 = x + x
        y2 = y + y
        z2 = z + z
        xx = x * x2
        xy = x * y2
        xz = x * z2
        yy = y * y2
        yz = y * z2
        zz = z * z2
        wx = w * x2
        wy = w * y2
        wz = w * z2

        s1: list[unifloat] = [1.0 - (yy + zz), xy - wz, xz + wy, 0.0]
        s2: list[unifloat] = [xy + wz, 1.0 - (xx + zz), yz - wx, 0.0]
        s3: list[unifloat] = [xz - wy, yz + wx, 1.0 - (xx + yy), 0.0]
        s4: list[unifloat] = [0.0, 0.0, 0.0, 1.0]
        return tensor([s1, s2, s3, s4])

    def toEulerAngles(self) -> vector[float]:
        """Convert the quaternion to Euler angles (roll, pitch, yaw).

        Returns:
            A vector of [roll, pitch, yaw] in radians.
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        yaw = cp.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pitch_sin = cp.clamp(2 * (w * y - z * x), -1.0, 1.0)
        pitch = cp.asin(pitch_sin)
        roll = cp.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

        return vector([roll, pitch, yaw])

    def toAxisAngle(self) -> tuple[vector[float], unifloat]:
        """Convert the quaternion to axis-angle representation.

        Returns:
            A tuple of (axis, angle) where axis is a unit vector and angle is in radians.
        """
        n = self.normalize()
        sin_half_angle_sq = 1 - n.w * n.w
        is_near_identity = sin_half_angle_sq < 1e-6
        s = 1 / cp.sqrt(cp.iif(is_near_identity, 1e-6, sin_half_angle_sq))
        angle = cp.iif(is_near_identity, 0.0, 2 * cp.acos(n.w))
        axis = vector([
            cp.iif(is_near_identity, 1.0, n.x * s),
            cp.iif(is_near_identity, 0.0, n.y * s),
            cp.iif(is_near_identity, 0.0, n.z * s),
        ])
        return axis, angle

    def conjugate(self) -> 'quaternion':
        """Return the conjugate of the quaternion.

        Returns:
            The conjugate quaternion (negates x, y, z components).
        """
        return quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> 'quaternion':
        """Return the inverse of the quaternion.

        Returns:
            The inverse quaternion. Returns identity if the norm is zero.
        """
        n2 = self.norm() ** 2
        if not isinstance(n2, value) and n2 == 0:
            return quaternion.identity()
        return quaternion(v / n2 for v in self.conjugate().values)

    def norm(self) -> unifloat:
        """Calculate the norm (magnitude) of the quaternion.

        Returns:
            The norm (square root of the sum of squared components).
        """
        return cp.sqrt(mixed_sum(v**2 for v in self.values))

    def rotate_vector(self, vec: vector[float]) -> vector[float]:
        """Rotate a 3D vector by this quaternion.

        Arguments:
            vec: A 3D vector to rotate.

        Returns:
            The rotated vector.
        """
        q_vec = quaternion(0, *vec)
        rotated_q = self @ q_vec @ self.inverse()
        return vector(rotated_q.values[1:])

    def map(self, func: Callable[[Any], value[float] | float]) -> 'quaternion':
        """Applies a function to each element of the quaternion and returns a new quaternion.

        Arguments:
            func: A function that takes a single argument.

        Returns:
            A new quaternion with the function applied to each element.
        """
        return quaternion(func(x) for x in self.values)

    def __neg__(self) -> 'quaternion':
        return quaternion(-self.w, -self.x, -self.y, -self.z)

    def __abs__(self) -> unifloat:
        return self.norm()

    def __repr__(self) -> str:
        return f"vector({self.values})"

    def __len__(self) -> int:
        return len(self.values)

    @overload
    def __getitem__(self, index: int) -> value[float] | float: ...
    @overload
    def __getitem__(self, index: slice) -> 'vector[float]': ...
    def __getitem__(self, index: int | slice) -> 'vector[float] | value[float] | float':
        if isinstance(index, slice):
            return vector(self.values[index])
        return self.values[index]

    @overload
    def __add__(self, other: 'quaternion') -> 'quaternion': ...
    @overload
    def __add__(self, other: NumLike) -> 'quaternion': ...
    def __add__(self, other: 'quaternion | NumLike') -> 'quaternion':
        if isinstance(other, quaternion):
            return quaternion(a + b for a, b in zip(self.values, other.values))
        if isinstance(other, value):
            return quaternion(v + other for v in self.values)
        o = value(other)  # Make sure a single constant is allocated
        return quaternion(a + o if isinstance(a, value) else a + other for a in self.values)

    def __radd__(self, other: int | float) -> 'quaternion':
        return self + other

    @overload
    def __sub__(self, other: 'quaternion') -> 'quaternion': ...
    @overload
    def __sub__(self, other: NumLike) -> 'quaternion': ...
    def __sub__(self, other: 'quaternion | NumLike') -> 'quaternion':
        if isinstance(other, quaternion):
            return quaternion(a - b for a, b in zip(self.values, other.values))
        if isinstance(other, value):
            return quaternion(v - other for v in self.values)
        o = value(other)  # Make sure a single constant is allocated
        return quaternion(a - o if isinstance(a, value) else a - other for a in self.values)

    def __rsub__(self, other: NumLike) -> 'quaternion':
        return -self + other

    def __mul__(self, other: NumLike) -> 'quaternion':
        if isinstance(other, value):
            return quaternion(v * other for v in self.values)
        o = value(other)  # Make sure a single constant is allocated
        return quaternion(v * o if isinstance(v, value) else v * other for v in self.values)

    def __rmul__(self, other: NumLike) -> 'quaternion':
        return self * other

    def __matmul__(self, other: 'quaternion') -> 'quaternion':
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return quaternion(w, x, y, z)

    def __truediv__(self, other: NumLike) -> 'quaternion':
        if isinstance(other, value):
            return quaternion(v / other for v in self.values)
        o = value(other)  # Make sure a single constant is allocated
        return quaternion(v / o if isinstance(v, value) else v / other for v in self.values)
