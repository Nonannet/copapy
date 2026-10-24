from typing import Any, Callable, Iterator, Sequence, TypeAlias, overload
from ._basic_types import value, to_float, unifloat, NumLike
from ._vectors import vector
from ._tensors import tensor
from ._arrays import _flatten
from ._casts import to_float as _tensor_to_float
from ._math import sqrt, abs as _abs, atan2, exp, cos, sin, log

CxScalarLike: TypeAlias = 'cxvalue | complex | NumLike'
CxTensorLike: TypeAlias = 'cxtensor | tensor[Any] | vector[Any] | CxScalarLike'


def _is_zero(x: Any) -> bool:
    """True for a number that is zero (a constant, not a copapy value)"""
    return not isinstance(x, value) and x == 0


def _float(x: NumLike) -> unifloat:
    return to_float(x) if isinstance(x, value) else float(x)


def _sub(a: Any, b: Any) -> Any:
    """a - b, with -b instead of 0 - b for a constant zero"""
    if _is_zero(a):
        return -b
    return a - b


def _scalar_parts(x: Any) -> tuple[unifloat, unifloat] | None:
    """Real and imaginary part of a complex scalar or a real number, None for other types"""
    if isinstance(x, cxvalue):
        return x.real, x.imag
    if isinstance(x, complex):
        return x.real, x.imag
    if isinstance(x, (value, int, float)):
        return _float(x), 0.0
    return None


class cxvalue:
    """Complex scalar of two float parts. The parts are copapy values or
    float constants: operations with constant zero parts (e.g. with a
    real number) are removed at trace time.

    Attributes:
        real: Real part
        imag: Imaginary part
    """

    def __init__(self, real: 'NumLike | complex' = 0.0, imag: NumLike = 0.0):
        """Create a complex scalar.

        Arguments:
            real: Real part or a Python complex number (imag is then ignored)
            imag: Imaginary part
        """
        if isinstance(real, complex):
            real, imag = real.real, real.imag
        self.real: unifloat = _float(real)
        self.imag: unifloat = _float(imag)

    @classmethod
    def from_polar(cls, r: NumLike, phi: NumLike) -> 'cxvalue':
        """Complex number r * e^(i*phi)"""
        return cls(r * cos(phi), r * sin(phi))

    def __repr__(self) -> str:
        return f"cxvalue(real={self.real}, imag={self.imag})"

    def conj(self) -> 'cxvalue':
        """Complex conjugate"""
        return cxvalue(self.real, -self.imag)

    def abs2(self) -> unifloat:
        """Squared magnitude re^2 + im^2"""
        if _is_zero(self.imag):
            return self.real * self.real
        return self.real * self.real + self.imag * self.imag

    def __abs__(self) -> unifloat:
        if _is_zero(self.imag):
            return _abs(self.real)
        return sqrt(self.abs2())

    def angle(self) -> unifloat:
        """Argument (phase angle) in radians in the range [-pi, pi]"""
        return atan2(self.imag, self.real)

    def exp(self) -> 'cxvalue':
        """e^z"""
        e = exp(self.real)
        if _is_zero(self.imag):
            return cxvalue(e)
        return cxvalue(e * cos(self.imag), e * sin(self.imag))

    def log(self) -> 'cxvalue':
        """Principal value of the natural logarithm"""
        return cxvalue(log(abs(self)), self.angle())

    def __neg__(self) -> 'cxvalue':
        return cxvalue(-self.real, -self.imag)

    def __pos__(self) -> 'cxvalue':
        return self

    def _tensor_op(self, other: Any, op: 'Callable[[cxtensor, cxvalue], cxtensor]') -> 'cxtensor | None':
        """Operation with a tensor operand by cxtensor, None for other operands"""
        if isinstance(other, (tensor, vector, cxtensor)):
            return op(cxtensor(other), self)
        return None

    @overload
    def __add__(self, other: 'cxtensor | tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __add__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __add__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__radd__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return cxvalue(self.real + p[0], self.imag + p[1])

    def __radd__(self, other: CxScalarLike) -> 'cxvalue':
        return self + other

    @overload
    def __sub__(self, other: 'cxtensor | tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __sub__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __sub__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__rsub__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return cxvalue(self.real - p[0], self.imag - p[1])

    @overload
    def __rsub__(self, other: 'tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __rsub__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __rsub__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__sub__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return cxvalue(_sub(p[0], self.real), _sub(p[1], self.imag))

    @overload
    def __mul__(self, other: 'cxtensor | tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __mul__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __mul__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__rmul__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        a, b = self.real, self.imag
        c, d = p
        return cxvalue(_sub(a * c, b * d), a * d + b * c)

    def __rmul__(self, other: CxScalarLike) -> 'cxvalue':
        return self * other

    @overload
    def __truediv__(self, other: 'cxtensor | tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __truediv__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __truediv__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__rtruediv__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return _div_scalar(self.real, self.imag, *p)

    @overload
    def __rtruediv__(self, other: 'tensor[Any] | vector[Any]') -> 'cxtensor': ...
    @overload
    def __rtruediv__(self, other: CxScalarLike) -> 'cxvalue': ...
    def __rtruediv__(self, other: Any) -> Any:
        if (ret := self._tensor_op(other, cxtensor.__truediv__)) is not None:
            return ret
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return _div_scalar(*p, self.real, self.imag)

    def __pow__(self, n: int) -> 'cxvalue':
        """Integer power by repeated squaring"""
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            return 1.0 / (self ** -n)
        result: cxvalue = cxvalue(1.0)
        base = self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def __eq__(self, other: CxScalarLike) -> Any:  # type: ignore[override]
        """Equality of both parts: a bool or a value[int] (0 or 1)"""
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return (self.real == p[0]) & (self.imag == p[1])

    def __ne__(self, other: CxScalarLike) -> Any:  # type: ignore[override]
        if (p := _scalar_parts(other)) is None:
            return NotImplemented
        return (self.real != p[0]) | (self.imag != p[1])

    __hash__ = None  # type: ignore[assignment]


def _div_scalar(a: unifloat, b: unifloat, c: unifloat, d: unifloat) -> cxvalue:
    """(a + bi) / (c + di)"""
    if _is_zero(d):
        return cxvalue(a / c, b / c)
    den = c * c + d * d
    return cxvalue((a * c + b * d) / den, _sub(b * c, a * d) / den)


# Parts of a complex tensor: the real part is a tensor, the imaginary
# part a tensor or None for a real tensor (no operations for it)
_Part: TypeAlias = 'tensor[float] | None'


def _add(a: Any, b: Any) -> Any:
    if a is None:
        return b
    return a if b is None else a + b


def _subt(a: Any, b: Any) -> Any:
    if b is None:
        return a
    return -b if a is None else a - b


def _mul(a: Any, b: Any, op: Callable[[Any, Any], Any]) -> Any:
    return None if a is None or b is None else op(a, b)


def _float_tensor(t: Any) -> tensor[float]:
    ft: tensor[Any] = t if isinstance(t, tensor) else tensor(t)
    return ft if ft.dtype == 'float' else _tensor_to_float(ft)


def _full(shape: tuple[int, ...], x: Any) -> tensor[float]:
    size = 1
    for d in shape:
        size *= d
    return tensor([_float(x)] * size, shape)


def _tensor_parts(x: Any) -> tuple[Any, Any] | None:
    """Parts of a cxtensor operand: tensors or scalars, None for a zero imaginary part"""
    if isinstance(x, cxtensor):
        return x._re, x._im
    if isinstance(x, vector):
        return tensor(x), None
    if isinstance(x, tensor):
        return x, None
    p = _scalar_parts(x)
    if p is not None:
        return p[0], None if _is_zero(p[1]) else p[1]
    if isinstance(x, Sequence):
        c = cxtensor(x)
        return c._re, c._im
    return None


class cxtensor:
    """Complex tensor of two float tensors for the real and the imaginary
    part. All operations are composed of tensor operations, so packed
    tensors use the array stencils. The imaginary part of a real tensor
    is not stored and generates no code.

    Attributes:
        real: Real part (float tensor)
        imag: Imaginary part (float tensor)
    """

    def __init__(self, real: Any, imag: Any = None):
        """Create a complex tensor.

        Arguments:
            real: Real part as tensor, vector or nested sequence, or a nested
                sequence of complex numbers and cxvalues (imag must be None)
            imag: Imaginary part with the shape of real, None for a real tensor
        """
        if isinstance(real, cxtensor):
            assert imag is None, "imag must be None for a cxtensor"
            re, im = real._re, real._im
        elif imag is None and not isinstance(real, (tensor, vector)):
            flat, shape = _flatten(real)
            parts = [_scalar_parts(v) for v in flat]
            if any(p is None for p in parts):
                raise ValueError("Elements must be numbers, complex numbers, copapy values or cxvalues")
            re = tensor([p[0] for p in parts if p], shape)
            im = None if all(p and _is_zero(p[1]) for p in parts) else tensor([p[1] for p in parts if p], shape)
        else:
            re = real
            im = None if imag is None else imag
        self._re: tensor[float] = _float_tensor(re)
        self._im: _Part = None if im is None else _float_tensor(im)
        if self._im is not None and self._im.shape != self._re.shape:
            raise ValueError(f"Shapes of real ({self._re.shape}) and imaginary part ({self._im.shape}) differ")

    @classmethod
    def _from_parts(cls, re: Any, im: Any) -> 'cxtensor':
        """Result of an operation: scalar parts are broadcast to the shape of the tensor part"""
        if not isinstance(re, tensor):
            assert isinstance(im, tensor)
            re = _full(im.shape, re)
        if im is not None and not isinstance(im, tensor):
            im = _full(re.shape, im)
        if im is not None and im.shape != re.shape:
            im = im._expand_to_shape(re.shape)  # Only one operand had an imaginary part
        return cls(re, im)

    @classmethod
    def from_polar(cls, r: Any, phi: Any) -> 'cxtensor':
        """Complex tensor r * e^(i*phi), element-wise"""
        return cls._from_parts(r * cos(phi), r * sin(phi))

    @property
    def real(self) -> tensor[float]:
        return self._re

    @property
    def imag(self) -> tensor[float]:
        return self._im if self._im is not None else _full(self.shape, 0.0)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._re.shape

    @property
    def ndim(self) -> int:
        return self._re.ndim

    def size(self) -> int:
        return self._re.size()

    def __len__(self) -> int:
        return len(self._re)

    def __repr__(self) -> str:
        return f"cxtensor(shape={self.shape})"

    def _map(self, f: Callable[[tensor[float]], Any]) -> 'cxtensor':
        """Apply a shape operation to both parts"""
        return cxtensor(f(self._re), None if self._im is None else f(self._im))

    def __getitem__(self, key: Any) -> 'cxtensor':
        return self._map(lambda t: t[key])

    def __iter__(self) -> Iterator['cxtensor']:
        if self.ndim == 0:
            raise TypeError("Cannot iterate over a 0-d cxtensor")
        for i in range(self.shape[0]):
            yield self[i]

    def item(self) -> cxvalue:
        """The element of a cxtensor with one element"""
        return cxvalue(self._re.item(), 0.0 if self._im is None else self._im.item())

    def get_scalar(self, *key: int) -> cxvalue:
        """Element at the given indices"""
        return cxvalue(self._re.get_scalar(*key), 0.0 if self._im is None else self._im.get_scalar(*key))

    def reshape(self, *shape: int) -> 'cxtensor':
        return self._map(lambda t: t.reshape(*shape))

    def flatten(self) -> 'cxtensor':
        return self._map(lambda t: t.flatten())

    def transpose(self, *axes: int) -> 'cxtensor':
        return self._map(lambda t: t.transpose(*axes))

    @property
    def T(self) -> 'cxtensor':
        """Transpose (without conjugation)"""
        return self.transpose()

    @property
    def H(self) -> 'cxtensor':
        """Conjugate transpose"""
        return self.conj().transpose()

    def conj(self) -> 'cxtensor':
        """Element-wise complex conjugate"""
        return cxtensor(self._re, None if self._im is None else -self._im)

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> 'cxvalue | cxtensor':
        """Sum of all elements as cxvalue for axis None without keepdims, else a cxtensor"""
        re = self._re.sum(axis, keepdims)
        im = None if self._im is None else self._im.sum(axis, keepdims)
        if isinstance(re, tensor):
            return cxtensor(re, im)
        assert not isinstance(im, tensor)
        return cxvalue(re, 0.0 if im is None else im)

    def abs2(self) -> tensor[float]:
        """Element-wise squared magnitude"""
        if self._im is None:
            return self._re * self._re
        return self._re * self._re + self._im * self._im

    def __abs__(self) -> tensor[float]:
        if self._im is None:
            return _abs(self._re)
        return sqrt(self.abs2())

    def angle(self) -> tensor[float]:
        """Element-wise argument (phase angle) in radians"""
        return atan2(self.imag, self._re)

    def exp(self) -> 'cxtensor':
        """Element-wise e^z"""
        e = exp(self._re)
        if self._im is None:
            return cxtensor(e)
        return cxtensor(e * cos(self._im), e * sin(self._im))

    def __neg__(self) -> 'cxtensor':
        return cxtensor(-self._re, None if self._im is None else -self._im)

    def __pos__(self) -> 'cxtensor':
        return self

    def __add__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return cxtensor._from_parts(self._re + p[0], _add(self._im, p[1]))

    def __radd__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return cxtensor._from_parts(p[0] + self._re, _add(p[1], self._im))

    def __sub__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return cxtensor._from_parts(self._re - p[0], _subt(self._im, p[1]))

    def __rsub__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return cxtensor._from_parts(p[0] - self._re, _subt(p[1], self._im))

    @staticmethod
    def _product(a: Any, b: Any, c: Any, d: Any, op: Callable[[Any, Any], Any]) -> 'cxtensor':
        """(a + bi) op (c + di) for a bilinear op (element-wise or matrix product)"""
        return cxtensor._from_parts(_subt(op(a, c), _mul(b, d, op)),
                                    _add(_mul(a, d, op), _mul(b, c, op)))

    def __mul__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return self._product(self._re, self._im, p[0], p[1], lambda x, y: x * y)

    def __rmul__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return self._product(p[0], p[1], self._re, self._im, lambda x, y: x * y)

    def __matmul__(self, other: 'cxtensor | tensor[Any] | vector[Any] | Sequence[Any]') -> 'cxtensor':
        if (p := _tensor_parts(other)) is None or not isinstance(p[0], tensor):
            return NotImplemented
        return self._product(self._re, self._im, p[0], p[1], lambda x, y: x @ y)

    def __rmatmul__(self, other: 'tensor[Any] | vector[Any] | Sequence[Any]') -> 'cxtensor':
        if (p := _tensor_parts(other)) is None or not isinstance(p[0], tensor):
            return NotImplemented
        return self._product(p[0], p[1], self._re, self._im, lambda x, y: x @ y)

    @staticmethod
    def _quotient(a: Any, b: Any, c: Any, d: Any) -> 'cxtensor':
        """(a + bi) / (c + di) element-wise"""
        if d is None:
            return cxtensor._from_parts(a / c, None if b is None else b / c)
        den = c * c + d * d
        re = _add(a * c, _mul(b, d, lambda x, y: x * y)) / den
        im = _subt(_mul(b, c, lambda x, y: x * y), a * d) / den
        return cxtensor._from_parts(re, im)

    def __truediv__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return self._quotient(self._re, self._im, p[0], p[1])

    def __rtruediv__(self, other: CxTensorLike) -> 'cxtensor':
        if (p := _tensor_parts(other)) is None:
            return NotImplemented
        return self._quotient(p[0], p[1], self._re, self._im)
