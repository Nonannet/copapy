from typing import Any, Generic, Sequence, Iterable, overload
from ._basic_types import value, Net, ArrayNet, ArrayConst, ArrayOp, ArrayElement, NumLike, transl_type, value_from_number, generic_sdb
from ._helper_types import TNum


def _flatten(values: Any) -> tuple[list[Any], tuple[int, ...]]:
    """Flatten nested sequences and return the values and the shape"""
    if not isinstance(values, Sequence):
        return [values], ()
    if not values:
        return [], (0,)
    sub = [_flatten(v) for v in values]
    sub_shape = sub[0][1]
    if any(s[1] != sub_shape for s in sub):
        raise ValueError("All elements must have consistent shape")
    return [v for s in sub for v in s[0]], (len(values),) + sub_shape


def _nest(values: Sequence[Any], shape: tuple[int, ...]) -> list[Any]:
    """Build nested lists from flat values"""
    if len(shape) <= 1:
        return list(values)
    step = len(values) // shape[0]
    return [_nest(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def _array_out_type(op: str, dtype1: str, dtype2: str) -> str:
    return 'float' if op == 'div' or dtype1 != dtype2 else dtype1


def _add_array_op(typed_op: str, args: list[Net], out_dtype: str, length: int | None = None) -> ArrayOp:
    if typed_op not in generic_sdb.stencil_definitions:
        raise NotImplementedError(f"Array operation {typed_op} not available, stencils might need to be rebuilt")
    return ArrayOp(typed_op, args, out_dtype, length)


class array(Generic[TNum]):
    """Homogeneous n-dimensional array stored contiguously (row-major) in the
    target memory. In contrast to tensor, operations on arrays compile to a single
    array stencil per operation instead of one stencil per element, which keeps
    compile time and code size independent of the array size.

    Attributes:
        shape: Size of each dimension.
        dtype: Element type ('int' or 'float').
        net: Underlying array net in the computation graph.
    """
    def __init__(self, values: 'Sequence[TNum] | Sequence[Sequence[TNum]] | Sequence[Any]', dtype: str | None = None):
        """Create an array from (nested) sequences of numbers. The values can be
        overwritten on the target with Target.write_value.

        Arguments:
            values: Nested sequences of int or float numbers.
            dtype: Element type ('int' or 'float'), inferred from values if omitted.
        """
        flat, shape = _flatten(values)
        if not shape:
            raise ValueError("Array requires at least one dimension")
        if any(isinstance(v, value) for v in flat):
            raise NotImplementedError("Arrays from copapy values are not supported yet")
        if not all(isinstance(v, int | float) for v in flat):
            raise ValueError("Array values must be int or float numbers")
        if dtype is None:
            dtype = 'int' if all(isinstance(v, int) for v in flat) else 'float'
        assert dtype in ('int', 'float'), f"Unsupported array type {dtype}"
        conv = int if dtype == 'int' else float
        source = ArrayConst([conv(v) for v in flat], dtype)
        self._init(ArrayNet(dtype, source, len(flat)), shape)

    def _init(self, net: ArrayNet, shape: tuple[int, ...]) -> None:
        self.net = net
        self.shape = shape
        self.dtype = net.dtype

    @classmethod
    def _from_net(cls, net: ArrayNet, shape: tuple[int, ...]) -> 'array[Any]':
        ret: array[Any] = cls.__new__(cls)
        ret._init(net, shape)
        return ret

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return self.net.length

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"array(shape={self.shape}, dtype={self.dtype})"

    def __bool__(self) -> bool:
        raise TypeError("The truth value of an array is ambiguous")

    def _flat_index(self, key: int | Sequence[int]) -> int:
        indices = (key,) if isinstance(key, int) else tuple(key)
        if len(indices) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(indices)}")
        flat = 0
        for i, dim in zip(indices, self.shape):
            if not -dim <= i < dim:
                raise IndexError(f"Index {i} out of bounds for dimension of size {dim}")
            flat = flat * dim + (i % dim)
        return flat

    def __getitem__(self, key: int | Sequence[int]) -> value[TNum]:
        """Get a single element by constant indices. No code is generated,
        the element is read directly from the array memory."""
        return value(Net(self.dtype, ArrayElement(self.net, self._flat_index(key))))

    def reshape(self, *shape: int) -> 'array[TNum]':
        size = 1
        for d in shape:
            size *= d
        if size != self.size:
            raise ValueError(f"Can not reshape array of size {self.size} into shape {shape}")
        return array._from_net(self.net, tuple(shape))

    def _binary_op(self, op: str, other: 'array[Any] | NumLike', reverse: bool = False) -> 'array[Any]':
        n = value_from_number(self.size).net
        if isinstance(other, array):
            if other.shape != self.shape:
                raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")
            a, b = (other, self) if reverse else (self, other)
            typed_op = f"{op}_{a.dtype}arr_{b.dtype}arr"
            out_dtype = _array_out_type(op, a.dtype, b.dtype)
            args = [a.net, b.net, n]
        elif isinstance(other, value | int | float):
            scalar = other if isinstance(other, value) else value_from_number(other)
            s_dtype = transl_type(scalar.dtype)
            if reverse:
                typed_op = f"{op}_{s_dtype}_{self.dtype}arr"
                out_dtype = _array_out_type(op, s_dtype, self.dtype)
                args = [scalar.net, self.net, n]
            else:
                typed_op = f"{op}_{self.dtype}arr_{s_dtype}"
                out_dtype = _array_out_type(op, self.dtype, s_dtype)
                args = [self.net, scalar.net, n]
        else:
            return NotImplemented
        node = _add_array_op(typed_op, args, out_dtype, self.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, self.shape)

    @overload
    def __add__(self: 'array[int]', other: 'array[int] | value[int] | int') -> 'array[int]': ...
    @overload
    def __add__(self, other: 'array[Any] | NumLike') -> 'array[Any]': ...
    def __add__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int | float) and other == 0:
            return self
        return self._binary_op('add', other)

    def __radd__(self, other: NumLike) -> 'array[Any]':
        return self + other

    def __sub__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int | float) and other == 0:
            return self
        return self._binary_op('sub', other)

    def __rsub__(self, other: NumLike) -> 'array[Any]':
        return self._binary_op('sub', other, reverse=True)

    @overload
    def __mul__(self: 'array[int]', other: 'array[int] | value[int] | int') -> 'array[int]': ...
    @overload
    def __mul__(self, other: 'array[Any] | NumLike') -> 'array[Any]': ...
    def __mul__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int | float) and other == 1:
            return self
        return self._binary_op('mul', other)

    def __rmul__(self, other: NumLike) -> 'array[Any]':
        return self * other

    def __truediv__(self, other: 'array[Any] | NumLike') -> 'array[float]':
        return self._binary_op('div', other)

    def __rtruediv__(self, other: NumLike) -> 'array[float]':
        return self._binary_op('div', other, reverse=True)

    def __neg__(self) -> 'array[TNum]':
        return self._binary_op('mul', -1)

    def sum(self) -> value[TNum]:
        """Sum of all elements."""
        n = value_from_number(self.size).net
        node = _add_array_op(f"sum_{self.dtype}arr", [self.net, n], self.dtype)
        return value(node.result)

    def dot(self, other: 'array[Any]') -> value[Any]:
        """Dot product of two arrays with the same number of elements."""
        if other.size != self.size:
            raise ValueError(f"Size mismatch: {self.size} and {other.size}")
        n = value_from_number(self.size).net
        out_dtype = _array_out_type('mul', self.dtype, other.dtype)
        node = _add_array_op(f"dot_{self.dtype}arr_{other.dtype}arr", [self.net, other.net, n], out_dtype)
        return value(node.result)

    def matmul(self, other: 'array[Any]') -> 'value[Any] | array[Any]':
        """Matrix multiplication for 1D @ 1D (dot product) and 2D @ 1D."""
        if self.ndim == 1 and other.ndim == 1:
            return self.dot(other)
        if self.ndim == 2 and other.ndim == 1:
            m, n = self.shape
            if n != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
            out_dtype = _array_out_type('mul', self.dtype, other.dtype)
            node = _add_array_op(f"matvec_{self.dtype}arr_{other.dtype}arr",
                                 [self.net, other.net, value_from_number(m).net, value_from_number(n).net],
                                 out_dtype, m)
            assert isinstance(node.result, ArrayNet)
            return array._from_net(node.result, (m,))
        raise NotImplementedError(f"matmul not implemented for shapes {self.shape} @ {other.shape}")

    def __matmul__(self, other: 'array[Any]') -> 'value[Any] | array[Any]':
        return self.matmul(other)

    def get_values(self, data: Sequence[int | float]) -> list[Any]:
        """Nest flat element data according to the array shape"""
        return _nest(data, self.shape)

    @staticmethod
    def flatten_data(data: Iterable[Any] | Sequence[Any]) -> list[Any]:
        return _flatten(data)[0]
