from typing import Any, Generic, Sequence, Iterable, overload
from ._basic_types import value, Net, ArrayNet, ArrayConst, ArrayOp, ArrayElement, ArrayPack, NumLike, transl_type, value_from_number, generic_sdb
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


def array_dtype(flat: Iterable[Any], strict: bool = False) -> str | None:
    """Element type of an array holding the values (copapy values or numbers),
    None if they can not be stored in one array: values of different types,
    int values together with float numbers, bools or other types. With strict
    int and float numbers are not mixed either."""
    value_types: set[str] = set()
    number_types: set[type] = set()
    for v in flat:
        if isinstance(v, value):
            value_types.add(v.dtype)
        elif isinstance(v, bool) or not isinstance(v, int | float):
            return None
        else:
            number_types.add(type(v))
    if len(value_types) > 1 or value_types - {'int', 'float'} or (strict and len(number_types) > 1):
        return None
    if value_types == {'int'}:
        return 'int' if number_types <= {int} else None
    if value_types == {'float'} or float in number_types:
        return 'float'
    return 'int'


def _size(shape: Sequence[int]) -> int:
    size = 1
    for d in shape:
        size *= d
    return size


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for d in reversed(shape):
        strides.insert(0, stride)
        stride *= d
    return tuple(strides)


def broadcast_shapes(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast shape of two shapes following numpy rules"""
    ndim = max(len(shape1), len(shape2))
    padded1 = (1,) * (ndim - len(shape1)) + shape1
    padded2 = (1,) * (ndim - len(shape2)) + shape2
    result: list[int] = []
    for d1, d2 in zip(padded1, padded2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Incompatible shapes for broadcasting: {shape1} vs {shape2}")
        result.append(d2 if d1 == 1 else d1)
    return tuple(result)


# Number of dimensions supported by the strided copy stencil
COPY_DIMS = 4


def _array_out_type(op: str, dtype1: str, dtype2: str) -> str:
    return 'float' if op in ('div', 'pow', 'atan2') or dtype1 != dtype2 else dtype1


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
        """Create an array from (nested) sequences of numbers or copapy values.
        An array of numbers can be overwritten on the target with Target.write_value.
        An array containing copapy values is filled with the current values on
        each run (one load and store per element).

        Arguments:
            values: Nested sequences of int or float numbers or copapy values.
            dtype: Element type ('int' or 'float'), inferred from values if omitted.
        """
        flat, shape = _flatten(values)
        if not shape:
            raise ValueError("Array requires at least one dimension")
        if not flat:
            raise ValueError("Empty arrays are not supported")
        if dtype is None:
            dtype = array_dtype(flat)
            if dtype is None:
                raise ValueError("Array values must be int or float numbers or copapy values of a single type")
        assert dtype in ('int', 'float'), f"Unsupported array type {dtype}"

        if any(isinstance(v, value) for v in flat):
            nets: list[Net] = []
            for v in flat:
                if isinstance(v, value):
                    if transl_type(v.dtype) != dtype:
                        raise ValueError(f"Value of type {v.dtype} in {dtype} array")
                    nets.append(v.net)
                else:
                    nets.append(value_from_number(int(v) if dtype == 'int' else float(v)).net)
            self._init(ArrayPack(nets, dtype).result, shape)
        else:
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

    def element(self, flat_index: int) -> value[TNum]:
        """Element by its flat index. No code is generated, the element
        is read directly from the array memory."""
        return value(Net(self.dtype, ArrayElement(self.net, flat_index)))

    def __getitem__(self, key: int | slice | Sequence[int | slice]) -> 'Any':
        """Get an element (all dimensions indexed by integers) or a sub-array
        (slices or fewer indices than dimensions). Sub-arrays are copied by
        a strided copy stencil."""
        keys = key if isinstance(key, Sequence) else (key,)
        if len(keys) > self.ndim:
            raise IndexError(f"Too many indices for array of rank {self.ndim}")
        if len(keys) == self.ndim and all(isinstance(k, int) for k in keys):
            return self.element(self._flat_index(keys))  # type: ignore[arg-type]

        strides = _contiguous_strides(self.shape)
        offset = 0
        shape: list[int] = []
        new_strides: list[int] = []
        for i, k in enumerate(keys):
            dim = self.shape[i]
            if isinstance(k, int):
                if not -dim <= k < dim:
                    raise IndexError(f"Index {k} out of bounds for dimension of size {dim}")
                offset += (k % dim) * strides[i]
            else:
                assert isinstance(k, slice), f"Indices must be integers or slices, not {type(k)}"
                start, stop, step = k.indices(dim)
                offset += start * strides[i]
                shape.append(len(range(start, stop, step)))
                new_strides.append(step * strides[i])
        shape += self.shape[len(keys):]
        new_strides += strides[len(keys):]
        return self._strided(offset, tuple(shape), tuple(new_strides))

    def _strided(self, offset: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> 'array[TNum]':
        """New array with the elements at offset + sum(index * stride)"""
        size = _size(shape)
        if size == 0:
            raise ValueError("Empty arrays are not supported")

        # Remove dimensions of size 1 and merge dimensions that are contiguous to each other
        dims: list[tuple[int, int]] = []
        for n, s in zip(shape, strides):
            if n == 1:
                continue
            if dims and dims[-1][1] == s * n:
                dims[-1] = (dims[-1][0] * n, s)
            else:
                dims.append((n, s))

        if offset == 0 and size == self.size and (not dims or dims == [(size, 1)]):
            return self.reshape(*shape)  # No copy required
        if len(dims) > COPY_DIMS:
            raise NotImplementedError(f"Strided copy for more than {COPY_DIMS} non-contiguous dimensions")

        dims = [(1, 0)] * (COPY_DIMS - len(dims)) + dims
        params = array([offset] + [n for n, _ in dims] + [s for _, s in dims], 'int')
        node = _add_array_op('copy_arr', [self.net, params.net], self.dtype, size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, shape)

    def reshape(self, *shape: int) -> 'array[TNum]':
        """Same data with a new shape, a dimension of -1 is inferred."""
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = tuple(shape[0])
        if shape.count(-1) == 1:
            known = -_size(shape)
            if known <= 0 or self.size % known:
                raise ValueError(f"Can not reshape array of size {self.size} into shape {shape}")
            shape = tuple(self.size // known if d == -1 else d for d in shape)
        if _size(shape) != self.size:
            raise ValueError(f"Can not reshape array of size {self.size} into shape {shape}")
        return array._from_net(self.net, tuple(shape))

    def flatten(self) -> 'array[TNum]':
        return self.reshape(self.size)

    def transpose(self, *axes: int) -> 'array[TNum]':
        """Permute the axes (reverse them if no axes are given)."""
        if len(axes) == 1 and isinstance(axes[0], Sequence):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(range(self.ndim - 1, -1, -1))
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError(f"Invalid axes {axes} for array of rank {self.ndim}")
        strides = _contiguous_strides(self.shape)
        return self._strided(0, tuple(self.shape[a] for a in axes), tuple(strides[a] for a in axes))

    @property
    def T(self) -> 'array[TNum]':
        return self.transpose()

    def broadcast_to(self, shape: Sequence[int]) -> 'array[TNum]':
        """Repeat dimensions of size 1 (numpy broadcasting rules)."""
        shape = tuple(shape)
        if broadcast_shapes(self.shape, shape) != shape:
            raise ValueError(f"Can not broadcast shape {self.shape} to {shape}")
        padded = (1,) * (len(shape) - self.ndim) + self.shape
        strides = (0,) * (len(shape) - self.ndim) + _contiguous_strides(self.shape)
        return self._strided(0, shape, tuple(0 if p == 1 else s for p, s in zip(padded, strides)))

    def _binary_op(self, op: str, other: 'array[Any] | NumLike', reverse: bool = False) -> 'array[Any]':
        if isinstance(other, array) and other.shape != self.shape:
            shape = broadcast_shapes(self.shape, other.shape)
            return self.broadcast_to(shape)._binary_op(op, other.broadcast_to(shape), reverse)

        n = value_from_number(self.size).net
        if isinstance(other, array):
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

    def __pow__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int) and not isinstance(other, bool) and 1 <= other < 8:
            ret: array[Any] = self
            for _ in range(other - 1):
                ret = ret * self
            return ret
        return self._binary_op('pow', other)

    def __rpow__(self, other: NumLike) -> 'array[float]':
        return self._binary_op('pow', other, reverse=True)

    def __abs__(self) -> 'array[TNum]':
        return self._unary_op('abs')

    def _unary_op(self, op: str) -> 'array[Any]':
        """Element-wise function: sqrt, exp, log, sin, cos, tan, asin, acos,
        atan, tanh (float result) or abs (result of the element type)"""
        n = value_from_number(self.size).net
        node = _add_array_op(f"{op}_{self.dtype}arr", [self.net, n], self.dtype if op == 'abs' else 'float', self.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, self.shape)

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
        """Matrix multiplication for 1D and 2D operands (numpy semantics)."""
        if self.ndim == 1 and other.ndim == 1:
            if self.size != other.size:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
            return self.dot(other)
        if self.ndim == 1 and other.ndim == 2:
            ret = self.reshape(1, self.size).matmul(other)
            assert isinstance(ret, array)
            return ret.reshape(ret.size)
        if self.ndim == 2 and other.ndim == 2:
            m, k = self.shape
            if k != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
            n = other.shape[1]
            out_dtype = _array_out_type('mul', self.dtype, other.dtype)
            dims = array([m, k, n], 'int')
            node = _add_array_op(f"matmul_{self.dtype}arr_{other.dtype}arr",
                                 [self.net, other.net, dims.net], out_dtype, m * n)
            assert isinstance(node.result, ArrayNet)
            return array._from_net(node.result, (m, n))
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
