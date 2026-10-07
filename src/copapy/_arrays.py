from typing import Any, Generic, Sequence, Iterable, Callable, TypeVar, overload
from ._basic_types import value, Net, ArrayNet, ArrayConst, ArrayOp, ArrayElement, ArrayPack, NumLike, transl_type, value_from_number, generic_sdb, to_float, add_op
from ._helper_types import TNum


def _flatten(values: Any) -> tuple[list[Any], tuple[int, ...]]:
    """Flatten nested sequences and return the values and the shape"""
    if not isinstance(values, Sequence) or isinstance(values, str):
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


def element_dtype(flat: Iterable[Any]) -> str:
    """Common type of numbers and copapy values (numpy-style promotion): 'float'
    if any element is float, otherwise 'int' if any element is int, otherwise
    'bool'."""
    has_float = has_int = False
    has_any = False
    for v in flat:
        if isinstance(v, value):
            dtype = v.dtype
        elif isinstance(v, bool):
            dtype = 'bool'
        elif isinstance(v, int):
            dtype = 'int'
        elif isinstance(v, float):
            dtype = 'float'
        else:
            raise ValueError("Elements must be numbers or copapy values")
        has_float |= dtype == 'float'
        has_int |= dtype == 'int'
        has_any = True
    if has_float or not has_any:
        return 'float'
    return 'int' if has_int else 'bool'


def convert_element(v: Any, dtype: str) -> Any:
    """Convert a number or copapy value to dtype: numbers directly, int and bool
    values to float by a float_int stencil, bool values to int without code.
    Float elements are not converted to int."""
    if dtype == 'float':
        return to_float(v) if isinstance(v, value) else float(v)
    if dtype == 'int':
        if isinstance(v, value):
            assert v.dtype != 'float', "Float value in int array"
            return v if v.dtype == 'int' else value(v.net, 'int')
        assert not isinstance(v, float), "Float number in int array"
        return int(v)
    return v


def _clamped_index(index: value[Any], dim: int) -> 'value[int] | int':
    """Index computed at runtime in the range [0, dim - 1]: negative indices
    count from the end (like Python), then the index is clamped"""
    if transl_type(index.dtype) != 'int':
        raise TypeError(f"Indices must be int values, not {index.dtype}")
    if dim == 1:
        return 0
    if index.dtype == 'bool':
        index = value(index.net, 'int')
    wrapped = index + (index < 0) * dim
    return add_op('min', [add_op('max', [wrapped, 0]), dim - 1])


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
    target memory. Operations on arrays compile to a array stencil instead of
    one stencil per element, which keeps compile time and code size
    independent of the array size.

    Attributes:
        shape: Size of each dimension.
        dtype: Element type ('int', 'float' or 'bool')
        net: Underlying array net in the computation graph.
    """
    def __init__(self, values: 'Sequence[TNum] | Sequence[Sequence[TNum]] | Sequence[Any]', dtype: str | None = None):
        """Create an array from (nested) sequences of numbers or copapy values.
        An array of numbers can be overwritten on the target with Target.write_value.
        An array containing copapy values is filled with the current values on
        each run (one load and store per element).

        Elements of different types are promoted: the array is float if any element
        is float, int elements are converted (int variables by a float_int stencil).
        An array of only bool elements is a bool array (stored as int).

        Arguments:
            values: Nested sequences of int or float numbers or copapy values.
            dtype: Element type ('int', 'float' or 'bool'), inferred from values if omitted.
        """
        flat, shape = _flatten(values)
        if not shape:
            raise ValueError("Array requires at least one dimension")
        if not flat:
            raise ValueError("Empty arrays are not supported")
        inferred = element_dtype(flat)
        if dtype is None:
            dtype = inferred
        if dtype not in ('int', 'float', 'bool'):
            raise ValueError(f"Unsupported array type {dtype}")
        if (dtype == 'int' and inferred == 'float') or (dtype == 'bool' and inferred != 'bool'):
            raise ValueError(f"{inferred} elements in a {dtype} array")

        stored = transl_type(dtype)  # Bool is stored as int
        flat = [convert_element(v, stored) for v in flat]
        if any(isinstance(v, value) for v in flat):
            nets = [v.net if isinstance(v, value) else value_from_number(v).net for v in flat]
            self._init(ArrayPack(nets, stored).result, shape, dtype)
        else:
            source = ArrayConst(flat, stored)
            self._init(ArrayNet(stored, source, len(flat)), shape, dtype)

    def _init(self, net: ArrayNet, shape: tuple[int, ...], dtype: str | None = None) -> None:
        assert dtype is None or transl_type(dtype) == net.dtype
        self.net = net
        self.shape = shape
        self.dtype = dtype or net.dtype

    @classmethod
    def _from_net(cls, net: ArrayNet, shape: tuple[int, ...], dtype: str | None = None) -> 'array[Any]':
        ret: array[Any] = cls.__new__(cls)
        ret._init(net, shape, dtype)
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
        return value(Net(self.net.dtype, ArrayElement(self.net, flat_index)), self.dtype)

    def __getitem__(self, key: 'int | value[int] | slice | Sequence[int | value[int] | slice]') -> 'Any':
        """Get an element (all dimensions indexed) or a sub-array (slices or fewer
        indices than dimensions). Sub-arrays are copied by a strided copy stencil."""
        keys = key if isinstance(key, Sequence) else (key,)
        if len(keys) > self.ndim:
            raise IndexError(f"Too many indices for array of rank {self.ndim}")
        if len(keys) == self.ndim and all(isinstance(k, int) for k in keys):
            return self.element(self._flat_index(keys))  # type: ignore[arg-type]

        strides = _contiguous_strides(self.shape)
        offset: int | value[int] = 0
        shape: list[int] = []
        new_strides: list[int] = []
        for i, k in enumerate(keys):
            dim = self.shape[i]
            if isinstance(k, int):
                if not -dim <= k < dim:
                    raise IndexError(f"Index {k} out of bounds for dimension of size {dim}")
                offset += (k % dim) * strides[i]
            elif isinstance(k, value):
                offset = offset + _clamped_index(k, dim) * strides[i]
            else:
                assert isinstance(k, slice), f"Indices must be integers or slices, not {type(k)}"
                start, stop, step = k.indices(dim)
                offset += start * strides[i]
                shape.append(len(range(start, stop, step)))
                new_strides.append(step * strides[i])
        shape += self.shape[len(keys):]
        new_strides += strides[len(keys):]
        ret = self._strided(offset, tuple(shape), tuple(new_strides))
        return ret.element(0) if not shape else ret

    def _strided(self, offset: 'int | value[int]', shape: tuple[int, ...], strides: tuple[int, ...]) -> 'array[TNum]':
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

        if isinstance(offset, int) and offset == 0 and size == self.size and (not dims or dims == [(size, 1)]):
            return self.reshape(*shape)  # No copy required
        if len(dims) > COPY_DIMS:
            raise NotImplementedError(f"Strided copy for more than {COPY_DIMS} non-contiguous dimensions")

        dims = [(1, 0)] * (COPY_DIMS - len(dims)) + dims
        params = array([offset] + [n for n, _ in dims] + [s for _, s in dims], 'int')
        node = _add_array_op('copy_arr', [self.net, params.net], self.net.dtype, size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, shape, self.dtype)

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
        return array._from_net(self.net, tuple(shape), self.dtype)

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

    def _computed(self) -> 'array[Any]':
        """The array with the type of computed results (bool as int), no code"""
        return self if self.dtype == self.net.dtype else array._from_net(self.net, self.shape)

    def _binary_op(self, op: str, other: 'array[Any] | NumLike', reverse: bool = False) -> 'array[Any]':
        if isinstance(other, array) and other.shape != self.shape:
            shape = broadcast_shapes(self.shape, other.shape)
            return self.broadcast_to(shape)._binary_op(op, other.broadcast_to(shape), reverse)

        n = value_from_number(self.size).net
        if isinstance(other, array):
            a, b = (other, self) if reverse else (self, other)
            typed_op = f"{op}_{a.net.dtype}arr_{b.net.dtype}arr"
            out_dtype = _array_out_type(op, a.net.dtype, b.net.dtype)
            args = [a.net, b.net, n]
        elif isinstance(other, value | int | float):
            scalar = other if isinstance(other, value) else value_from_number(other)
            s_dtype = transl_type(scalar.dtype)
            if reverse:
                typed_op = f"{op}_{s_dtype}_{self.net.dtype}arr"
                out_dtype = _array_out_type(op, s_dtype, self.net.dtype)
                args = [scalar.net, self.net, n]
            else:
                typed_op = f"{op}_{self.net.dtype}arr_{s_dtype}"
                out_dtype = _array_out_type(op, self.net.dtype, s_dtype)
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
            return self._computed()
        return self._binary_op('add', other)

    def __radd__(self, other: NumLike) -> 'array[Any]':
        return self + other

    def __sub__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int | float) and other == 0:
            return self._computed()
        return self._binary_op('sub', other)

    def __rsub__(self, other: NumLike) -> 'array[Any]':
        return self._binary_op('sub', other, reverse=True)

    @overload
    def __mul__(self: 'array[int]', other: 'array[int] | value[int] | int') -> 'array[int]': ...
    @overload
    def __mul__(self, other: 'array[Any] | NumLike') -> 'array[Any]': ...
    def __mul__(self, other: 'array[Any] | NumLike') -> 'array[Any]':
        if isinstance(other, int | float) and other == 1:
            return self._computed()
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
            ret: array[Any] = self._computed()
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
        node = _add_array_op(f"{op}_{self.net.dtype}arr", [self.net, n], self.net.dtype if op == 'abs' else 'float', self.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, self.shape)

    def sum(self) -> value[TNum]:
        """Sum of all elements."""
        n = value_from_number(self.size).net
        node = _add_array_op(f"sum_{self.net.dtype}arr", [self.net, n], self.net.dtype)
        return value(node.result)

    def min(self) -> value[TNum]:
        """Smallest element."""
        return self._min_max('min')

    def max(self) -> value[TNum]:
        """Largest element."""
        return self._min_max('max')

    def _min_max(self, op: str) -> value[TNum]:
        n = value_from_number(self.size).net
        node = _add_array_op(f"{op}_{self.net.dtype}arr", [self.net, n], self.net.dtype)
        return value(node.result, self.dtype)

    def sort(self) -> 'array[TNum]':
        """Sorted copy of a 1D array. A sorting network is used: the execution
        time only depends on the number of elements, not on the values."""
        self._check_1d('sort')
        n = value_from_number(self.size).net
        node = _add_array_op(f"sort_{self.net.dtype}arr", [self.net, n], self.net.dtype, self.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, self.shape, self.dtype)

    def argsort(self) -> 'array[int]':
        """Indices that sort a 1D array, equal elements in order of their index
        (like a stable sort). The execution time only depends on the number of
        elements, not on the values."""
        self._check_1d('argsort')
        n = value_from_number(self.size).net
        node = _add_array_op(f"argsort_{self.net.dtype}arr", [self.net, n], 'int', self.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, self.shape)

    def _check_1d(self, name: str) -> None:
        if self.ndim != 1:
            raise ValueError(f"{name} requires a 1D array, got shape {self.shape}")

    def dot(self, other: 'array[Any]') -> value[Any]:
        """Dot product of two arrays with the same number of elements."""
        if other.size != self.size:
            raise ValueError(f"Size mismatch: {self.size} and {other.size}")
        n = value_from_number(self.size).net
        out_dtype = _array_out_type('mul', self.net.dtype, other.net.dtype)
        node = _add_array_op(f"dot_{self.net.dtype}arr_{other.net.dtype}arr", [self.net, other.net, n], out_dtype)
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
            out_dtype = _array_out_type('mul', self.net.dtype, other.net.dtype)
            dims = array([m, k, n], 'int')
            node = _add_array_op(f"matmul_{self.net.dtype}arr_{other.net.dtype}arr",
                                 [self.net, other.net, dims.net], out_dtype, m * n)
            assert isinstance(node.result, ArrayNet)
            return array._from_net(node.result, (m, n))
        if self.ndim == 2 and other.ndim == 1:
            m, n = self.shape
            if n != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
            out_dtype = _array_out_type('mul', self.net.dtype, other.net.dtype)
            node = _add_array_op(f"matvec_{self.net.dtype}arr_{other.net.dtype}arr",
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


TArrayType = TypeVar('TArrayType', bound='ArrayType[Any]')


class ArrayType(Generic[TNum]):
    """Base class for a fixed number of elements that are numbers or
    copapy values.
    """

    pack_threshold: int | None = None
    """Instances are packed if more elements than pack_threshold are copapy values
    or constants other than zero, None disables packing and type promotion"""

    shape: tuple[int, ...]
    ndim: int

    # The elements are defined by _values if set, otherwise by _array
    _values: tuple[TNum | value[TNum], ...] | None  # Element values
    _array: array[Any] | None  # Packed array
    _element_refs: tuple[value[TNum], ...] | None  # Cached references into _array
    _packed: bool | None  # True/False: always/never use array stencils, None: by pack_threshold
    _dtype: str

    def _set_elements(self, values: Iterable[Any], shape: tuple[int, ...], packed: bool | None = None) -> None:
        """Set the elements, converted to their common type if packing is enabled for the class"""
        self._values = tuple(values)
        self._array = None
        self._element_refs = None
        self._packed = packed
        self.shape = shape
        self.ndim = len(shape)
        dtype = element_dtype(self._values)
        self._dtype = dtype
        if type(self).pack_threshold is not None:
            self._values = tuple(convert_element(v, dtype) for v in self._values)

    def _copy_elements(self, other: 'ArrayType[Any]', packed: bool | None = None) -> None:
        """Take over the elements (and packed array) of another instance"""
        self.shape = other.shape
        self.ndim = other.ndim
        self._dtype = other._dtype
        self._packed = packed
        self._element_refs = None
        if packed is False:
            self._values = other.values
            self._array = None
        else:
            self._values = other._values
            self._array = other._array
            self._element_refs = other._element_refs

    @classmethod
    def _from_array(cls: type[TArrayType], arr: array[Any], packed: bool | None = None) -> TArrayType:
        """Instance backed by an array, the elements are references to the array memory"""
        ret = cls.__new__(cls)
        ret._values = None
        ret._array = arr
        ret._element_refs = None
        ret._packed = packed
        ret._dtype = arr.dtype
        ret.shape = arr.shape
        ret.ndim = arr.ndim
        return ret

    @property
    def values(self) -> tuple[TNum | value[TNum], ...]:
        """Flat tuple of all elements"""
        if self._values is not None:
            return self._values
        if self._element_refs is None:
            assert self._array is not None
            self._element_refs = tuple(self._array.element(i) for i in range(self._array.size))
        return self._element_refs

    @property
    def dtype(self) -> str:
        """Type of the elements: 'int', 'float' or 'bool'"""
        return self._dtype

    def map(self, func: Callable[[TNum | value[TNum]], Any]) -> 'ArrayType[Any]':
        return self

    def __bool__(self) -> bool:
        raise TypeError(f"The truth value of a {type(self).__name__} is ambiguous, "
                        "compare the .values or the elements instead")

    def _get_array(self, force: bool = False) -> 'array[Any] | None':
        """Packed array of the elements if array stencils are used for them

        Packed are instances with more than pack_threshold elements that are copapy
        values or constants other than zero. Operations with zeros are eliminated
        in the scalar path, so only these elements generate code there.

        Arguments:
            force: Pack an instance containing copapy values independent of
                pack_threshold, constant instances are only packed by the rule above
        """
        threshold = type(self).pack_threshold
        if self._packed is False or (threshold is None and not self._packed) or not self.shape:
            return None
        if self._array is None:
            values = self.values
            if not self._packed and not (force and not self._is_constant()):
                assert threshold is not None
                if sum(1 for v in values if isinstance(v, value) or v != 0) <= threshold:
                    return None
            self._array = array(list(values), self._dtype).reshape(*self.shape)
        return self._array

    def _force_array(self) -> 'array[Any]':
        """Elements as array independent of the pack threshold, constant
        elements are stored as data"""
        arr = self._get_array(force=True)
        if arr is None:
            # Packing disabled for this instance or class or constant elements
            arr = array(list(self.values), self._dtype).reshape(*self.shape)
        return arr

    def _is_constant(self) -> bool:
        """True if all elements are numbers (no copapy values)"""
        return self._values is not None and not any(isinstance(v, value) for v in self._values)

    def _view_array(self) -> 'array[Any] | None':
        """Array for reshaping, transposing and slicing: only if the instance is already
        packed and not made of constants, a copy of constants does not generate code"""
        if self._array is None or self._packed is False:
            return None
        if self._values is not None and not any(isinstance(v, value) for v in self._values):
            return None
        return self._array

    def _packed_array(self) -> 'array[Any] | None':
        """Array holding all elements if the elements are only available
        as array, otherwise None"""
        if self._values is None:
            return self._array
        return None

    def _array_op(self: TArrayType, other: Any, op: str, reverse: bool = False) -> TArrayType | None:
        """Element-wise operation with array stencils, None if not applicable.
        The result has the class of self."""
        if self._is_constant() and (isinstance(other, int | float) or (isinstance(other, ArrayType) and other._is_constant())):
            return None  # Constants are evaluated at trace time
        a = self._get_array()
        if isinstance(other, ArrayType):
            b = other._get_array()
            if a is None and b is None:
                return None
            if a is None:
                a = self._get_array(force=True)
            if b is None:
                b = other._get_array(force=True)
            if a is None or b is None:
                return None
            return type(self)._from_array(a._binary_op(op, b, reverse))
        if a is None or isinstance(other, bool) or not isinstance(other, value | int | float) or \
           (isinstance(other, value) and other.dtype == 'bool'):
            return None
        if not reverse and isinstance(other, int | float) and \
           ((op in ('add', 'sub') and other == 0) or (op == 'mul' and other == 1)):
            return self
        return type(self)._from_array(a._binary_op(op, other, reverse))
