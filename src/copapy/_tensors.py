from copapy._basic_types import NumLike, ArrayType
from . import value
from ._vectors import vector
from ._mixed import mixed_sum
from typing import TypeVar, Any, overload, TypeAlias, Callable, Iterator, Sequence
from ._helper_types import TNum

TensorNumLike: TypeAlias = 'tensor[Any] | vector[Any] | value[Any] | int | float | bool'
TensorIntLike: TypeAlias = 'tensor[int] | value[int] | int'
TensorFloatLike: TypeAlias = 'tensor[float] | value[float] | float'
TensorSequence: TypeAlias = 'Sequence[TNum | value[TNum]] | Sequence[Sequence[TNum | value[TNum]]] | Sequence[Sequence[Sequence[TNum | value[TNum]]]]'
U = TypeVar("U", int, float)


class tensor(ArrayType[TNum]):
    """Generalized n-dimensional tensor class supporting numpy-style operations.

    A tensor can have any number of dimensions and supports element-wise operations,
    reshaping, transposition, and various reduction operations.
    """

    def __init__(self, values: 'TNum | value[TNum] | vector[TNum] | tensor[TNum] | TensorSequence[TNum]', shape: Sequence[int] | None = None):
        """Create a tensor with given values.

        Arguments:
            values: Nested iterables of constant values or copapy values.
                    Can be a scalar, 1D iterable (vector),
                    or n-dimensional nested structure.
        """
        if shape:
            self.shape: tuple[int, ...] = tuple(shape)
            assert (isinstance(values, Sequence) and
                    any(isinstance(v, (value, int, float)) for v in values)), \
                    "Values must be a sequence of values if shape is provided"
            self.values: tuple[TNum | value[TNum], ...] = tuple(v for v in values if not isinstance(v, Sequence))
            self.ndim: int = len(shape)
        elif isinstance(values, (int, float)):
            # Scalar case: 0-dimensional tensor
            self.shape = ()
            self.values = (values,)
            self.ndim = 0
        elif isinstance(values, value):
            # Scalar value case
            self.shape = ()
            self.values = (values,)
            self.ndim = 0
        elif isinstance(values, vector):
            # 1D case from vector
            self.shape = (len(values),)
            self.values = values.values
            self.ndim = 1
        elif isinstance(values, tensor):
            # Copy constructor
            self.shape = values.shape
            self.values = values.values
            self.ndim = values.ndim
        else:
            # General n-dimensional case
            self.values, self.shape = self._infer_shape_and_flatten(values)
            self.ndim = len(self.shape)

    @staticmethod
    def _infer_shape_and_flatten(values: Sequence[Any]) -> tuple[tuple[Any, ...], tuple[int, ...]]:
        """Infer the shape of a nested iterable and validate consistency."""
        def get_shape(val: int | float | value[Any] | Sequence[Any]) -> list[int]:
            if isinstance(val, int | float):
                return []
            if isinstance(val, value):
                return []
            else:
                if not val:
                    return [0]
                sub_shape = get_shape(val[0])
                if any(get_shape(item) != sub_shape for item in val[1:]):
                    raise ValueError("All elements must have consistent shape")
                return [len(val)] + sub_shape
            return []

        shape = tuple(get_shape(values))
        if not shape:
            # Scalar
            return (values,), ()

        # Flatten nested structure
        def flatten_recursive(val: Any) -> list[Any]:
            if isinstance(val, int | float | value):
                return [val]
            else:
                result: list[value[Any]] = []
                for item in val:
                    if isinstance(item, int | float | value | Sequence):
                        result.extend(flatten_recursive(item))
                return result

        flattened = flatten_recursive(values)
        return tuple(flattened), shape

    def _get_flat_index(self, indices: Sequence[int]) -> int:
        """Convert multi-dimensional indices to flat index."""
        if len(indices) != len(self.shape):
            raise IndexError(f"Expected {len(self.shape)} indices, got {len(indices)}")

        flat_idx = 0
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if not (0 <= indices[i] < self.shape[i]):
                raise IndexError(f"Index {indices[i]} out of bounds for dimension {i} with size {self.shape[i]}")
            flat_idx += indices[i] * stride
            stride *= self.shape[i]
        return flat_idx

    def _get_indices_from_flat(self, flat_idx: int) -> tuple[int, ...]:
        """Convert flat index to multi-dimensional indices."""
        indices: list[int] = []
        for dim_size in reversed(self.shape):
            indices.append(flat_idx % dim_size)
            flat_idx //= dim_size
        return tuple(reversed(indices))

    def __repr__(self) -> str:
        return f"tensor(shape={self.shape}, values={self.values if self.ndim == 0 else '...'})"

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def get_scalar(self: 'tensor[TNum]', *key: int) -> TNum | value[TNum]:
        """Get a single scalar value from the tensor given multi-dimensional indices."""
        assert len(key) == self.ndim, f"Expected {self.ndim} indices, got {len(key)}"
        flat_idx = self._get_flat_index(key)
        return self.values[flat_idx]

    def __getitem__(self, key: int | slice | Sequence[int | slice]) -> 'tensor[TNum]':
        """Get a sub-tensor or element.

        Arguments:
            key: Integer index (returns tensor of rank n-1),
                 slice object (returns tensor with same rank),
                 tuple of indices/slices (returns sub-tensor or element),
                 or tuple of indices (returns single element).

        Returns:
            Sub-tensor or element value.
        """
        if self.ndim == 0:
            raise TypeError("Cannot index a 0-d tensor")

        # Handle single slice
        if isinstance(key, slice):
            return self._handle_slice((key,))

        # Handle tuple of indices/slices
        if isinstance(key, Sequence):
            return self._handle_slice(key)

        # Handle single integer index
        assert isinstance(key, int), f"indices must be integers, slices, or tuples thereof, not {type(key)}"
        # Return a sub-tensor of rank n-1
        if not (-self.shape[0] <= key < self.shape[0]):
            raise IndexError(f"Index {key} out of bounds for dimension 0 with size {self.shape[0]}")

        if key < 0:
            key += self.shape[0]

        # Calculate which elements belong to this slice
        sub_shape = self.shape[1:]
        sub_size = 1
        for s in sub_shape:
            sub_size *= s

        start_idx = key * sub_size
        end_idx = start_idx + sub_size

        sub_values = self.values[start_idx:end_idx]

        if not sub_shape:
            #assert False, (sub_shape, len(sub_shape), sub_values[0])
            return tensor(sub_values[0])

        return tensor(sub_values, sub_shape)

    def _handle_slice(self, keys: Sequence[int | slice]) -> 'tensor[TNum]':
        """Handle slicing operations on the tensor."""
        # Process all keys and identify ranges for each dimension
        ranges: list[range] = []

        for i, key in enumerate(keys):
            if i >= self.ndim:
                raise IndexError(f"Too many indices for tensor of rank {self.ndim}")

            dim_size = self.shape[i]

            if isinstance(key, int):
                if not (-dim_size <= key < dim_size):
                    raise IndexError(f"Index {key} out of bounds for dimension {i} with size {dim_size}")
                if key < 0:
                    key += dim_size
                ranges.append(range(key, key + 1))
            else:
                assert isinstance(key, slice), f"indices must be integers or slices, not {type(key)}"
                start, stop, step = key.indices(dim_size)
                ranges.append(range(start, stop, step))

        # Handle remaining dimensions (full ranges)
        for i in range(len(keys), self.ndim):
            ranges.append(range(self.shape[i]))

        # Collect elements matching the ranges
        selected_values: list[TNum | value[TNum]] = []
        new_shape: list[int] = []

        # Calculate new shape (only include dimensions that weren't single integers)
        for i, key in enumerate(keys):
            if not isinstance(key, int):
                new_shape.append(len(ranges[i]))

        # Add remaining dimensions
        for i in range(len(keys), self.ndim):
            new_shape.append(self.shape[i])

        # Iterate through all combinations of indices in the ranges
        def iterate_ranges(range_list: list[range], current_indices: list[int]) -> None:
            if len(current_indices) == len(range_list):
                # Compute flat index
                flat_idx = 0
                stride = 1
                for i in range(len(self.shape) - 1, -1, -1):
                    flat_idx += current_indices[i] * stride
                    stride *= self.shape[i]
                selected_values.append(self.values[flat_idx])
            else:
                dim = len(current_indices)
                for idx in range_list[dim]:
                    iterate_ranges(range_list, current_indices + [idx])

        iterate_ranges(ranges, [])

        # Return based on result shape
        if not new_shape:
            # Single element (all were integers)
            return tensor(selected_values[0])

        return tensor(tuple(selected_values), tuple(new_shape))

    def __iter__(self) -> Iterator['tensor[TNum]']:
        """Iterate over the first dimension."""
        if self.ndim == 0:
            raise TypeError("Cannot iterate over a 0-d tensor")

        for i in range(self.shape[0]):
            yield self[i]

    def __neg__(self) -> 'tensor[TNum]':
        """Negate all elements."""
        negated_values: tuple[Any, ...] = tuple(-v for v in self.values)
        return tensor(negated_values, self.shape)

    @overload
    def __add__(self: 'tensor[int]', other: TensorFloatLike) -> 'tensor[float]': ...
    @overload
    def __add__(self: 'tensor[int]', other: TensorIntLike) -> 'tensor[int]': ...
    @overload
    def __add__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __add__(self, other: TensorNumLike) -> 'tensor[int] | tensor[float]': ...
    def __add__(self, other: TensorNumLike) -> Any:
        """Element-wise addition."""
        return self._binary_op(other, lambda a, b: a + b)

    @overload
    def __radd__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __radd__(self: 'tensor[int]', other: value[int] | int) -> 'tensor[int]': ...
    def __radd__(self, other: Any) -> Any:
        return self + other

    @overload
    def __sub__(self: 'tensor[int]', other: TensorFloatLike) -> 'tensor[float]': ...
    @overload
    def __sub__(self: 'tensor[int]', other: TensorIntLike) -> 'tensor[int]': ...
    @overload
    def __sub__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __sub__(self, other: TensorNumLike) -> 'tensor[int] | tensor[float]': ...
    def __sub__(self, other: TensorNumLike) -> Any:
        """Element-wise subtraction."""
        return self._binary_op(other, lambda a, b: a - b, commutative=False)

    @overload
    def __rsub__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __rsub__(self: 'tensor[int]', other: value[int] | int) -> 'tensor[int]': ...
    def __rsub__(self, other: TensorNumLike) -> Any:
        return self._binary_op(other, lambda a, b: b - a, commutative=False, reversed=True)

    @overload
    def __mul__(self: 'tensor[int]', other: TensorFloatLike) -> 'tensor[float]': ...
    @overload
    def __mul__(self: 'tensor[int]', other: TensorIntLike) -> 'tensor[int]': ...
    @overload
    def __mul__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __mul__(self, other: TensorNumLike) -> 'tensor[int] | tensor[float]': ...
    def __mul__(self, other: TensorNumLike) -> Any:
        """Element-wise multiplication."""
        return self._binary_op(other, lambda a, b: a * b)

    @overload
    def __rmul__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __rmul__(self: 'tensor[int]', other: value[int] | int) -> 'tensor[int]': ...
    def __rmul__(self, other: TensorNumLike) -> Any:
        return self * other

    def __truediv__(self, other: TensorNumLike) -> 'tensor[float]':
        """Element-wise division."""
        return self._binary_op(other, lambda a, b: a / b, commutative=False)

    def __rtruediv__(self, other: TensorNumLike) -> 'tensor[float]':
        """Element-wise right division."""
        return self._binary_op(other, lambda a, b: b / a, commutative=False, reversed=True)

    @overload
    def __pow__(self: 'tensor[int]', other: TensorFloatLike) -> 'tensor[float]': ...
    @overload
    def __pow__(self: 'tensor[int]', other: TensorIntLike) -> 'tensor[int]': ...
    @overload
    def __pow__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __pow__(self, other: TensorNumLike) -> 'tensor[int] | tensor[float]': ...
    def __pow__(self, other: TensorNumLike) -> Any:
        """Element-wise power."""
        return self._binary_op(other, lambda a, b: a ** b, commutative=False)

    @overload
    def __rpow__(self: 'tensor[float]', other: TensorNumLike) -> 'tensor[float]': ...
    @overload
    def __rpow__(self: 'tensor[int]', other: value[int] | int) -> 'tensor[int]': ...
    def __rpow__(self, other: TensorNumLike) -> Any:
        return self._binary_op(other, lambda a, b: b ** a, commutative=False, reversed=True)

    def __gt__(self, other: TensorNumLike) -> 'tensor[int]':
        """Element-wise greater than."""
        return self._binary_op(other, lambda a, b: a > b, commutative=False)

    def __lt__(self, other: TensorNumLike) -> 'tensor[int]':
        """Element-wise less than."""
        return self._binary_op(other, lambda a, b: a < b, commutative=False)

    def __ge__(self, other: TensorNumLike) -> 'tensor[int]':
        """Element-wise greater than or equal."""
        return self._binary_op(other, lambda a, b: a >= b, commutative=False)

    def __le__(self, other: TensorNumLike) -> 'tensor[int]':
        """Element-wise less than or equal."""
        return self._binary_op(other, lambda a, b: a <= b, commutative=False)

    def __eq__(self, other: TensorNumLike) -> 'tensor[int]':  # type: ignore
        """Element-wise equality."""
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other: TensorNumLike) -> 'tensor[int]':  # type: ignore
        """Element-wise inequality."""
        return self._binary_op(other, lambda a, b: a != b)

    def _binary_op(self, other: TensorNumLike, op: Callable[[Any, Any], 'tensor[TNum]'],
                   commutative: bool = True, reversed: bool = False) -> 'tensor[Any]':
        """Perform binary operation with broadcasting support.
        """
        seen_consts: dict[NumLike, NumLike] = {}

        def call_op(a: TNum | value[TNum], b: NumLike) -> Any:
            if isinstance(b, value) or not isinstance(a, value):
                b_trans = b
            else:
                if b in seen_consts:
                    b_trans = seen_consts[b]
                else:
                    b_trans = value(b)
                    seen_consts[b] = b_trans
            if reversed:
                return op(b_trans, a)
            else:
                return op(a, b_trans)

        if isinstance(other, Sequence | vector):
            other_tensor: tensor[Any] = tensor(other)
            return self._binary_op(other_tensor, op, commutative, reversed)

        elif isinstance(other, tensor):
            self_shape = self.shape
            other_shape = other.shape

            # Check if shapes are identical
            if self_shape == other_shape:
                result_vals = tuple(call_op(a, b) for a, b in zip(self.values, other.values))
                return tensor(result_vals, self_shape)

            # Broadcast shapes using numpy-style broadcasting rules
            result_shape = self._broadcast_shapes(self_shape, other_shape)

            # Expand both tensors to the broadcast shape
            self_expanded = self._expand_to_shape(result_shape)
            other_expanded = other._expand_to_shape(result_shape)

            # Apply operation element-wise
            result_vals = tuple(call_op(a, b) for a, b in zip(self_expanded.values, other_expanded.values))
            return tensor(result_vals, result_shape)

        else:
            # Broadcast scalar
            result_vals = tuple(call_op(v, other) for v in self.values)
            return tensor(result_vals, self.shape)

    def _broadcast_shapes(self, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the broadcast shape of two shapes following numpy rules.

        Rules:
        - Dimensions are compared from right to left
        - Dimensions must either be equal or one must be 1
        - Missing dimensions are treated as 1
        """
        # Align from the right
        max_ndim = max(len(shape1), len(shape2))

        # Pad with 1s on the left
        padded_shape1 = (1,) * (max_ndim - len(shape1)) + shape1
        padded_shape2 = (1,) * (max_ndim - len(shape2)) + shape2

        result_shape: list[int] = []
        for dim1, dim2 in zip(padded_shape1, padded_shape2):
            if dim1 == dim2:
                result_shape.append(dim1)
            elif dim1 == 1:
                result_shape.append(dim2)
            elif dim2 == 1:
                result_shape.append(dim1)
            else:
                raise ValueError(f"Incompatible shapes for broadcasting: {shape1} vs {shape2}")

        return tuple(result_shape)

    def _expand_to_shape(self, target_shape: tuple[int, ...]) -> 'tensor[TNum]':
        """Expand tensor to target shape using broadcasting (repeating dimensions of size 1).
        """
        if self.shape == target_shape:
            return self

        # Pad self.shape with 1s on the left to match target_shape length
        padded_self_shape = (1,) * (len(target_shape) - len(self.shape)) + self.shape

        # Validate broadcasting is possible
        for s, t in zip(padded_self_shape, target_shape):
            if s != t and s != 1:
                raise ValueError(f"Cannot broadcast shape {self.shape} to {target_shape}")

        # Expand step by step from left to right
        current_tensor = self
        current_shape = self.shape

        # Add missing dimensions on the left
        if len(self.shape) < len(target_shape):
            diff = len(target_shape) - len(self.shape)
            for _ in range(diff):
                # Reshape to add dimension of size 1 on the left
                current_tensor = current_tensor.reshape(1, *current_tensor.shape)
                current_shape = (1,) + current_shape

        # Expand each dimension that is 1 to the target size
        for i, (curr_dim, target_dim) in enumerate(zip(current_shape, target_shape)):
            if curr_dim == 1 and target_dim != 1:
                current_tensor = current_tensor._repeat_along_axis(i, target_dim)
                current_shape = current_tensor.shape

        return current_tensor

    def _repeat_along_axis(self, axis: int, repetitions: int) -> 'tensor[TNum]':
        """Repeat tensor along a specific axis.
        """
        if self.shape[axis] != 1:
            raise ValueError(f"Can only repeat dimensions of size 1, got {self.shape[axis]}")

        # Create list of indices to select (repeat the single index)
        new_shape = list(self.shape)
        new_shape[axis] = repetitions
        new_values: list[TNum | value[TNum]] = []

        # Iterate through all positions in the new tensor
        def iterate_and_repeat(current_indices: list[int], depth: int) -> None:
            if depth == len(new_shape):
                # Get the source index (with 0 for the repeated dimension)
                source_indices = list(current_indices)
                source_indices[axis] = 0
                source_idx = self._get_flat_index(tuple(source_indices))
                new_values.append(self.values[source_idx])
            else:
                for i in range(new_shape[depth]):
                    iterate_and_repeat(current_indices + [i], depth + 1)

        iterate_and_repeat([], 0)
        return tensor(tuple(new_values), tuple(new_shape))

    def reshape(self, *new_shape: int) -> 'tensor[TNum]':
        """Reshape the tensor to a new shape.

        Arguments:
            *new_shape: New shape dimensions. Use -1 for one dimension to infer automatically.

        Returns:
            A new tensor with the specified shape.
        """
        shape_arg: tuple[int, ...] | int = new_shape if len(new_shape) != 1 else new_shape[0]
        if isinstance(shape_arg, int):
            new_shape = (shape_arg,)
        else:
            new_shape = shape_arg

        # Handle -1 in shape (automatic dimension inference)
        neg_one_count = sum(1 for d in new_shape if d == -1)
        if neg_one_count > 1:
            raise ValueError("Only one dimension can be -1")

        if neg_one_count == 1:
            known_size = 1
            for dim in new_shape:
                if dim != -1:
                    known_size *= dim

            if self.size() % known_size != 0:
                raise ValueError(f"Cannot infer dimension from size {self.size()} with shape {new_shape}")

            inferred_dim = self.size() // known_size
            new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)

        total_size = 1
        for dim in new_shape:
            total_size *= dim

        if total_size != self.size():
            raise ValueError(f"Cannot reshape tensor of size {self.size()} into shape {new_shape}")

        return tensor(self.values, new_shape)

    @overload
    def trace(self: 'tensor[TNum]') -> TNum | value[TNum]: ...
    @overload
    def trace(self: 'tensor[int]') -> int | value[int]: ...
    @overload
    def trace(self: 'tensor[float]') -> float | value[float]: ...
    def trace(self) -> Any:
        """Calculate the trace (sum of diagonal elements)."""
        assert self.ndim == 2 and self.shape[0] == self.shape[1], "Trace is only defined for square matrices"
        return mixed_sum(self.get_scalar(i, i) for i in range(self.shape[0]))

    def transpose(self, *axes: int) -> 'tensor[TNum]':
        """Transpose the tensor.

        Arguments:
            *axes: Permutation of axes. If not provided, reverses all axes.

        Returns:
            A transposed tensor.
        """
        if not axes:
            axes = tuple(range(self.ndim - 1, -1, -1))

        if len(axes) != self.ndim:
            raise ValueError("axes don't match tensor")

        if any(not (0 <= ax < self.ndim) for ax in axes):
            raise ValueError(f"Invalid axes for tensor of rank {self.ndim}")

        new_shape = tuple(self.shape[ax] for ax in axes)
        new_values: list[Any] = [None] * len(self.values)

        for old_idx in range(len(self.values)):
            old_indices = self._get_indices_from_flat(old_idx)
            new_indices = tuple(old_indices[ax] for ax in axes)

            new_flat_idx = 0
            stride = 1
            for i in range(len(new_shape) - 1, -1, -1):
                new_flat_idx += new_indices[i] * stride
                stride *= new_shape[i]

            new_values[new_flat_idx] = self.values[old_idx]

        return tensor(new_values, new_shape)

    def flatten(self) -> 'tensor[TNum]':
        """Flatten the tensor to 1D.

        Returns:
            A flattened 1D tensor.
        """
        return self.reshape(-1)

    def size(self) -> int:
        """Return total number of elements."""
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def matmul(self, other: 'tensor[TNum] | vector[TNum]') -> 'TNum | value[TNum] | tensor[TNum]':
        """Matrix multiplication (@ operator).

        Arguments:
            other: Another tensor to multiply with.

        Returns:
            Result of matrix multiplication.

        Raises:
            ValueError: If shapes are incompatible for matrix multiplication.
        """
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("matmul requires tensors with at least 1 dimension")

        # For 1D x 1D: dot product (returns scalar)
        if self.ndim == 1 and other.ndim == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shape mismatch: ({self.shape[0]},) @ ({other.shape[0]},)")
            result = mixed_sum(a * b for a, b in zip(self.values, other.values))
            return result

        # For 2D x 2D: standard matrix multiplication
        if self.ndim == 2 and other.ndim == 2 and isinstance(other, tensor):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")

            result_values: list[Any] = []
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    dot_sum = sum(self.get_scalar(i, k) * other.get_scalar(k, j) for k in range(self.shape[1]))
                    result_values.append(dot_sum)

            return tensor(tuple(result_values), (self.shape[0], other.shape[1]))

        # For 1D x 2D: treat 1D as row vector
        if self.ndim == 1 and other.ndim == 2 and isinstance(other, tensor):
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shape mismatch: ({self.shape[0]},) @ {other.shape}")

            result_values = []
            for j in range(other.shape[1]):
                dot_sum = sum(self.get_scalar(k) * other.get_scalar(k, j) for k in range(self.shape[0]))
                result_values.append(dot_sum)

            return tensor(tuple(result_values), (other.shape[1],))

        # For 2D x 1D: treat 1D as column vector
        if self.ndim == 2 and other.ndim == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shape mismatch: {self.shape} @ ({other.shape[0]},)")

            result_values = []

            if isinstance(other, vector):
                for i in range(self.shape[0]):
                    dot_sum = value(0)
                    for k in range(self.shape[1]):
                        dot_sum = dot_sum + self.get_scalar(i, k) * other.get_scalar(k)
                    result_values.append(dot_sum)
            else:
                for i in range(self.shape[0]):
                    dot_sum = value(0)
                    for k in range(self.shape[1]):
                        dot_sum = dot_sum + self.get_scalar(i, k) * other.get_scalar(k)
                    result_values.append(dot_sum)

            return tensor(tuple(result_values), (self.shape[0],))

        raise NotImplementedError(f"matmul not implemented for shapes {self.ndim}D @ {other.ndim}D")

    def __matmul__(self, other: 'tensor[TNum] | vector[TNum]') -> 'TNum | value[TNum] | tensor[TNum]':
        """Matrix multiplication operator (@)."""
        return self.matmul(other)

    def __rmatmul__(self, other: 'tensor[TNum] | vector[TNum]') -> 'TNum | value[TNum] | tensor[TNum]':
        """Right matrix multiplication operator."""
        if isinstance(other, tensor):
            return other.matmul(self)
        return NotImplemented

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> TNum | value[TNum] | 'tensor[TNum]':
        """Sum all or along specified axis/axes.

        Arguments:
            axis: Axis or tuple of axes along which to sum. If None, sums all elements.
            keepdims: If True, keep reduced dimensions as size 1.

        Returns:
            Scalar or tensor with reduced dimension(s).
        """
        if axis is None:
            result = mixed_sum(self.values)
            if keepdims:
                # Return tensor with all dimensions set to 1
                new_shape = [1 for _ in self.shape]
                return tensor((result,), new_shape)
            return result

        # Handle single axis (convert to tuple for uniform processing)
        if isinstance(axis, int):
            axes: tuple[int, ...] = (axis,)
        else:
            axes = tuple(axis)

        # Validate and normalize axes
        normalized_axes: list[int] = []
        for ax in axes:
            if not (0 <= ax < self.ndim):
                raise ValueError(f"Axis {ax} is out of bounds for tensor of rank {self.ndim}")
            if ax not in normalized_axes:
                normalized_axes.append(ax)

        # Sort axes in descending order for easier dimension removal
        normalized_axes = sorted(set(normalized_axes), reverse=True)

        # Sum along specified axes
        new_shape = list(self.shape)
        for ax in normalized_axes:
            new_shape.pop(ax)

        if not new_shape:
            # All axes summed - return scalar
            return mixed_sum(self.values)

        new_size = 1
        for dim in new_shape:
            new_size *= dim

        new_values: list[TNum | value[TNum]] = [self.values[0]] * new_size
        new_v_mask: list[bool] = [False] * new_size

        for old_idx in range(len(self.values)):
            old_indices = list(self._get_indices_from_flat(old_idx))

            # Build new indices by removing summed axes
            new_indices: list[int] = []
            for i, idx in enumerate(old_indices):
                if i not in normalized_axes:
                    new_indices.append(idx)

            # Compute flat index in new shape
            new_flat_idx = 0
            stride = 1
            for i in range(len(new_shape) - 1, -1, -1):
                new_flat_idx += new_indices[i] * stride
                stride *= new_shape[i]

            if new_v_mask[new_flat_idx]:
                new_values[new_flat_idx] = new_values[new_flat_idx] + self.values[old_idx]
            else:
                new_values[new_flat_idx] = self.values[old_idx]
                new_v_mask[new_flat_idx] = True

        if keepdims:
            # Restore reduced dimensions as size 1
            full_shape = list(self.shape)
            for ax in normalized_axes:
                full_shape[ax] = 1
            return tensor(new_values, tuple(full_shape))

        if not new_shape:
            return new_values[0]
        return tensor(new_values, tuple(new_shape))

    def mean(self, axis: int | None = None) -> Any:
        """Calculate mean along axis or overall.

        Arguments:
            axis: Axis along which to compute mean. If None, computes overall mean.

        Returns:
            Scalar or tensor with reduced dimension.
        """
        if axis is None:
            total_sum: Any = mixed_sum(self.values)
            return total_sum / self.size()

        sum_result: Any = self.sum(axis)
        axis_size = self.shape[axis]

        if isinstance(sum_result, tensor):
            return sum_result / axis_size
        else:
            return sum_result / axis_size

    def map(self, func: Callable[[Any], value[U] | U]) -> 'tensor[U]':
        """Apply a function to each element.

        Arguments:
            func: Function to apply to each element.

        Returns:
            A new tensor with the function applied element-wise.
        """
        result_vals = tuple(func(v) for v in self.values)
        return tensor(result_vals, self.shape)

    def homogenize(self) -> 'tensor[TNum]':
        """Convert all elements to copapy values if any element is a copapy value."""
        if any(isinstance(val, value) for val in self.values):
            homogenized: tuple[value[Any], ...] = tuple(value(val) if not isinstance(val, value) else val for val in self.values)
            return tensor(homogenized, self.shape)
        return self

    @property
    def T(self) -> 'tensor[TNum]':
        """Transpose all axes (equivalent to transpose() with no args)."""
        return self.transpose()


def zeros(shape: Sequence[int] | int) -> tensor[int]:
    """Create a zero tensor of given shape."""
    if isinstance(shape, int):
        shape = (shape,)

    size = 1
    for dim in shape:
        size *= dim

    return tensor([0] * size, tuple(shape))


def ones(shape: Sequence[int] | int) -> tensor[int]:
    """Create a tensor of ones with given shape."""
    if isinstance(shape, int):
        shape = (shape,)

    size = 1
    for dim in shape:
        size *= dim

    return tensor([1] * size, tuple(shape))


def arange(start: int | float, stop: int | float | None = None,
           step: int | float = 1) -> tensor[int] | tensor[float]:
    """Create a tensor with evenly spaced values.

    Arguments:
        start: Start value (or stop if stop is None).
        stop: Stop value (exclusive).
        step: Step between values.

    Returns:
        A 1D tensor.
    """
    if stop is None:
        stop = start
        start = 0

    # Determine type
    values_list: list[value[Any]] = []
    current = start
    if step > 0:
        while current < stop:
            values_list.append(value(current))
            current += step
    elif step < 0:
        while current > stop:
            values_list.append(value(current))
            current += step
    else:
        raise ValueError("step cannot be zero")

    return tensor(tuple(values_list), (len(values_list),))


def eye(rows: int, cols: int | None = None) -> tensor[int]:
    """Create an identity tensor with ones on diagonal.

    Arguments:
        rows: Number of rows.
        cols: Number of columns (defaults to rows).

    Returns:
        A 2D identity tensor.
    """
    if cols is None:
        cols = rows

    values_list: list[value[int]] = []

    for i in range(rows):
        for j in range(cols):
            if i == j:
                values_list.append(value(1))
            else:
                values_list.append(value(0))

    return tensor(tuple(values_list), (rows, cols))


def identity(size: int) -> tensor[int]:
    """Create a square identity tensor.

    Arguments:
        size: Size of the square tensor.

    Returns:
        A square 2D identity tensor.
    """
    return eye(size, size)


@overload
def diagonal(vec: 'tensor[int] | vector[int]') -> tensor[int]: ...
@overload
def diagonal(vec: 'tensor[float] | vector[float]') -> tensor[float]: ...
def diagonal(vec: 'tensor[Any] | vector[Any]') -> 'tensor[Any]':
    """Create a diagonal tensor from a 1D tensor.

    Arguments:
        vec: A 1D tensor with values to place on the diagonal.

    Returns:
        A 2D tensor with the input values on the diagonal and zeros elsewhere.
    """
    if vec.ndim != 1:
        raise ValueError(f"Input must be 1D, got {vec.ndim}D")

    size = len(vec)
    values_list: list[Any] = []

    for i in range(size):
        for j in range(size):
            if i == j:
                values_list.append(vec[i])
            else:
                values_list.append(value(0))

    return tensor(tuple(values_list), (size, size))
