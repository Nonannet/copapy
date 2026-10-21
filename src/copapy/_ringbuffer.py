"""Ring buffer of the last n values of a signal. The buffer is state of the
task computing it (like delay): each run writes the current value in place
to one slot of the buffer, there is no copy of the buffer for the update.
The slot of the current value (index) increases by one each run and wraps
around at n.
"""
from typing import Any, Generic, Sequence, TypeVar, cast, overload
from ._basic_types import value, Net, ArrayDelay, ArrayNet, transl_type, value_from_number, to_float
from ._arrays import array, ArrayType, _add_array_op, _flatten
from ._vectors import vector
from ._tensors import tensor
from ._quaternion import quaternion
from ._math import minimum, maximum
from ._flow import iif
from ._temporal import feedback
from ._helper_types import TNum, U

TSample = TypeVar("TSample", bound='value[Any] | ArrayType[Any] | array[Any]')


def _sample_array(x: Any) -> array[Any] | None:
    """Elements of a vector, tensor or array sample as array, None for scalars"""
    if isinstance(x, ArrayType):
        return x._force_array()
    if isinstance(x, array):
        return x
    if isinstance(x, value | int | float):
        return None
    raise TypeError(f"Unsupported type {type(x).__name__}, use a value, vector, tensor or array")


def _init_elements(init: Any, size: int, dtype: str) -> list[int | float]:
    """Initial elements of one sample: one number for all elements or a
    vector, tensor, array or (nested) sequence of numbers of the sample size"""
    if isinstance(init, value) or (isinstance(init, ArrayType) and not init._is_constant()):
        raise TypeError("Initial values must be numbers, not copapy values")
    if isinstance(init, int | float):
        elements: list[Any] = [init] * size
    elif isinstance(init, ArrayType):
        elements = list(init.values)
    elif isinstance(init, array):
        raise TypeError("The initial value must be a number, vector, tensor or sequence, not an array")
    else:
        elements = _flatten(init)[0]
    if len(elements) != size:
        raise ValueError(f"{len(elements)} initial values for a sample of {size} elements")
    if dtype == 'float':
        return [float(e) for e in elements]
    if any(isinstance(e, float) and not e.is_integer() for e in elements):
        raise TypeError(f"Initial values of an {dtype} ring buffer must be integers")
    return [int(e) for e in elements]


class ringbuffer(array[TNum], Generic[TNum, TSample]):
    """Ring buffer holding the values of x at the last n runs of the task
    computing it, the current value included. At each run the current value
    is written in place to one slot of the buffer (no copy of the buffer), so
    the cost per run is independent of n. Before n runs the remaining slots
    hold the initial value.

    The ring buffer is an array of shape (n, *shape of x) in time order: the
    oldest value first and the current value last. Using it as array copies
    the buffer once per run into time order. Reductions that are independent
    of the order (sum, min, max and sort, and sums along the time axis 0)
    are computed from the slots without copy.

        >>> rb = cp.ringbuffer(x, 1000)
        >>> fir = (rb * taps).sum()     # Oldest value first
        >>> mean = cp.tensor(rb).mean() # No copy
        >>> delayed = rb.delay(250)     # x 250 runs ago

    The buffer is computed by the task computing x. To use it in another task,
    slots and index must be outputs of the task computing it (like for delay).
    Values read from the buffer are constants for the gradient, only delay(0)
    (x itself) is differentiable.

    Attributes:
        n: Number of values in the buffer
        index: Slot of the current value in slots (0 to n - 1)
        slots: The memory of the buffer as array of shape (n, *shape of x) in
            the order of the slots, reading it costs no copy.
    """
    @overload
    def __init__(self: 'ringbuffer[int, value[int]]', x: 'value[int] | int', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[float, value[float]]', x: 'value[float] | float', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[float, quaternion]', x: quaternion, n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[U, vector[U]]', x: 'vector[U]', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[U, tensor[U]]', x: 'tensor[U]', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[U, array[U]]', x: 'array[U]', n: int, init: Any = 0) -> None: ...
    def __init__(self, x: Any, n: int, init: Any = 0) -> None:
        """Ring buffer of the last n values of x.

        Arguments:
            x: Value, vector, tensor, quaternion or array of the current run.
                Python numbers are converted to constant values.
            n: Number of values in the buffer (at least 1)
            init: Value of the slots before they are written: a number for
                all elements or the shape of x
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, not {n}")
        if isinstance(x, int | float):
            x = value_from_number(x)
        sample = _sample_array(x)
        net: Net
        shape: tuple[int, ...]
        if sample is not None:
            if sample.dtype == 'byte':
                raise TypeError("Ring buffers of byte arrays are not supported, convert them to int or float")
            shape, dtype, net = sample.shape, sample.dtype, sample.net
        else:
            assert isinstance(x, value)
            shape, dtype, net = (), x.dtype, x.net
        stored = transl_type(dtype)  # Bool is stored as int
        if isinstance(init, float) and stored != 'float' and not init.is_integer():
            raise TypeError(f"Initial value {init} of an {dtype} ring buffer is no integer")
        if stored == 'float' and isinstance(x, value) and x.dtype != 'float':
            net = to_float(x).net

        size = sample.size if sample is not None else 1
        self.n: int = n
        self._sample: Any = x
        self._sample_shape = shape
        self._oldest: TSample | None = None

        state = ArrayDelay(_init_elements(init, size, stored) * n, stored)
        state_net = ArrayNet(stored, state, n * size)

        # Slot of the current value: 0 at the first run, wraps around at n
        self.index: value[int] = feedback(lambda prev: iif(prev >= n - 1, 0, prev + 1), init=n - 1)

        put = _add_array_op('put32_arr', [state_net, net, self.index.net, value_from_number(size).net],
                            n * size, stored, inplace=True)
        assert isinstance(put.result, ArrayNet)
        state.input = put.result
        self.slots: array[TNum] = array._from_net(put.result, (n, *shape), dtype)

        # Time order: the buffer rotated to start at the slot of the oldest value.
        # The copy is only compiled if the ring buffer itself is used as array.
        start = self._slot(n - 1)
        if size != 1:
            start = start * size
        roll = _add_array_op('roll32_arr', [put.result, start.net, value_from_number(n * size).net],
                             n * size, stored)
        assert isinstance(roll.result, ArrayNet)
        self._init(roll.result, (n, *shape), dtype)

    def __repr__(self) -> str:
        return f"ringbuffer(n={self.n}, shape={self._sample_shape}, dtype={self.dtype})"

    def _slot(self, k: 'int | value[int]') -> 'value[int]':
        """Slot of the value k runs ago"""
        slot = self.index - k
        return slot + iif(slot < 0, self.n, 0)

    def _wrap(self, arr: array[Any]) -> Any:
        """Sample as array in the type of x"""
        if isinstance(self._sample, quaternion | vector | tensor):
            return type(self._sample)._from_array(arr)
        return arr

    def _reduces_time(self, axis: 'int | Sequence[int]') -> bool:
        """True if the reduction along axis includes the time axis 0: its result
        is independent of the order of the slots"""
        return any(ax % self.ndim == 0 for ax in ((axis,) if isinstance(axis, int) else axis))

    @overload
    def sum(self, axis: None = None, keepdims: bool = False) -> value[TNum]: ...
    @overload
    def sum(self, axis: int | Sequence[int], keepdims: bool = False) -> 'value[TNum] | array[TNum]': ...
    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Any:
        """Sum of all elements or along the given axis or axes. Sums along
        the time axis 0 are computed from the slots without copy."""
        if axis is None:
            return self.slots.sum()
        if self._reduces_time(axis):
            return self.slots.sum(axis, keepdims)
        return super().sum(axis, keepdims)

    def _min_max(self, op: str) -> value[TNum]:
        # Smallest or largest element: independent of the order of the slots
        return self.slots._min_max(op)

    def sort(self) -> array[TNum]:
        """Sorted elements (1D only), computed from the slots without copy."""
        self._check_1d('sort')
        return self.slots.sort()

    def delay(self, k: 'int | value[int]') -> TSample:
        """Value of x k runs ago, read from its slot without a copy of the
        buffer: k = 0 is the current value, k = n - 1 the oldest value in the
        buffer. A value k is limited to 0 to n - 1.

        Arguments:
            k: Number of runs (int or int value)
        """
        if isinstance(k, value):
            if transl_type(k.dtype) != 'int':
                raise TypeError(f"k must be an int value, not {k.dtype}")
            k = minimum(maximum(k, 0), self.n - 1)
        elif not isinstance(k, int):
            raise TypeError(f"k must be an int or an int value, not {type(k).__name__}")
        elif not 0 <= k < self.n:
            raise IndexError(f"k must be from 0 to {self.n - 1}, not {k}")
        elif k == 0:
            return cast(TSample, self._sample)
        result: Any = self.slots[self._slot(k)]
        return cast(TSample, self._wrap(result) if self._sample_shape else result)

    @property
    def oldest(self) -> TSample:
        """Oldest value in the buffer (n - 1 runs ago), e.g. for a moving sum
        updated by the value entering and the value leaving the window."""
        if self._oldest is None:
            self._oldest = self.delay(self.n - 1)
        return self._oldest
