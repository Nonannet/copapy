"""Ring buffer of the last n values of a signal. The buffer is state of the
task computing it (like delay): each run writes the current value in place
to one slot of the buffer, there is no copy of the buffer for the update.
The slot of the current value (index) increases by one each run and wraps
around at n.
"""
from typing import Any, Generic, TypeVar, cast, overload
from ._basic_types import value, Net, ArrayDelay, ArrayNet, transl_type, value_from_number, to_float
from ._arrays import array, ArrayType, _add_array_op, _flatten
from ._vectors import vector
from ._tensors import tensor
from ._quaternion import quaternion
from ._math import minimum, maximum
from ._flow import iif
from ._temporal import feedback

T = TypeVar("T")


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


class ringbuffer(Generic[T]):
    """Ring buffer holding the values of x at the last n runs of the task
    computing it, the current value included. At each run the current value
    is written in place to one slot of the buffer (no copy of the buffer), so
    the cost per run is independent of n. Before n runs the remaining slots
    hold the initial value.

        >>> rb = cp.ringbuffer(x, 1000)
        >>> mean = cp.filters.mean(rb.data)  # Order of the slots is irrelevant
        >>> delayed = rb[250]                # x 250 runs ago
        >>> fir = (rb.window() * taps).sum() # Oldest value first

    The buffer is computed by the task computing x. To use it in another task,
    data and index must be outputs of the task computing it (like for delay).
    Values read from the buffer are constants for the gradient, only rb[0]
    (x itself) is differentiable.

    Attributes:
        n: Number of values in the buffer
        shape: Shape of one value: () for scalars
        index: Slot of the current value in data (0 to n - 1)
        data: The buffer as array of shape (n, *shape) in the order of the
            slots. It is the memory of the buffer, reading it costs no copy.
    """
    @overload
    def __init__(self: 'ringbuffer[value[float]]', x: 'value[float] | float', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[value[int]]', x: 'value[int] | int', n: int, init: Any = 0) -> None: ...
    @overload
    def __init__(self: 'ringbuffer[T]', x: T, n: int, init: Any = 0) -> None: ...
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
        self.n = n
        self.shape: tuple[int, ...] = shape
        self._sample = x
        self._size = size
        self._dtype = dtype
        self._stored = stored
        self._oldest: Any = None

        state = ArrayDelay(_init_elements(init, size, stored) * n, stored)
        state_net = ArrayNet(stored, state, n * size)

        # Slot of the current value: 0 at the first run, wraps around at n
        self.index: value[int] = feedback(lambda prev: iif(prev >= n - 1, 0, prev + 1), init=n - 1)

        put = _add_array_op('put32_arr', [state_net, net, self.index.net, value_from_number(size).net],
                            n * size, stored, inplace=True)
        assert isinstance(put.result, ArrayNet)
        state.input = put.result
        self.data: array[Any] = array._from_net(put.result, (n, *shape), dtype)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"ringbuffer(n={self.n}, shape={self.shape}, dtype={self._dtype})"

    def _slot(self, k: 'int | value[int]') -> 'value[int]':
        """Slot of the value k runs ago"""
        slot = self.index - k
        return slot + iif(slot < 0, self.n, 0)

    def _wrap(self, arr: array[Any]) -> Any:
        """Sample as array in the type of x"""
        if isinstance(self._sample, quaternion | vector | tensor):
            return type(self._sample)._from_array(arr)
        return arr

    def __getitem__(self, k: 'int | value[int]') -> T:
        """Value of x k runs ago: k = 0 is the current value, k = n - 1 the
        oldest value in the buffer. A value k is limited to 0 to n - 1.

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
            return cast(T, self._sample)
        result: Any = self.data[self._slot(k)]
        return cast(T, self._wrap(result) if self.shape else result)

    @property
    def oldest(self) -> T:
        """Oldest value in the buffer (n - 1 runs ago), e.g. for a moving sum
        updated by the value entering and the value leaving the window."""
        if self._oldest is None:
            self._oldest = self[self.n - 1]
        return cast(T, self._oldest)

    def window(self) -> array[Any]:
        """Values of the buffer in time order, the oldest value first and the
        current value last: a copy of the buffer (array of shape (n, *shape))."""
        start = self._slot(self.n - 1)
        if self._size != 1:
            start = start * self._size
        total = self.n * self._size
        node = _add_array_op('roll32_arr', [self.data.net, start.net, value_from_number(total).net],
                             total, self._stored)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, (self.n, *self.shape), self._dtype)
