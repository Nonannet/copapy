"""Time dependent operations. Every copapy value is in concept a signal:
the sequence of its values at the runs of the task computing it. delay,
integrate, rate and feedback are functions of these signals without
side effects. They keep their state in the memory of the task; the state is
initialized at deployment and updated after each run of the task.

The time step is the period of the task (dt). Delays are counted in runs of
the task: for tasks without period (run by exec) a delay is the value of the
previous exec.
"""
from typing import Any, Callable, Sequence, TypeVar, overload
from ._basic_types import value, Net, Delay, TimeStep, transl_type, value_from_number, to_float, unifloat
from ._arrays import array, ArrayType
from ._vectors import vector
from ._tensors import tensor
from ._flow import iif

T = TypeVar("T")

dt: value[float] = value(Net('float', TimeStep()), 'float')
"""Period of the task computing the value in seconds. It is set at deployment,
a task using it must have a period."""

_METHODS = ('forward', 'backward', 'trapezoid')


def _initial(init: Any, dtype: str) -> int | float:
    """Initial value of a delay as number of the type of the delayed value"""
    if isinstance(init, value):
        raise TypeError("Initial values must be numbers, not copapy values")
    if transl_type(dtype) == 'float':
        return float(init)
    if isinstance(init, float) and not init.is_integer():
        raise TypeError(f"Initial value {init} of a delayed {dtype} value is no integer")
    return int(init)


def _init_values(init: Any, x: ArrayType[Any]) -> list[Any]:
    """Initial values for the elements of a vector or tensor: one number for
    all elements or a vector, tensor or sequence of the shape of x"""
    if isinstance(init, int | float | value):
        return [init] * len(x.values)
    if not isinstance(init, ArrayType):
        init = vector(init) if isinstance(x, vector) else tensor(init)
    values = list(init.values)
    if len(values) != len(x.values):
        raise ValueError(f"{len(values)} initial values for {len(x.values)} elements")
    return values


def _delay_value(x: value[Any] | int | float, init: Any) -> value[Any]:
    xv = x if isinstance(x, value) else value_from_number(x)
    node = Delay(_initial(init, xv.dtype), xv.net)
    return value(Net(node.dtype, node), xv.dtype)


@overload
def delay(x: value[Any] | int | float, init: Any = 0, n: int = 1) -> value[Any]: ...
@overload
def delay(x: vector[Any], init: Any = 0, n: int = 1) -> vector[Any]: ...
@overload
def delay(x: tensor[Any], init: Any = 0, n: int = 1) -> tensor[Any]: ...
@overload
def delay(x: Sequence[Any], init: Any = 0, n: int = 1) -> Any: ...
def delay(x: Any, init: Any = 0, n: int = 1) -> Any:
    """Value of x at the n-th previous run of the task (z^-n). For the
    first n runs the result is the initial value.

    Arguments:
        x: Value, vector, tensor or a tuple or list of them
        init: Initial value: a number for all elements or the same
            structure as x
        n: Number of runs to delay

    Returns:
        Delayed values of the type of x
    """
    if n < 0:
        raise ValueError(f"n must not be negative, not {n}")
    for _ in range(n):
        x = _map_init(x, init, _delay_value)
    return x


def _item_inits(init: Any, x: tuple[Any, ...] | list[Any]) -> list[Any]:
    """Initial values for the items of a tuple or list: one for all items or
    a tuple or list of the length of x"""
    inits = list(init) if isinstance(init, tuple | list) else [init] * len(x)
    if len(inits) != len(x):
        raise ValueError(f"{len(inits)} initial values for {len(x)} items")
    return inits


def _map_init(x: Any, init: Any, func: Callable[[Any, Any], Any]) -> Any:
    """Applies func to each scalar of x and its initial value"""
    if isinstance(x, value | int | float):
        return func(x, init)
    if isinstance(x, ArrayType):
        inits = iter(_init_values(init, x))
        return x.map(lambda e: func(e, next(inits)))
    if isinstance(x, array):
        raise NotImplementedError("Delays of arrays are not supported, use a vector or tensor")
    if isinstance(x, tuple | list):
        return type(x)(_map_init(xi, ii, func) for xi, ii in zip(x, _item_inits(init, x)))
    raise TypeError(f"Unsupported type {type(x).__name__}")


def _state(init: Any, delays: list[Delay]) -> Any:
    """Delays without input for each number of the structure init"""
    if isinstance(init, value):
        raise TypeError("Initial values must be numbers, not copapy values")
    if isinstance(init, int | float):
        node = Delay(init)
        delays.append(node)
        return value(Net(node.dtype, node), 'bool' if isinstance(init, bool) else node.dtype)
    if isinstance(init, ArrayType):
        return init.map(lambda e: _state(e, delays))
    if isinstance(init, tuple | list):
        return type(init)(_state(i, delays) for i in init)
    raise TypeError(f"Unsupported type of the initial value: {type(init).__name__}")


def _close(state: Any, out: Any, delays: list[Delay]) -> None:
    """Sets the input of each delay of state to the corresponding scalar of out"""
    if isinstance(state, value):
        if not isinstance(out, value | int | float):
            raise TypeError(f"feedback function returned {type(out).__name__} for a scalar")
        node = delays.pop(0)
        if isinstance(out, value):
            if transl_type(out.dtype) != node.dtype:
                if node.dtype != 'float':
                    raise TypeError(f"feedback function returned {out.dtype} for an int state, "
                                    "use a float initial value (e.g. 0.0)")
                out = to_float(out)
            node.input = out.net
        else:
            node.input = value_from_number(float(out) if node.dtype == 'float' else _initial(out, 'int')).net
    elif isinstance(state, ArrayType):
        if not isinstance(out, ArrayType) or out.shape != state.shape:
            raise TypeError(f"feedback function must return a {type(state).__name__} of shape {state.shape}")
        for s, o in zip(state.values, out.values):
            _close(s, o, delays)
    else:
        if not isinstance(out, tuple | list) or len(out) != len(state):
            raise TypeError(f"feedback function must return a {type(state).__name__} of length {len(state)}")
        for s, o in zip(state, out):
            _close(s, o, delays)


def feedback(func: Callable[[Any], T], init: Any) -> T:
    """Recurrence: y = func(delay(y, init)). The function gets the result of
    the previous run of the task (the initial value at the first run) and
    returns the new result. The loop always passes the delay, there is no
    algebraic loop.

        >>> # First order low-pass filter
        >>> y = cp.feedback(lambda y_prev: y_prev + 0.1 * (u - y_prev), init=0.0)

    Arguments:
        func: Function computing the new result from the previous one
        init: Initial value: a number, a vector or tensor of numbers or a
            tuple or list of them. Its structure and types are the
            structure and types of the result.

    Returns:
        The result of func
    """
    delays: list[Delay] = []
    state = _state(init, delays)
    out = func(state)
    _close(state, out, delays)
    return out


def _step(dt_: 'unifloat | None') -> unifloat:
    return dt if dt_ is None else dt_


def _first_run() -> value[int]:
    """1 at the first run of the task, then 0"""
    return _delay_value(0, 1)


def _like(x: Any, init: Any) -> Any:
    """Initial float value of the structure of x"""
    if isinstance(x, ArrayType):
        inits = iter(_init_values(init, x))
        return x.map(lambda _: _initial(next(inits), 'float'))
    if isinstance(x, tuple | list):
        return type(x)(_like(xi, ii) for xi, ii in zip(x, _item_inits(init, x)))
    return float(init)


@overload
def integrate(x: 'unifloat | value[int] | int', init: float = 0.0, method: str = 'trapezoid', dt: 'unifloat | None' = None) -> value[float]: ...
@overload
def integrate(x: vector[Any], init: Any = 0.0, method: str = 'trapezoid', dt: 'unifloat | None' = None) -> vector[float]: ...
@overload
def integrate(x: tensor[Any], init: Any = 0.0, method: str = 'trapezoid', dt: 'unifloat | None' = None) -> tensor[float]: ...
def integrate(x: Any, init: Any = 0.0, method: str = 'trapezoid', dt: 'unifloat | None' = None) -> Any:
    """Integral of x over time. The result is init at the first run of the
    task and increases by the integral of x over one time step at each further
    run, with x[k] the value at the k-th run:

    - 'forward' (forward Euler): dt * x[k-1], the result does not depend on the
      current x (no direct feedthrough)
    - 'backward' (backward Euler): dt * x[k]
    - 'trapezoid': dt * (x[k-1] + x[k]) / 2

    Arguments:
        x: Value, vector or tensor to integrate
        init: Initial value of the integral, a number or the structure of x
        method: 'forward', 'backward' or 'trapezoid'
        dt: Time step, by default the period of the task

    Returns:
        Integral as float value, vector or tensor
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}, not {method!r}")
    if isinstance(x, tuple | list | array):
        raise TypeError(f"Unsupported type {type(x).__name__}, use a vector or tensor")
    step = _step(dt)
    if method == 'forward':
        increment = delay(x) * step
    else:
        increment = (x if method == 'backward' else (x + delay(x)) * 0.5) * step
        increment = iif(_first_run(), 0.0, increment)
    return feedback(lambda prev: prev + increment, _like(x, init))


@overload
def rate(x: 'unifloat | value[int] | int', dt: 'unifloat | None' = None) -> value[float]: ...
@overload
def rate(x: vector[Any], dt: 'unifloat | None' = None) -> vector[float]: ...
@overload
def rate(x: tensor[Any], dt: 'unifloat | None' = None) -> tensor[float]: ...
def rate(x: Any, dt: 'unifloat | None' = None) -> Any:
    """Rate of change of x over time (derivative by the backward difference
    (x[k] - x[k-1]) / dt). It is 0 at the first run of the task.

    Arguments:
        x: Value, vector or tensor
        dt: Time step, by default the period of the task

    Returns:
        Rate as float value, vector or tensor
    """
    if isinstance(x, tuple | list | array):
        raise TypeError(f"Unsupported type {type(x).__name__}, use a vector or tensor")
    return iif(_first_run(), 0.0, (x - delay(x)) / _step(dt))
