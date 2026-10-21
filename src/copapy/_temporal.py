"""Time dependent operations. Every copapy value is in concept a signal:
the sequence of its values at the runs of the task computing it. delay,
integrate, rate and feedback are functions of these signals without
side effects. They keep their state in the memory of the task; the state is
initialized at deployment and updated after each run of the task.

The time step is the period of the task (dt). Delays are counted in runs of
the task: for tasks without period (run by exec) a delay is the value of the
previous exec.

Orientations as quaternions are integrated from angular velocities by
integrate_rotation, angular_rate is the angular velocity of an orientation.

Time constants (factors of dt) are written to the memory of the task at
deployment. Increments that are zero at the first run are multiplied with a
delayed time constant, which is zero at the first run, instead of masking.
"""
import math
from typing import Any, Callable, Sequence, TypeVar, overload
from ._basic_types import value, Net, Delay, TimeStep, transl_type, value_from_number, to_float, unifloat
from ._arrays import array, ArrayType
from ._vectors import vector, cross
from ._tensors import tensor
from ._quaternion import quaternion
from ._math import sqrt, atan2, maximum
from ._flow import iif
from ._helper_types import U

T = TypeVar("T", bound='value[Any] | ArrayType[Any] | tuple[Any, ...] | list[Any] | int | float')

dt: value[float] = value(Net('float', TimeStep()), 'float')
"""Period of the task computing the value in seconds. It is set at deployment,
a task using it must have a period."""

_METHODS = ('forward', 'backward', 'trapezoid')
_FRAMES = ('body', 'world')

# Time constants by factor and exponent of dt
_time_constants: dict[tuple[float, int], value[float]] = {(1.0, 1): dt}


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
def delay(x: int, init: Any = 0, n: int = 1) -> value[int]: ...
@overload
def delay(x: float, init: Any = 0, n: int = 1) -> value[float]: ...
@overload
def delay(x: value[U], init: Any = 0, n: int = 1) -> value[U]: ...
@overload
def delay(x: quaternion, init: Any = 0, n: int = 1) -> quaternion: ...
@overload
def delay(x: vector[U], init: Any = 0, n: int = 1) -> vector[U]: ...
@overload
def delay(x: tensor[U], init: Any = 0, n: int = 1) -> tensor[U]: ...
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
    if isinstance(x, int | float):
        x = value_from_number(x)  # A value also for n = 0
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


def _time_constant(dt_: 'unifloat | None', factor: float, exponent: int) -> unifloat:
    """factor * dt^exponent (exponent 1, 2 or -1). For the period of the task it
    is written to the memory of the task at deployment, so it costs no operation."""
    if dt_ is None:
        key = (factor, exponent)
        if key not in _time_constants:
            _time_constants[key] = value(Net('float', TimeStep(factor, exponent)), 'float')
        return _time_constants[key]
    if exponent == -1:
        return factor / dt_
    return factor * dt_ * dt_ if exponent == 2 else factor * dt_


def _gated(c: unifloat) -> value[float]:
    """0.0 at the first run of the task, then c: for an increment that is
    zero at the first run without masking the elements of the increment"""
    return _delay_value(c, 0.0)


def _like(x: Any, init: Any) -> Any:
    """Initial float value of the structure of x"""
    if isinstance(x, ArrayType):
        inits = iter(_init_values(init, x))
        return x.map(lambda _: _initial(next(inits), 'float'))
    if isinstance(x, tuple | list):
        return type(x)(_like(xi, ii) for xi, ii in zip(x, _item_inits(init, x)))
    return float(init)


def _check_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}, not {method!r}")


def _check_frame(frame: str) -> bool:
    """True for the body frame"""
    if frame not in _FRAMES:
        raise ValueError(f"frame must be 'body' or 'world', not {frame!r}")
    return frame == 'body'


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

    Orientations are integrated by integrate_rotation.

    Arguments:
        x: Value, vector or tensor to integrate
        init: Initial value of the integral, a number or the structure of x
        method: 'forward', 'backward' or 'trapezoid'
        dt: Time step, by default the period of the task

    Returns:
        Integral as float value, vector or tensor
    """
    _check_method(method)
    if isinstance(x, quaternion):
        raise TypeError("Quaternions are not integrated component-wise, "
                        "use integrate_rotation for the angular velocity")
    if isinstance(x, tuple | list | array):
        raise TypeError(f"Unsupported type {type(x).__name__}, use a vector or tensor")
    if method == 'forward':
        increment = delay(x) * _time_constant(dt, 1.0, 1)
    elif method == 'backward':
        increment = x * _gated(_time_constant(dt, 1.0, 1))
    else:
        increment = (x + delay(x)) * _gated(_time_constant(dt, 0.5, 1))
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

    The angular velocity of an orientation is computed by angular_rate.

    Arguments:
        x: Value, vector or tensor
        dt: Time step, by default the period of the task

    Returns:
        Rate as float value, vector or tensor
    """
    if isinstance(x, quaternion):
        raise TypeError("The rate of a quaternion is not computed component-wise, "
                        "use angular_rate for the angular velocity")
    if isinstance(x, tuple | list | array):
        raise TypeError(f"Unsupported type {type(x).__name__}, use a vector or tensor")
    return (x - delay(x)) * _gated(_time_constant(dt, 1.0, -1))


# Truncation error of the series of the rotation (float32 precision)
_SERIES_TOLERANCE = 2.0 ** -24


def _series(max_angle: float) -> tuple[list[float], list[float]]:
    """Taylor coefficients in a^2 of cos(a) and sin(a) / a, as many as
    required for half rotation angles a up to max_angle / 2"""
    if not 0.0 < max_angle <= 2.0 * math.pi:
        raise ValueError(f"max_angle must be above 0 and at most 2 * pi, not {max_angle}")
    a = max_angle / 2.0
    n = 1
    while a ** (2 * n) / math.factorial(2 * n) > _SERIES_TOLERANCE:
        n += 1
    return ([(-1) ** k / math.factorial(2 * k) for k in range(n)],
            [(-1) ** k / math.factorial(2 * k + 1) for k in range(n)])


def _polynomial(coefficients: list[float], x: Any) -> Any:
    """Polynomial by the Horner scheme"""
    result: Any = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        result = result * x + c
    return result


def _rotation(half: vector[Any], max_angle: float) -> quaternion:
    """Unit quaternion exp(half) of the rotation by the rotation vector 2 * half.
    The series of cos and sin need no square root, no division and no branch;
    they are accurate to float precision for rotation angles up to max_angle."""
    cos_series, sinc_series = _series(max_angle)
    a2 = half.dot(half)
    return quaternion(_polynomial(cos_series, a2), *(half * _polynomial(sinc_series, a2)).values)


def _renormalized(q: quaternion) -> quaternion:
    """q scaled to unit length by one Newton step of 1 / sqrt(|q|^2): for
    quaternions with a norm close to 1, without square root and division"""
    n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
    return q * (1.5 - 0.5 * n2)


def _mul_conjugate(a: quaternion, b: quaternion, body: bool) -> quaternion:
    """conjugate(a) @ b (body) or b @ conjugate(a) (world)"""
    if not body:
        return quaternion(b.w * a.w + b.x * a.x + b.y * a.y + b.z * a.z,
                          b.x * a.w - b.w * a.x - b.y * a.z + b.z * a.y,
                          b.y * a.w - b.w * a.y + b.x * a.z - b.z * a.x,
                          b.z * a.w - b.w * a.z - b.x * a.y + b.y * a.x)
    return quaternion(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z,
                      a.w * b.x - a.x * b.w - a.y * b.z + a.z * b.y,
                      a.w * b.y + a.x * b.z - a.y * b.w - a.z * b.x,
                      a.w * b.z - a.x * b.y + a.y * b.x - a.z * b.w)


def _rotation_vector(d: quaternion) -> vector[float]:
    """Rotation vector (2 log(d)) of the shorter of the two rotations of the
    quaternions d and -d, which describe the same orientation"""
    sign = iif(d.w < 0.0, -1.0, 1.0)
    v = vector(d.values[1:])
    s = sqrt(v.dot(v))
    # atan2(s, w) / s for s -> 0 is 1 / w; v is zero for s = 0
    return v * (2.0 * sign * atan2(s, d.w * sign) / maximum(s, 1e-30))


def _vector3(omega: Any) -> vector[Any]:
    if isinstance(omega, quaternion) or not isinstance(omega, vector | tuple | list):
        raise TypeError(f"The angular velocity must be a vector, not {type(omega).__name__}")
    result: vector[Any] = omega if isinstance(omega, vector) else vector(omega)
    if len(result) != 3:
        raise ValueError(f"The angular velocity must have 3 elements, not {len(result)}")
    return result


def integrate_rotation(omega: 'vector[Any] | Sequence[unifloat]', init: quaternion | None = None,
                       method: str = 'trapezoid', dt: 'unifloat | None' = None, frame: str = 'body',
                       max_angle: float = math.pi) -> quaternion:
    """Orientation as unit quaternion from the angular velocity. At each run of
    the task the orientation is rotated by the integral of the angular velocity
    over the time step, with omega[k] the value at the k-th run:

    - 'forward': rotation by dt * omega[k-1] (no direct feedthrough)
    - 'backward': rotation by dt * omega[k], exact for an angular velocity
      that is constant over the time step
    - 'trapezoid': rotation by dt * (omega[k-1] + omega[k]) / 2 with the
      second order (coning) term cross(omega[k-1], omega[k]) * dt^2 / 12 for an
      angular velocity changing linearly over the time step

        >>> orientation = cp.integrate_rotation(gyro)

    Arguments:
        omega: Angular velocity in rad/s, vector of 3 elements
        init: Orientation at the first run, quaternion of numbers (normalized),
            by default the identity
        method: 'forward', 'backward' or 'trapezoid'
        dt: Time step, by default the period of the task
        frame: Frame of the angular velocity: 'body' (rotating with the
            orientation, e.g. measured by a gyroscope: q = q @ rotation)
            or 'world' (fixed: q = rotation @ q)
        max_angle: Largest rotation angle in rad in one time step (|omega| * dt)
            for which the rotation is accurate to float precision, at most 2 * pi.
            A smaller angle needs fewer operations (e.g. 8 instead of 24
            multiplications and additions for 0.1 instead of pi), larger
            rotations lose accuracy.

    Returns:
        Orientation as unit quaternion
    """
    _check_method(method)
    body = _check_frame(frame)
    omega = _vector3(omega)
    _series(max_angle)  # Check before building the graph
    if init is None:
        init = quaternion.identity()
    numbers = [v for v in init.values if isinstance(v, int | float)] \
        if isinstance(init, quaternion) and init._values is not None else []
    if len(numbers) != 4:
        raise TypeError("The initial orientation must be a quaternion of numbers")
    norm = math.sqrt(sum(v * v for v in numbers))
    if not norm:
        raise ValueError("The initial orientation must not be zero")

    # Half of the rotation vector of the time step
    if method == 'forward':
        half = delay(omega) * _time_constant(dt, 0.5, 1)
    elif method == 'backward':
        half = omega * _gated(_time_constant(dt, 0.5, 1))
    else:
        # The coning term is zero at the first run (the delayed omega is zero)
        prev = delay(omega)
        half = (omega + prev) * _gated(_time_constant(dt, 0.25, 1)) + \
            cross(prev, omega) * _time_constant(dt, 1 / 24 if body else -1 / 24, 2)
    rotation = _rotation(half, max_angle)

    def step_rotation(q: quaternion) -> quaternion:
        return _renormalized(q @ rotation if body else rotation @ q)

    return feedback(step_rotation, quaternion(v / norm for v in numbers))


def angular_rate(q: quaternion, dt: 'unifloat | None' = None, frame: str = 'body') -> vector[float]:
    """Angular velocity (in rad/s) of an orientation as unit quaternion: the
    rotation from the previous to the current orientation divided by the time
    step. It is the inverse of integrate_rotation (exact for its method
    'backward') and 0 at the first run of the task. A change of sign of the
    quaternion (q and -q describe the same orientation) has no effect.

    Arguments:
        q: Orientation as unit quaternion
        dt: Time step, by default the period of the task
        frame: Frame of the angular velocity: 'body' (rotating with the
            orientation, like a gyroscope) or 'world' (fixed)

    Returns:
        Angular velocity as vector of 3 elements
    """
    body = _check_frame(frame)
    if not isinstance(q, quaternion):
        raise TypeError(f"The orientation must be a quaternion, not {type(q).__name__}")
    # The previous orientation at the first run is the identity, the result is zero
    d = _mul_conjugate(delay(q, init=(1.0, 0.0, 0.0, 0.0)), q, body)
    return _rotation_vector(d) * _gated(_time_constant(dt, 1.0, -1))
