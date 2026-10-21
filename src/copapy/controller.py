"""Controllers for periodic tasks. Like the time dependent operations
(cp.delay, cp.integrate, cp.rate, cp.feedback) they are functions of signals
without side effects: their state is kept in the memory of the task computing
them, the time step is the period of the task (cp.dt) or a given dt.

Gains and limits can be numbers or copapy values; values can be changed by
tg.write_value while the task is running (e.g. for tuning).
"""
from typing import Any
from ._basic_types import value, unifloat
from ._temporal import delay, feedback, rate, _time_constant, _gated, _delay_value
from ._math import minimum, maximum
from ._flow import iif

__all__ = ["pid"]


def _is_number(x: Any, number: float) -> bool:
    return isinstance(x, int | float) and x == number


def _weighted(w: unifloat, r: unifloat, y: unifloat) -> unifloat:
    """w * r - y without operations for the weights 0 and 1"""
    if _is_number(w, 0.0):
        return -y
    if _is_number(w, 1.0):
        return r - y
    return w * r - y


def _limit(u: unifloat, lo: 'unifloat | None', hi: 'unifloat | None') -> unifloat:
    if lo is not None:
        u = maximum(u, lo)
    if hi is not None:
        u = minimum(u, hi)
    return u


def _derivative(x: unifloat, kd: unifloat, tf: unifloat, dt: 'unifloat | None') -> unifloat:
    """kd * dx/dt, low-pass filtered with the time constant tf, discretized by
    the backward Euler method: d[k] = (tf * d[k-1] + kd * (x[k] - x[k-1])) / (tf + dt).
    It is 0 at the first run of the task."""
    if _is_number(tf, 0.0):
        return kd * rate(x, dt)
    inverse = 1.0 / (tf + _time_constant(dt, 1.0, 1))
    pole = tf * inverse
    gain = _gated(kd * inverse)
    diff = x - delay(x)
    filtered: value[float] = feedback(lambda prev: pole * prev + gain * diff, 0.0)
    return filtered


def pid(setpoint: unifloat, measurement: unifloat, kp: unifloat, ki: unifloat = 0.0, kd: unifloat = 0.0, *,
        tf: unifloat = 0.0, b: unifloat = 1.0, c: unifloat = 0.0,
        out_min: 'unifloat | None' = None, out_max: 'unifloat | None' = None, tt: 'unifloat | None' = None,
        feedforward: unifloat = 0.0, enable: 'value[Any] | bool' = True, default_out: unifloat = 0.0,
        dt: 'unifloat | None' = None) -> value[float]:
    """PID controller in parallel form with setpoint weighting, derivative
    filter, output limits with anti-windup and bumpless enabling:

        u = sat(kp * (b * r - y) + I + D + feedforward)
        I = integral of ki * (r - y), starting at default_out
        D = kd * d/dt (c * r - y), low-pass filtered by tf

    with r the setpoint and y the measurement. The integral is accumulated by
    the forward Euler method (I[k+1] = I[k] + dt * ki * e[k]), so the output
    depends on the current error by the proportional and derivative terms only.
    As ki is inside the integral, a change of ki does not change the output
    (bumpless tuning). The derivative is 0 at the first run of the task, there
    is no kick from the initial measurement.

    With out_min or out_max the integral is protected from winding up by
    back-calculation: while the output is limited, the integral tracks the
    limited output with the time constant tt.

    While enable is false, the output is default_out (limited) and the integral
    tracks it: after enabling the controller continues from default_out without
    a jump (for a controller with integral).

        >>> u = cp.controller.pid(setpoint, position, kp=2.0, ki=5.0, kd=0.1,
        ...                       tf=0.01, out_min=-1.0, out_max=1.0)

    Arguments:
        setpoint: Reference value r
        measurement: Measured value y of the controlled process
        kp: Proportional gain
        ki: Integral gain (kp / Ti), 0 for a controller without integral
        kd: Derivative gain (kp * Td), 0 for a controller without derivative
        tf: Time constant of the first order low-pass filter of the derivative
            in seconds, 0 for no filter. A typical value is Td / N with N
            between 5 and 20. The filter is stable for any time step.
        b: Setpoint weight of the proportional term; values below 1 reduce
            the overshoot after setpoint steps
        c: Setpoint weight of the derivative term; 0 (default) is the
            derivative of the measurement only (no kick by setpoint steps),
            1 the derivative of the error
        out_min: Lower limit of the output, None for no limit
        out_max: Upper limit of the output, None for no limit
        tt: Tracking time constant of the anti-windup in seconds, by default
            Ti = kp / ki. Smaller values reset the integral faster.
        feedforward: Value added to the output before the limitation
        enable: Controller active (value of type bool or number); while it is
            false the output is default_out
        default_out: Output while the controller is disabled and initial value
            of the integral: at the first run the output for zero error. For a
            controller without integral (ki = 0) it is a constant offset of
            the output.
        dt: Time step, by default the period of the task

    Returns:
        Control output u (limited to out_min and out_max)
    """
    for name, arg in (('tf', tf), ('tt', tt), ('dt', dt)):
        if isinstance(arg, int | float) and (arg < 0 if name == 'tf' else arg <= 0):
            raise ValueError(f"{name} must be {'non-negative' if name == 'tf' else 'positive'}, not {arg}")
    if isinstance(out_min, int | float) and isinstance(out_max, int | float) and out_min > out_max:
        raise ValueError(f"out_min {out_min} is above out_max {out_max}")
    if not isinstance(enable, value) and enable is not True:
        raise TypeError("enable must be a copapy value or True")
    limited = out_min is not None or out_max is not None
    switch = enable if isinstance(enable, value) else None

    v: unifloat = _weighted(b, setpoint, measurement)
    if not _is_number(kp, 1.0):
        v = kp * v
    if not _is_number(kd, 0.0):
        v = v + _derivative(_weighted(c, setpoint, measurement), kd, tf, dt)
    if not _is_number(feedforward, 0.0):
        v = v + feedforward

    if _is_number(ki, 0.0):
        if not _is_number(default_out, 0.0):
            v = v + default_out
        return _as_value(_limit(v if switch is None else iif(switch, v, default_out), out_min, out_max))

    step = _time_constant(dt, 1.0, 1)
    if limited:
        if tt is None:
            if _is_number(kp, 0.0):
                raise ValueError("tt is required for a limited output without proportional gain")
            tracking = step * ki / kp
        else:
            tracking = step / tt
    if isinstance(default_out, value):
        # The integral starts at 0, default_out is added at the first run
        first_run = _delay_value(0.0, 1.0)
        initial = 0.0
    else:
        initial = float(default_out)
    output: list[unifloat] = []

    def update(integral: value[float]) -> unifloat:
        if isinstance(default_out, value):
            integral = integral + first_run * default_out
        unlimited = v + integral
        u = _limit(unlimited if switch is None else iif(switch, unlimited, default_out), out_min, out_max)
        output.append(u)
        increment = ki * step * (setpoint - measurement)
        if limited:
            increment = increment + tracking * (u - unlimited)
        updated = integral + increment
        # Disabled: the integral for the output u at the next run
        return updated if switch is None else iif(switch, updated, u - v)

    feedback(update, initial)
    return _as_value(output[0])


def _as_value(x: unifloat) -> value[float]:
    if isinstance(x, value):
        return x
    raise TypeError("The setpoint, measurement or a gain must be a copapy value")
