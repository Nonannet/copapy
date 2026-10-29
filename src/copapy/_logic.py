"""Boolean signals over time: edges, latches, hysteresis and timers. Like
delay and feedback they keep their state in the memory of the task.

Conditions are values of dtype bool (e.g. results of comparisons combined
by & and |) or other values, which are true if they are non-zero. The
results are values of dtype bool (0 or 1).

Timers count the runs of the task: a duration is rounded to the nearest
number of periods (dt).
"""
from typing import Any
from ._basic_types import value, unifloat, NumLike
from ._temporal import delay, feedback, time_constant
from ._flow import iif

# Run count of a timer that expired before deployment: no duration is longer
_EXPIRED = 2 ** 30


def truth(x: NumLike) -> 'value[int] | int':
    """x as boolean: a bool value or 0 or 1"""
    if isinstance(x, value):
        return value(x.net, 'bool') if x.dtype == 'bool' else x != 0
    return int(bool(x))


def as_bool(x: 'value[Any] | int') -> value[int]:
    if isinstance(x, value):
        return value(x.net, 'bool')
    return value(bool(x))


def rising(x: NumLike, init: bool = False) -> value[int]:
    """True at the runs of the task at which x became true (rising edge).

    Arguments:
        x: Condition
        init: Value of x assumed before the first run: with False the
            result is true at the first run if x is true

    Returns:
        Boolean value
    """
    b = truth(x)
    return as_bool(b > delay(b, init=int(init)))


def falling(x: NumLike, init: bool = False) -> value[int]:
    """True at the runs of the task at which x became false (falling edge).

    Arguments:
        x: Condition
        init: Value of x assumed before the first run

    Returns:
        Boolean value
    """
    b = truth(x)
    return as_bool(delay(b, init=int(init)) > b)


def latch(set: NumLike, reset: NumLike,
          init: bool = False, dominant: str = 'reset') -> value[int]:
    """Set-reset latch: becomes true if set is true and stays true until
    reset is true.

        >>> # Shut down on overpressure until acknowledged by the host
        >>> shutdown = cp.latch(pressure > 12.0, reset=ack)

    Arguments:
        set: Condition setting the latch
        reset: Condition resetting the latch
        init: State before the first run
        dominant: 'reset' or 'set': the result if both are true

    Returns:
        Boolean value
    """
    if dominant not in ('reset', 'set'):
        raise ValueError(f"dominant must be 'reset' or 'set', not {dominant!r}")
    s, r = truth(set), truth(reset)
    if dominant == 'reset':
        return as_bool(feedback(lambda q: (s | q) > r, init=bool(init)))
    return as_bool(feedback(lambda q: s | (q > r), init=bool(init)))


def hysteresis(x: NumLike, on: NumLike, off: NumLike, init: bool = False) -> value[int]:
    """Threshold with hysteresis (Schmitt trigger). For on > off the result
    becomes true if x reaches on and false if x falls to off. For on < off
    it becomes true if x falls to on and false if x reaches off.

        >>> overpressure = cp.hysteresis(pressure, on=12.0, off=10.0)

    Arguments:
        x: Value to compare
        on: Threshold switching the result on
        off: Threshold switching the result off. If on or off is a copapy
            value, off must not be above on.
        init: State before the first run

    Returns:
        Boolean value
    """
    if isinstance(on, value) or isinstance(off, value) or on >= off:
        return as_bool(feedback(lambda q: (x >= on) | ((x > off) & q), init=bool(init)))
    return as_bool(feedback(lambda q: (x <= on) | ((x < off) & q), init=bool(init)))


def runs_limit(seconds: unifloat, dt: 'unifloat | None') -> unifloat:
    """Limit of a runs count reaching the duration seconds: a count above it
    reaches the duration in runs, rounded to the nearest one"""
    if isinstance(seconds, value):
        return seconds * time_constant(dt, 1.0, -1) - 0.5
    return time_constant(dt, float(seconds), -1) - 0.5


def _timer(cond: 'value[int] | int', seconds: unifloat, dt: 'unifloat | None', expired: bool) -> value[int]:
    """True if cond is true since at least seconds"""
    limit = runs_limit(seconds, dt)
    # Runs since cond became true, -1 while false; it stops counting above the limit
    count = feedback(lambda n: iif(cond, n + (n < limit), -1), init=_EXPIRED if expired else -1)
    return as_bool(count > limit)


def held(condition: NumLike, seconds: unifloat, dt: 'unifloat | None' = None) -> value[int]:
    """True if condition is true continuously since at least seconds
    (on-delay). It is false again as soon as condition is false.

        >>> slow = cp.held(speed < 50.0, 300.0)  # for 5 minutes

    Arguments:
        condition: Condition
        seconds: Duration
        dt: Time step, by default the period of the task

    Returns:
        Boolean value
    """
    return _timer(truth(condition), seconds, dt, expired=False)


def released(condition: NumLike, seconds: unifloat, dt: 'unifloat | None' = None) -> value[int]:
    """True while condition is true and for seconds after it became false
    (off-delay). It is false at the first runs if condition is false.

        >>> fan_on = cp.released(motor_on, 60.0)  # runs on for one minute

    Arguments:
        condition: Condition
        seconds: Duration
        dt: Time step, by default the period of the task

    Returns:
        Boolean value
    """
    return as_bool(_timer(truth(condition) == 0, seconds, dt, expired=True) == 0)
