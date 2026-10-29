"""Tests for the boolean signal operations (edges, latch, hysteresis,
timers) and state machines on the local target."""
from typing import Any

import pytest

import copapy as cp


def run_sequence(outputs: Any, inputs: dict[cp.value[Any], list[Any]], period: float = 0.01) -> list[Any]:
    """Results of the periodic task computing outputs for each step of inputs"""
    tg = cp.LocalTarget(tick=period)
    tg.deploy(cp.Task(outputs, period=period))
    results: list[Any] = []
    for k in range(len(next(iter(inputs.values())))):
        for v, xs in inputs.items():
            tg.write_value(v, xs[k])
        tg.step((k + 1) * period)
        results.append(tg.read_value(outputs))
    return results


def column(results: list[Any], i: int) -> list[int]:
    return [int(r[i]) for r in results]


def test_bool_dtype_of_logic_ops() -> None:
    a = cp.value(1.0)
    assert ((a > 0.0) & (a < 2.0)).dtype == 'bool'
    assert ((a > 0.0) | True).dtype == 'bool'
    assert ((a > 0.0) & cp.value(3)).dtype == 'int'


def test_edges() -> None:
    u = cp.value(0)
    xs = [1, 1, 0, 0, 1, 0]
    results = run_sequence([cp.rising(u), cp.falling(u), cp.rising(u, init=True), cp.falling(u > 0, init=True)], {u: xs})
    assert column(results, 0) == [1, 0, 0, 0, 1, 0]
    assert column(results, 1) == [0, 0, 1, 0, 0, 1]
    assert column(results, 2) == [0, 0, 0, 0, 1, 0]
    assert column(results, 3) == [0, 0, 1, 0, 0, 1]


def test_latch() -> None:
    s = cp.value(0)
    r = cp.value(0)
    results = run_sequence([cp.latch(s, r), cp.latch(s, r, dominant='set'), cp.latch(s, r, init=True)],
                           {s: [0, 1, 0, 1, 0, 0], r: [0, 0, 0, 1, 1, 0]})
    assert column(results, 0) == [0, 1, 1, 0, 0, 0]
    assert column(results, 1) == [0, 1, 1, 1, 0, 0]
    assert column(results, 2) == [1, 1, 1, 0, 0, 0]
    with pytest.raises(ValueError):
        cp.latch(s, r, dominant='both')


def test_hysteresis() -> None:
    x = cp.value(0.0)
    xs = [5.0, 11.0, 12.0, 11.0, 10.5, 10.0, 11.0, 0.0]
    results = run_sequence([cp.hysteresis(x, on=12.0, off=10.0), cp.hysteresis(x, on=10.0, off=12.0)], {x: xs})
    assert column(results, 0) == [0, 0, 1, 1, 1, 0, 0, 0]
    assert column(results, 1) == [1, 1, 0, 0, 0, 1, 1, 1]


def test_held_and_released() -> None:
    u = cp.value(0)
    xs = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    results = run_sequence([cp.held(u, 0.03), cp.released(u, 0.03), cp.held(u, 0.0)], {u: xs})
    assert column(results, 0) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert column(results, 1) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert column(results, 2) == xs
    assert column(run_sequence([cp.released(u, 0.02)], {u: [0, 0, 1, 0]}), 0) == [0, 0, 1, 1]


def test_held_long_duration() -> None:
    u = cp.value(1)
    tg = cp.LocalTarget(tick=0.001)
    h = cp.held(u, 300.0)
    tg.deploy(cp.Task(h, period=0.001))
    tg.step(299.999)
    assert not tg.read_value(h)
    tg.step(300.001)
    assert tg.read_value(h)


class Pump(cp.StateMachine):
    RUN = cp.State(initial=True)
    STANDBY = cp.State()
    SHUTDOWN = cp.State()

    trip = cp.ANY >> SHUTDOWN
    idle = RUN >> STANDBY
    start = STANDBY >> RUN
    recover = SHUTDOWN >> STANDBY


def test_declaration() -> None:
    assert Pump.states == (Pump.RUN, Pump.STANDBY, Pump.SHUTDOWN)
    assert [int(s) for s in Pump.states] == [0, 1, 2]
    assert Pump.initial is Pump.RUN
    assert [t.name for t in Pump.transitions] == ['trip', 'idle', 'start', 'recover']
    assert Pump.idle.sources == (Pump.RUN,) and Pump.idle.target is Pump.STANDBY
    assert Pump.trip.sources is None
    assert repr(Pump.STANDBY) == 'Pump.STANDBY'

    class Valve(cp.StateMachine):
        OPEN = cp.State()
        CLOSED = cp.State(initial=True)
        MOVING = cp.State()
        stop = (OPEN | CLOSED) >> MOVING

    assert Valve.initial is Valve.CLOSED
    assert Valve.stop.sources == (Valve.OPEN, Valve.CLOSED)

    with pytest.raises(TypeError):
        class Twice(cp.StateMachine):
            A = cp.State(initial=True)
            B = cp.State(initial=True)

    with pytest.raises(TypeError):
        class Foreign(cp.StateMachine):
            A = cp.State()
            go = A >> Pump.RUN


def test_conditions_checked() -> None:
    c = cp.value(0)
    with pytest.raises(TypeError, match='recover'):
        Pump(trip=c, idle=c, start=c)
    with pytest.raises(TypeError, match='stop'):
        Pump(trip=c, idle=c, start=c, recover=c, stop=c)


def test_pump() -> None:
    pressure = cp.value(5.0)
    speed = cp.value(100.0)
    start = cp.value(0)
    ack = cp.value(0)
    pump = Pump(trip=pressure > 12.0, idle=cp.after(speed < 50.0, 0.03),
                start=start, recover=ack & (pressure < 10.0))
    motor = pump.select({Pump.RUN: 2.0 * speed}, default=0.0)
    outputs = [pump.state, motor, pump.entered(Pump.STANDBY), pump.fired(Pump.trip), pump.time_in(Pump.RUN)]

    #             0      1      2      3      4      5      6      7      8      9      10
    p = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 13.0, 13.0, 9.0, 9.0]
    v = [100.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
    s = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    a = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
    results = run_sequence(outputs, {pressure: p, speed: v, start: s, ack: a})
    R, S, X = int(Pump.RUN), int(Pump.STANDBY), int(Pump.SHUTDOWN)
    # Slow since run 1, standby after 3 runs; start; trip while running;
    # the acknowledgment counts only below the pressure limit
    assert column(results, 0) == [R, R, R, R, S, R, R, X, X, S, S]
    assert [r[1] for r in results] == [200.0, 80.0, 80.0, 80.0, 0.0, 80.0, 80.0, 0.0, 0.0, 0.0, 0.0]
    assert column(results, 2) == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert column(results, 3) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert [r[4] for r in results] == pytest.approx([0.0, 0.01, 0.02, 0.03, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]
    assert Pump.states[results[-1][0]] is Pump.STANDBY


def test_after_restarts_in_source() -> None:
    class Blink(cp.StateMachine):
        ON = cp.State()
        OFF = cp.State()
        off = ON >> OFF
        on = OFF >> ON

    always = cp.value(1)
    blink = Blink(off=cp.after(always, 0.02), on=cp.after(always, 0.01))
    results = run_sequence([blink.state, blink.entered(Blink.ON)], {always: [1] * 9})
    # The timer of a transition starts at the run entering its source,
    # for the initial state at the first run
    assert column(results, 0) == [0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert column(results, 1) == [1, 0, 0, 1, 0, 0, 1, 0, 0]


def test_priority() -> None:
    class M(cp.StateMachine):
        A = cp.State()
        B = cp.State()
        C = cp.State()
        to_c = A >> C
        to_b = A >> B

    c = cp.value(0)
    b = cp.value(0)
    m = M(to_c=c, to_b=b)
    results = run_sequence([m.state, m.fired(M.to_b), m.fired(M.to_c), m.in_(M.B, M.C)], {c: [0, 1], b: [0, 1]})
    assert results[1][:3] == [int(M.C), 0, 1]
    assert column(results, 3) == [0, 1]
    m2 = M(to_c=c, to_b=b)
    results = run_sequence([m2.state, m2.fired(M.to_b)], {c: [0], b: [1]})
    assert results[0] == [int(M.B), 1]


def test_select_requires_all_states() -> None:
    pump = Pump(trip=0, idle=0, start=0, recover=0)
    with pytest.raises(ValueError):
        pump.select({Pump.RUN: 1.0})
    with pytest.raises(ValueError):
        pump.entered(cp.State())


def test_select_types() -> None:
    c = cp.value(0)
    pump = Pump(trip=0, idle=0, start=c, recover=0)
    as_int = pump.select({Pump.RUN: 1, Pump.STANDBY: cp.value(2)}, default=3)
    as_float = pump.select({Pump.RUN: cp.value(2)}, default=0.5)
    as_vector = pump.select({Pump.RUN: cp.vector([1.0, 2.0])}, default=cp.vector([0.0, 0.0]))
    assert as_int.dtype == 'int' and as_float.dtype == 'float'
    assert isinstance(as_vector, cp.vector)
    results = run_sequence([as_int, as_float, as_vector], {c: [0]})
    assert results[0][0] == 1 and results[0][1] == 2.0
    assert results[0][2].values == (1.0, 2.0)

    class Single(cp.StateMachine):
        ONLY = cp.State()

    single = Single()
    assert isinstance(single.select({Single.ONLY: 4.0}), cp.value)
