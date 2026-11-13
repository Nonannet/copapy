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


def test_invert() -> None:
    x = cp.value(0.0)
    n = cp.value(5)
    results = run_sequence([~(x > 1.0), ~(x > 1.0) & (x > -1.0), ~n], {x: [0.0, 2.0, -2.0]})
    assert column(results, 0) == [1, 0, 1]
    assert column(results, 1) == [1, 0, 0]
    assert column(results, 2) == [-6, -6, -6]
    assert (~(x > 1.0)).dtype == 'bool' and (~n).dtype == 'int'
    with pytest.raises(TypeError):
        ~x  # type: ignore[misc]


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


def test_timers_with_half_period_duration() -> None:
    # A duration of k + 0.5 periods is rounded up, the timers must still expire
    u = cp.value(0)
    xs = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    results = run_sequence([cp.held(u, 1.25), cp.held(u, 0.25), cp.released(u, 1.25), cp.held(u, 1.0)],
                           {u: xs}, period=0.5)
    assert column(results, 0) == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    assert column(results, 1) == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert column(results, 2) == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert column(results, 3) == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


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


def test_time_in_is_cached() -> None:
    pump = Pump(trip=cp.value(0), idle=cp.value(0), start=cp.value(0), recover=cp.value(0))
    assert pump.time_in(Pump.RUN) is pump.time_in(Pump.RUN)
    assert pump.time_in(Pump.RUN) is not pump.time_in(Pump.STANDBY)


def test_subclass_can_not_extend() -> None:
    class Named(Pump):
        def running(self) -> cp.value[int]:
            return self.in_(Pump.RUN)

    assert Named.states is Pump.states and Named.transitions is Pump.transitions
    c = cp.value(0)
    assert Named(trip=c, idle=c, start=c, recover=c).running().dtype == 'bool'

    with pytest.raises(TypeError, match="can not extend the state machine Pump by EXTRA"):
        class MoreStates(Pump):
            EXTRA = cp.State()

    with pytest.raises(TypeError, match="can not extend the state machine Pump by stop"):
        class MoreTransitions(Pump):
            stop = Pump.RUN >> Pump.SHUTDOWN

    with pytest.raises(TypeError, match="declares transitions"):
        class NoStates(cp.StateMachine):
            stop = Pump.RUN >> Pump.SHUTDOWN


def test_after_combined() -> None:
    class Door(cp.StateMachine):
        CLOSED = cp.State()
        OPEN = cp.State()
        open = CLOSED >> OPEN
        close = OPEN >> CLOSED

    a, b, c = cp.value(0), cp.value(0), cp.value(0)
    # a since 0.02 s and b since 0.01 s in CLOSED and not c; back by c or 0.03 s in OPEN
    door = Door(open=cp.after(a, 0.02) & ~(c > 0) & cp.after(b, 0.01), close=c | cp.after(1, 0.03))
    seq = {a: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           b: [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           c: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]}
    results = run_sequence([door.state, door.fired(Door.open), door.fired(Door.close)], seq)
    # Both times are reached at run 4, where c blocks: open at run 5. Closed by its time
    # at run 8. The times restart in CLOSED: reached at run 10, where c blocks again
    assert column(results, 0) == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
    assert column(results, 1) == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert column(results, 2) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


def test_after_inverted_and_reflected() -> None:
    class Lamp(cp.StateMachine):
        OFF = cp.State()
        ON = cp.State()
        on = OFF >> ON
        off = ON >> OFF

    u = cp.value(0)
    timer = cp.after(u, 0.02)
    # The same after in two transitions has a time for each of them
    lamp = Lamp(on=u & ~timer, off=(u > 0) | (timer | 0))
    results = run_sequence([lamp.state], {u: [0, 1, 1, 1, 1, 1, 0, 0]})
    # on at once while u is not yet true for 0.02 s in OFF, off by u at the next run
    assert column(results, 0) == [0, 1, 0, 1, 0, 1, 1, 1]


def test_after_errors() -> None:
    u = cp.value(0)
    with pytest.raises(TypeError, match="no truth value"):
        bool(cp.after(u, 1.0) & u)
    with pytest.raises(TypeError, match="can not contain after"):
        cp.after(cp.after(u, 1.0), 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported type"):
        cp.after(u, 1.0) & "x"  # type: ignore[operator]


def test_reserved_names() -> None:
    for name in ['select', 'state', 'dt', 'states', 'transitions', 'initial', 'time_in', '_active']:
        with pytest.raises(TypeError, match=f"{name} can not be used as name"):
            type('M', (cp.StateMachine,), {'A': cp.State(), 'B': cp.State(), name: cp.State() >> cp.State()})
    with pytest.raises(TypeError, match="fired can not be used as name"):
        class StateNamed(cp.StateMachine):
            fired = cp.State()  # type: ignore[assignment]


def test_alias_is_rejected() -> None:
    with pytest.raises(TypeError, match="M.ALIAS is the same state as A"):
        class M(cp.StateMachine):
            A = cp.State()
            B = cp.State()
            ALIAS = A
            go = A >> B

    with pytest.raises(TypeError, match="N.again is the same transition as go"):
        class N(cp.StateMachine):
            A = cp.State()
            B = cp.State()
            go = A >> B
            again = go
