"""Tests for the time dependent operations delay, feedback, integrate,
rate and dt on the local target."""
from typing import Any

import pytest

import copapy as cp


def approx(v: object) -> object:
    return pytest.approx(v, rel=1e-5, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def run_sequence(outputs: Any, u: cp.value[float], inputs: list[float], period: float = 0.01) -> list[Any]:
    """Results of the periodic task computing outputs for each input value"""
    tg = cp.LocalTarget(tick=period)
    tg.deploy(cp.Task(outputs, period=period))
    results: list[Any] = []
    for k, x in enumerate(inputs):
        tg.write_value(u, x)
        tg.step((k + 1) * period)
        results.append(tg.read_value(outputs))
    return results


def test_delay() -> None:
    u = cp.value(0.0)
    i = cp.value(0)
    xs = [1.0, 2.0, 4.0, 8.0, 16.0]
    outputs = [cp.delay(u), cp.delay(u, init=-1.0), cp.delay(u, n=2), cp.delay(u * 2, n=0),
               cp.delay(i + 1, init=5), cp.delay(3.0, init=1.0)]
    results = run_sequence(outputs, u, xs)
    assert [r[0] for r in results] == [0.0, 1.0, 2.0, 4.0, 8.0]
    assert [r[1] for r in results] == [-1.0, 1.0, 2.0, 4.0, 8.0]
    assert [r[2] for r in results] == [0.0, 0.0, 1.0, 2.0, 4.0]
    assert [r[3] for r in results] == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert [r[4] for r in results] == [5, 1, 1, 1, 1]
    assert [r[5] for r in results] == [1.0, 3.0, 3.0, 3.0, 3.0]


def test_delay_structures() -> None:
    u = cp.value(0.0)
    v = cp.vector([u, u + 1.0])
    t = cp.tensor([[u, u - 1.0], [u * 0.5, u * 3]])
    dv = cp.delay(v, init=[7.0, 8.0])
    dt_ = cp.delay(t, init=9.0)
    dl = cp.delay((u, v), init=(1.0, 2.0))
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task([dv, dt_, dl], period=0.01))
    tg.write_value(u, 2.0)
    tg.step(0.01)
    assert tg.read_value(dv).values == (7.0, 8.0)
    assert tg.read_value(dt_).values == (9.0, 9.0, 9.0, 9.0)
    assert tg.read_value(dl[0]) == 1.0
    assert tg.read_value(dl[1]).values == (2.0, 2.0)
    tg.step(0.02)
    assert tg.read_value(dv).values == (2.0, 3.0)
    assert tg.read_value(dt_).values == (2.0, 1.0, 1.0, 6.0)
    assert tg.read_value(dl[0]) == 2.0


def test_feedback() -> None:
    u = cp.value(0.0)
    xs = [1.0, 1.0, 3.0, -2.0, 0.5]
    alpha = 0.25
    y = cp.feedback(lambda prev: prev + alpha * (u - prev), init=0.5)
    # Two states: counter and sum of the inputs
    n, total = cp.feedback(lambda p: (p[0] + 1, p[1] + u), init=(0, 0.0))
    results = run_sequence([y, n, total], u, xs)

    expected: list[float] = []
    prev = 0.5
    for x in xs:
        prev = prev + alpha * (x - prev)
        expected.append(prev)
    assert [r[0] for r in results] == approx(expected)
    assert [r[1] for r in results] == [1, 2, 3, 4, 5]
    assert [r[2] for r in results] == approx([1.0, 2.0, 5.0, 3.0, 3.5])


def test_feedback_vector() -> None:
    # Rotation of a vector by a constant matrix in each step, packed
    # vectors with many elements
    u = cp.value(1.0)
    m = cp.tensor([[0.0, -1.0], [1.0, 0.0]])
    rot = cp.feedback(lambda p: m @ p, init=cp.vector([1.0, 0.0]))
    large = cp.feedback(lambda p: p + u, init=cp.vector([float(i) for i in range(40)]))
    results = run_sequence([rot, large], u, [1.0, 1.0, 1.0, 1.0])
    assert [r[0].values for r in results] == approx([(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)])
    assert results[-1][1].values == approx([i + 4.0 for i in range(40)])


def test_integrate() -> None:
    u = cp.value(0.0)
    xs = [1.0, 2.0, 2.0, -1.0, 0.0]
    period = 0.02
    outputs = [cp.integrate(u, init=1.0), cp.integrate(u, method='forward'),
               cp.integrate(u, method='backward'), cp.integrate(u, dt=0.5),
               cp.integrate(cp.vector([u, cp.value(1.0)]), init=[0.0, 2.0], method='backward')]
    results = run_sequence(outputs, u, xs, period)

    def reference(method: str, init: float, step: float) -> list[float]:
        y, out = init, []
        for k, x in enumerate(xs):
            if k:
                y += step * {'forward': xs[k - 1], 'backward': x, 'trapezoid': 0.5 * (x + xs[k - 1])}[method]
            out.append(y)
        return out

    assert [r[0] for r in results] == approx(reference('trapezoid', 1.0, period))
    assert [r[1] for r in results] == approx(reference('forward', 0.0, period))
    assert [r[2] for r in results] == approx(reference('backward', 0.0, period))
    assert [r[3] for r in results] == approx(reference('trapezoid', 0.0, 0.5))
    assert [r[4].values[0] for r in results] == approx(reference('backward', 0.0, period))
    assert [r[4].values[1] for r in results] == approx([2.0 + k * period for k in range(5)])


def test_rate() -> None:
    u = cp.value(0.0)
    xs = [1.0, 2.0, 4.0, 4.0]
    results = run_sequence([cp.rate(u), cp.rate(cp.vector([u, u * u]), dt=1.0)], u, xs)
    assert [r[0] for r in results] == approx([0.0, 100.0, 200.0, 0.0])
    assert [r[1].values for r in results] == approx([(0.0, 0.0), (1.0, 3.0), (2.0, 12.0), (0.0, 0.0)])


def test_closed_loop() -> None:
    # PI controller and first order plant in one task
    setpoint = cp.value(0.0)
    kp, ki, tau = 2.0, 5.0, 0.1
    period = 0.01

    def step(prev: tuple[Any, Any]) -> tuple[Any, Any]:
        plant, integral = prev
        error = setpoint - plant
        control = kp * error + ki * integral
        return plant + cp.dt / tau * (control - plant), integral + cp.dt * error

    plant, _ = cp.feedback(step, init=(0.0, 0.0))
    results = run_sequence(plant, setpoint, [1.0] * 50, period)

    p, integral, expected = 0.0, 0.0, []
    for _ in range(50):
        error = 1.0 - p
        control = kp * error + ki * integral
        p, integral = p + period / tau * (control - p), integral + period * error
        expected.append(p)
    assert results == approx(expected)


def test_dt_per_task() -> None:
    t1 = cp.integrate(1.0)
    t2 = cp.integrate(1.0)
    task1 = cp.Task(t1, period=0.01)
    task2 = cp.Task(t2, period=0.05)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(task1, task2)
    tg.step(0.1)
    assert tg.read_value(t1) == approx(0.09)
    assert tg.read_value(t2) == approx(0.05)


def test_delay_between_tasks() -> None:
    u = cp.value(0.0)
    # The delay is computed by the task with the output, read by the other one
    d = cp.delay(u * 2)
    e = d + 1.0
    fast = cp.Task(d, period=0.01)
    slow = cp.Task(e, period=0.02)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(fast, slow)
    tg.write_value(u, 1.0)
    tg.step(0.02)
    assert tg.read_value([d, e]) == [2.0, 1.0]
    tg.step(0.04)
    assert tg.read_value([d, e]) == [2.0, 3.0]


def test_triggered_task() -> None:
    # Delays of a task without period count the exec calls
    u = cp.value(0.0)
    counter = cp.feedback(lambda c: c + 1, init=0)
    d = cp.delay(u)
    tg = cp.LocalTarget()
    tg.deploy(counter, d)
    for x in [3.0, 4.0, 5.0]:
        tg.write_value(u, x)
        tg.exec()
    assert tg.read_value(counter) == 3
    assert tg.read_value(d) == 4.0


def test_redeploy_resets_state() -> None:
    counter = cp.feedback(lambda c: c + 1, init=0)
    task = cp.Task(counter, period=0.01)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(task)
    tg.step(0.05)
    assert tg.read_value(counter) == 5
    tg.deploy(task)
    tg.step(0.02)
    assert tg.read_value(counter) == 2


def test_grad_through_delay() -> None:
    # Delayed values are constants for the gradient (truncated)
    u = cp.value(3.0)
    d = cp.delay(u, init=2.0)
    y = u * d
    g = cp.grad(y, u)
    tg = cp.LocalTarget()
    tg.deploy(y, g)
    tg.exec()
    assert tg.read_value(g) == 2.0
    tg.exec()
    assert tg.read_value(g) == 3.0


def test_errors() -> None:
    u = cp.value(1.0)
    with pytest.raises(ValueError, match="no period"):
        cp.LocalTarget().deploy(cp.integrate(u))
    with pytest.raises(ValueError, match="by"):
        d = cp.delay(u)
        cp.LocalTarget(tick=0.01).deploy(cp.Task(d + 1, period=0.01), cp.Task(d + 2, period=0.02))
    with pytest.raises(ValueError, match="cycle"):
        a, b = cp.feedback(lambda p: (p[1], p[0]), init=(0.0, 1.0))
        cp.LocalTarget().deploy(a, b)
    with pytest.raises(ValueError, match="require tasks"):
        cp.Target().compile(cp.delay(u))
    with pytest.raises(TypeError, match="float initial value"):
        cp.feedback(lambda p: p + 0.5, init=0)
    with pytest.raises(TypeError, match="no integer"):
        cp.delay(cp.value(1), init=0.5)
    with pytest.raises(ValueError, match="method"):
        cp.integrate(u, method='simpson')
    with pytest.raises(NotImplementedError):
        cp.delay(cp.array([1.0, 2.0]))  # type: ignore[call-overload]
