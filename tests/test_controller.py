"""Tests for the PID controller of copapy.controller on the local target."""
from typing import Any

import pytest

import copapy as cp
from copapy.controller import pid


def approx(v: object) -> object:
    return pytest.approx(v, rel=1e-4, abs=1e-5)  # pyright: ignore[reportUnknownMemberType]


def run_sequence(outputs: Any, inputs: dict[cp.value[Any], list[Any]], period: float = 0.01) -> list[Any]:
    """Results of the periodic task computing outputs for each step of the inputs"""
    tg = cp.LocalTarget(tick=period)
    tg.deploy(cp.Task(outputs, period=period))
    results: list[Any] = []
    for k in range(len(next(iter(inputs.values())))):
        for v, xs in inputs.items():
            tg.write_value(v, xs[k])
        tg.step((k + 1) * period)
        results.append(tg.read_value(outputs))
    return results


def reference(rs: list[float], ys: list[float], kp: float, ki: float = 0.0, kd: float = 0.0,
              tf: float = 0.0, b: float = 1.0, c: float = 0.0, out_min: float | None = None,
              out_max: float | None = None, tt: float | None = None, feedforward: float = 0.0,
              default_out: float = 0.0, enable: list[bool] | None = None, dt: float = 0.01) -> list[float]:
    integral, d, x_prev, out = default_out, 0.0, 0.0, []
    for k, (r, y) in enumerate(zip(rs, ys)):
        en = enable[k] if enable else True
        x = c * r - y
        if k:
            d = (tf * d + kd * (x - x_prev)) / (tf + dt)
        x_prev = x
        v = kp * (b * r - y) + d + feedforward
        unlimited = v + integral
        u = unlimited if en else default_out
        if out_min is not None:
            u = max(u, out_min)
        if out_max is not None:
            u = min(u, out_max)
        if ki:
            if not en:
                integral = u - v
            else:
                integral += ki * dt * (r - y)
                if out_min is not None or out_max is not None:
                    integral += dt / (tt if tt is not None else kp / ki) * (u - unlimited)
        out.append(u)
    return out


RS = [1.0] * 6 + [3.0] * 6 + [-2.0] * 8
YS = [0.0, 0.2, 0.5, 0.7, 0.9, 1.0, 1.2, 1.8, 2.5, 2.9, 3.1, 3.0, 2.0, 0.5, -1.0, -1.8, -2.2, -2.1, -2.0, -2.0]
ENABLE = [False] * 3 + [True] * 7 + [False] * 4 + [True] * 6

CASES: list[dict[str, Any]] = [
    dict(kp=2.0),
    dict(kp=2.0, ki=5.0),
    dict(kp=2.0, kd=0.1),
    dict(kp=2.0, ki=5.0, kd=0.1, tf=0.02),
    dict(kp=2.0, ki=5.0, kd=0.1, tf=0.02, b=0.5, c=1.0, feedforward=0.3, default_out=0.2),
    dict(kp=1.5, ki=20.0, out_min=-1.0, out_max=1.0),
    dict(kp=1.5, ki=20.0, out_min=-1.0, out_max=1.0, tt=0.05),
    dict(kp=1.5, ki=20.0, kd=0.05, tf=0.01, out_max=2.0, dt=0.05),
    dict(kp=2.0, default_out=0.5),
]

ENABLE_CASES: list[dict[str, Any]] = [
    dict(kp=2.0, ki=5.0, kd=0.1, tf=0.02, default_out=0.4),
    dict(kp=1.5, ki=20.0, out_min=-1.0, out_max=1.0, default_out=1.5),
    dict(kp=2.0, kd=0.1, default_out=-0.3),
]


@pytest.mark.parametrize('kwargs', CASES)
def test_pid_reference(kwargs: dict[str, Any]) -> None:
    r, y = cp.value(0.0), cp.value(0.0)
    u = pid(r, y, **kwargs)
    results = run_sequence(u, {r: RS, y: YS})
    assert results == approx(reference(RS, YS, **kwargs))


@pytest.mark.parametrize('kwargs', ENABLE_CASES)
def test_pid_enable(kwargs: dict[str, Any]) -> None:
    r, y, en = cp.value(0.0), cp.value(0.0), cp.value(0.0)
    u = pid(r, y, enable=en, **kwargs)
    results = run_sequence(u, {r: RS, y: YS, en: [float(e) for e in ENABLE]})
    assert results == approx(reference(RS, YS, enable=ENABLE, **kwargs))


def test_bumpless_enable() -> None:
    # Constant error: after enabling, the output continues from default_out
    r, y, en = cp.value(1.0), cp.value(0.0), cp.value(False)
    u = pid(r, y, kp=2.0, ki=5.0, enable=en, default_out=0.7)
    results = run_sequence(u, {en: [False] * 3 + [True] * 2})
    assert results == approx([0.7, 0.7, 0.7, 0.7, 0.75])


def test_default_out_value() -> None:
    # default_out as value: initial integral, output while disabled
    r, y, default = cp.value(0.0), cp.value(0.0), cp.value(0.5)
    u = pid(r, y, kp=2.0, ki=5.0, default_out=default)
    results = run_sequence(u, {r: [0.0, 1.0, 1.0], default: [0.5, 0.9, 0.9]})
    assert results == approx([0.5, 2.5, 2.55])


def test_tunable_gains() -> None:
    # Gains as values: changing ki keeps the output (the integral includes ki)
    r, y = cp.value(1.0), cp.value(0.0)
    kp, ki = cp.value(2.0), cp.value(5.0)
    u = pid(r, y, kp, ki)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task(u, period=0.01))
    tg.step(0.05)
    assert tg.read_value(u) == approx(2.0 + 4 * 0.05)
    tg.write_value(ki, 50.0)
    tg.step(0.06)
    assert tg.read_value(u) == approx(2.0 + 4 * 0.05 + 0.05)
    tg.write_value(kp, 0.0)
    tg.step(0.07)
    assert tg.read_value(u) == approx(4 * 0.05 + 0.05 + 0.5)


def test_closed_loop_windup() -> None:
    # First order plant with a limited actuator: back-calculation keeps the
    # overshoot far below the one without anti-windup
    period, tau, kp, ki, limit = 0.01, 0.1, 1.0, 10.0, 1.5
    setpoint = cp.value(1.0)

    def plant(y: cp.value[float]) -> Any:
        u = pid(setpoint, y, kp=kp, ki=ki, out_min=-limit, out_max=limit)
        return y + cp.dt / tau * (u - y)

    y = cp.feedback(plant, init=0.0)
    results = run_sequence(y, {setpoint: [1.0] * 200}, period)
    assert results[-1] == approx(1.0)

    yp, integral, windup = 0.0, 0.0, []
    for _ in range(200):
        u = min(max(kp * (1.0 - yp) + integral, -limit), limit)
        integral += ki * period * (1.0 - yp)
        yp += period / tau * (u - yp)
        windup.append(yp)
    assert max(results) - 1.0 < 0.5 * (max(windup) - 1.0)


def test_errors() -> None:
    r, y = cp.value(0.0), cp.value(0.0)
    with pytest.raises(ValueError, match="tf"):
        pid(r, y, 1.0, kd=1.0, tf=-0.1)
    with pytest.raises(ValueError, match="out_min"):
        pid(r, y, 1.0, out_min=1.0, out_max=0.0)
    with pytest.raises(ValueError, match="tt is required"):
        pid(r, y, 0.0, 1.0, out_max=1.0)
    with pytest.raises(TypeError, match="enable"):
        pid(r, y, 1.0, enable=False)
    with pytest.raises(ValueError, match="no period"):
        cp.LocalTarget().deploy(pid(r, y, 1.0, 1.0))
