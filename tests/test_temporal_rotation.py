"""Tests for integrate_rotation and angular_rate of orientations as quaternions"""
import math
from typing import Any, Sequence

import pytest

import copapy as cp

Quat = tuple[float, float, float, float]
Vec = Sequence[float]


def approx(v: object, tol: float = 2e-5, rel: float | None = None) -> object:
    return pytest.approx(v, rel=rel, abs=tol)  # pyright: ignore[reportUnknownMemberType]


def qmul(a: Quat, b: Quat) -> Quat:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qexp(theta: Vec) -> Quat:
    """Rotation by the rotation vector theta"""
    angle = math.sqrt(sum(c * c for c in theta)) / 2
    s = math.sin(angle) / angle / 2 if angle else 0.5
    return (math.cos(angle), theta[0] * s, theta[1] * s, theta[2] * s)


def rotate(q: Quat, theta: Vec, frame: str) -> Quat:
    return qmul(q, qexp(theta)) if frame == 'body' else qmul(qexp(theta), q)


def same_orientation(a: Sequence[float], b: Quat) -> Sequence[float]:
    """a with the sign of b"""
    return a if sum(x * y for x, y in zip(a, b)) >= 0 else [-x for x in a]


Q0: Quat = (math.cos(0.3), math.sin(0.3), 0.0, 0.0)


def run_rotation(make: Any, inputs: list[Sequence[float]], period: float) -> list[Any]:
    """Results of a periodic task computing make(input values) for each input"""
    values = [cp.value(0.0) for _ in inputs[0]]
    out = make(values)
    tg = cp.LocalTarget(tick=period)
    tg.deploy(cp.Task(out, period=period))
    results: list[Any] = []
    for k, xs in enumerate(inputs):
        for v, x in zip(values, xs):
            tg.write_value(v, x)
        tg.step((k + 1) * period)
        results.append(tg.read_value(out).values)
    return results


@pytest.mark.parametrize('frame', ['body', 'world'])
@pytest.mark.parametrize('method', ['forward', 'backward', 'trapezoid'])
def test_integrate_constant_rate(method: str, frame: str) -> None:
    # Exact for a constant angular velocity
    omega, period, steps = (0.3, -1.2, 2.0), 0.01, 40
    results = run_rotation(lambda w: cp.integrate_rotation(cp.vector(w), init=cp.quaternion(*Q0),
                                                  method=method, frame=frame),
                           [omega] * steps, period)
    for k, q in enumerate(results):
        expected = rotate(Q0, [c * period * k for c in omega], frame)
        assert q == approx(expected), k


@pytest.mark.parametrize('frame', ['body', 'world'])
def test_integrate_changing_rate(frame: str) -> None:
    # Reference: rotation with the angular velocity changing linearly between
    # the samples, integrated with fine steps
    period, steps, fine = 0.05, 20, 400
    omegas = [(4.0 * math.sin(0.5 * k), 3.0 * math.cos(0.3 * k), 2.0 + math.sin(0.7 * k)) for k in range(steps)]
    reference = [Q0]
    for k in range(1, steps):
        q = reference[-1]
        for i in range(fine):
            t = (i + 0.5) / fine
            w = [a + (b - a) * t for a, b in zip(omegas[k - 1], omegas[k])]
            q = rotate(q, [c * period / fine for c in w], frame)
        reference.append(q)

    def error(method: str) -> float:
        results = run_rotation(lambda w: cp.integrate_rotation(cp.vector(w), init=cp.quaternion(*Q0),
                                                      method=method, frame=frame),
                               omegas, period)
        return max(abs(a - b) for q, r in zip(results, reference) for a, b in zip(q, r))

    trapezoid_error = error('trapezoid')
    assert trapezoid_error < 2e-4
    assert trapezoid_error * 10 < error('forward')


@pytest.mark.parametrize('frame', ['body', 'world'])
def test_rate(frame: str) -> None:
    # Orientations rotated by the angular velocities (incl. a tiny and no
    # rotation), every second one with flipped sign
    period = 0.01
    omegas = [(0.0, 0.0, 0.0), (1.0, -2.0, 0.5), (0.01, 0.0, 0.0), (0.0, 0.0, 0.0),
              (-3.0, 0.2, 4.0), (100.0, 50.0, -80.0)]
    orientations: list[Quat] = []
    q = Q0
    for k, w in enumerate(omegas):
        q = rotate(q, [c * period for c in w], frame) if k else q
        orientations.append(q if k % 2 == 0 else (-q[0], -q[1], -q[2], -q[3]))
    results = run_rotation(lambda v: cp.angular_rate(cp.quaternion(*v), frame=frame), orientations, period)
    assert results == [approx(w, 1e-4, 1e-3) for w in omegas]


@pytest.mark.parametrize('frame', ['body', 'world'])
def test_rate_inverts_integrate(frame: str) -> None:
    omegas = [(0.5 * k, 1.0 - 0.3 * k, -0.2 * k * k) for k in range(10)]

    def make(w: list[cp.value[float]]) -> cp.vector[float]:
        q = cp.integrate_rotation(cp.vector(w), init=cp.quaternion(*Q0), method='backward', frame=frame)
        return cp.angular_rate(q, frame=frame)

    results = run_rotation(make, omegas, 0.01)
    assert results[0] == (0.0, 0.0, 0.0)
    assert results[1:] == [approx(w, 1e-4, 1e-3) for w in omegas[1:]]


def test_integrate_with_dt_and_sequence() -> None:
    q = cp.integrate_rotation([0.0, 0.0, cp.value(math.pi)], dt=0.5)
    tg = cp.LocalTarget()
    tg.deploy(q)
    tg.exec()
    tg.exec()
    expected = qexp((0.0, 0.0, math.pi / 2))
    assert same_orientation(tg.read_value(q).values, expected) == approx(expected)


@pytest.mark.parametrize('angle', [0.0, 1e-6, 0.5, 2.0, 3.1])
def test_large_rotation_per_step(angle: float) -> None:
    # The series of the rotation is accurate up to rotations of pi per step
    axis = (2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0)
    period, steps = 0.01, 5
    omega = [c * angle / period for c in axis]
    results = run_rotation(lambda w: cp.integrate_rotation(cp.vector(w), method='backward'), [omega] * steps, period)
    for k, q in enumerate(results):
        expected = qexp([c * angle * k for c in axis])
        assert same_orientation(q, expected) == approx(expected, 1e-5), k
    rates = run_rotation(lambda w: cp.angular_rate(cp.integrate_rotation(cp.vector(w), method='backward')),
                         [omega] * steps, period)
    assert rates[1:] == [approx(omega, 1e-3, 1e-4)] * (steps - 1)


@pytest.mark.parametrize('max_angle, angle', [(1e-4, 1e-4), (0.1, 0.1), (0.1, 0.05), (1.0, 1.0),
                                              (2 * math.pi, 6.0)])
def test_max_angle(max_angle: float, angle: float) -> None:
    axis = (0.0, 0.6, -0.8)
    period, steps = 0.01, 5
    omega = [c * angle / period for c in axis]
    results = run_rotation(lambda w: cp.integrate_rotation(cp.vector(w), method='backward', max_angle=max_angle),
                           [omega] * steps, period)
    for k, q in enumerate(results):
        expected = qexp([c * angle * k for c in axis])
        assert same_orientation(q, expected) == approx(expected, 2e-6), k


def test_max_angle_operations() -> None:
    from copapy._compiler import get_dag_stats

    def operations(max_angle: float) -> int:
        q = cp.integrate_rotation(cp.vector([cp.value(0.1), cp.value(0.2), cp.value(0.3)]), max_angle=max_angle)
        stats = get_dag_stats([v.net for v in q.values if isinstance(v, cp.value)])
        return sum(n for op, n in stats.items() if not op.startswith(('const', 'delay', 'dt')))

    assert operations(0.1) == operations(math.pi) - 16
    # One term: cos and sin(a) / a are 1, the angle and the products
    # with them are eliminated (series 8, a^2 5, scaling 3, rotation 4)
    assert operations(1e-4) == operations(0.1) - 20


def test_normalized_init() -> None:
    q = cp.integrate_rotation(cp.vector([cp.value(0.0), 0.0, 0.0]), init=cp.quaternion(2.0, 0.0, 0.0, 2.0))
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task(q, period=0.01))
    tg.step(0.05)
    assert tg.read_value(q).values == approx((0.5 ** 0.5, 0.0, 0.0, 0.5 ** 0.5))


def test_errors() -> None:
    q = cp.quaternion(cp.value(1.0), 0.0, 0.0, 0.0)
    w = cp.vector([cp.value(1.0), 0.0, 0.0])
    with pytest.raises(TypeError, match="integrate_rotation"):
        cp.integrate(q)
    with pytest.raises(TypeError, match="angular_rate"):
        cp.rate(q)
    with pytest.raises(TypeError, match="angular velocity must be a vector"):
        cp.integrate_rotation(q)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="3 elements"):
        cp.integrate_rotation(cp.vector([cp.value(1.0), 0.0]))
    with pytest.raises(TypeError, match="quaternion of numbers"):
        cp.integrate_rotation(w, init=q)
    with pytest.raises(ValueError, match="not be zero"):
        cp.integrate_rotation(w, init=cp.quaternion(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="method"):
        cp.integrate_rotation(w, method='rk4')
    with pytest.raises(ValueError, match="frame"):
        cp.angular_rate(q, frame='inertial')
    with pytest.raises(ValueError, match="max_angle"):
        cp.integrate_rotation(w, max_angle=0.0)
    with pytest.raises(ValueError, match="max_angle"):
        cp.integrate_rotation(w, max_angle=7.0)
    with pytest.raises(TypeError, match="must be a quaternion"):
        cp.angular_rate(w)  # type: ignore[arg-type]
