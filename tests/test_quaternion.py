import math
import operator
from typing import Any, Callable

import pytest

import copapy as cp
from copapy import Target, quaternion, vector


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def approx(values: Any, abs: float = 1e-9) -> Any:
    return pytest.approx(list(values), abs=abs)  # pyright: ignore[reportUnknownMemberType]


def variable_quaternion(w: float, x: float, y: float, z: float) -> quaternion:
    return quaternion(cp.value(w), cp.value(x), cp.value(y), cp.value(z))


def hamilton_ref(p: tuple[float, ...], q: tuple[float, ...]) -> tuple[float, ...]:
    """Hamilton product of (w, x, y, z) tuples"""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


S2 = math.sqrt(2) / 2

Q1 = (4.0, 1.0, 2.0, 3.0)
Q2 = (0.5, -1.5, 2.0, 0.25)


@pytest.mark.parametrize("q", [quaternion(), quaternion.identity()], ids=['default', 'identity'])
def test_identity(q: quaternion):
    assert (q.w, q.x, q.y, q.z) == (1.0, 0.0, 0.0, 0.0)


def test_constructor_with_values():
    q = quaternion(1.0, 2.0, 3.0, 4.0)
    assert q.w == 1.0
    assert q.x == 2.0
    assert q.y == 3.0
    assert q.z == 4.0
    assert q.values == (1.0, 2.0, 3.0, 4.0)
    assert len(q) == 4
    assert list(q) == [1.0, 2.0, 3.0, 4.0]
    assert q[0] == 1.0
    assert q[3] == 4.0


def test_constructor_from_vector():
    q = quaternion(0.0, *vector([1.0, 2.0, 3.0]))
    assert q.values == (0.0, 1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("euler", "expected"),
    [
        ((math.pi / 2, 0.0, 0.0), (S2, S2, 0.0, 0.0)),
        ((0.0, math.pi / 2, 0.0), (S2, 0.0, S2, 0.0)),
        ((0.0, 0.0, math.pi / 2), (S2, 0.0, 0.0, S2)),
        ((math.pi, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ((0.0, 0.0, -math.pi / 2), (S2, 0.0, 0.0, -S2)),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ],
    ids=['roll', 'pitch', 'yaw', 'roll_180', 'yaw_neg', 'zero'],
)
def test_from_euler(euler: tuple[float, float, float], expected: tuple[float, ...]):
    q = quaternion.from_euler(*euler)
    assert q.values == approx(expected)
    assert q.norm() == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


def test_from_euler_is_composition_of_axis_rotations():
    roll, pitch, yaw = 0.3, 0.5, 0.7
    q = quaternion.from_euler(roll, pitch, yaw)
    composed = quaternion.from_euler(0.0, 0.0, yaw) @ quaternion.from_euler(0.0, pitch, 0.0) @ quaternion.from_euler(roll, 0.0, 0.0)
    assert q.values == approx(composed.values)


@pytest.mark.parametrize("euler", [(math.pi / 4, math.pi / 6, math.pi / 3), (0.0, 0.0, 0.0), (-0.3, 0.2, -2.5),
                                   (3.0, -1.2, 0.1), (0.1, 1.5, 0.2)])
def test_to_euler_roundtrip(euler: tuple[float, float, float]):
    q = quaternion.from_euler(*euler)
    assert list(q.toEulerAngles()) == approx(euler)


def test_normalize():
    q = quaternion(0.0, 2.0, 0.0, 0.0).normalize()
    assert q.values == approx((0.0, 1.0, 0.0, 0.0))

    q = quaternion(*Q1)
    n = q.normalize()
    norm = math.sqrt(sum(x * x for x in Q1))
    assert n.norm() == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
    assert n.values == approx([x / norm for x in Q1])

    assert quaternion.identity().normalize().values == approx((1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(("q", "expected"), [((1.0, 0.0, 0.0, 0.0), 1.0), ((0.0, 1.0, 0.0, 0.0), 1.0),
                                             ((1.0, 2.0, 2.0, 4.0), 5.0), (Q1, math.sqrt(30.0))])
def test_norm(q: tuple[float, ...], expected: float):
    assert quaternion(*q).norm() == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]
    assert abs(quaternion(*q)) == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_conjugate():
    assert quaternion(*Q1).conjugate().values == (4.0, -1.0, -2.0, -3.0)
    assert quaternion(*Q1).conjugate().conjugate().values == Q1


def test_negation():
    assert (-quaternion(*Q1)).values == (-4.0, -1.0, -2.0, -3.0)


@pytest.mark.parametrize("q", [Q1, Q2, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -2.0, 0.0)])
def test_inverse(q: tuple[float, ...]):
    qq = quaternion(*q)
    inv = qq.inverse()
    assert (qq @ inv).values == approx((1.0, 0.0, 0.0, 0.0))
    assert (inv @ qq).values == approx((1.0, 0.0, 0.0, 0.0))
    assert inv.inverse().values == approx(q)


def test_inverse_of_unit_quaternion_is_conjugate():
    q = quaternion.from_euler(0.4, -0.2, 1.3)
    assert q.inverse().values == approx(q.conjugate().values)


OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]


@pytest.mark.parametrize("op", OPERATORS, ids=[op.__name__ for op in OPERATORS])
def test_scalar_operators(op: Callable[[Any, Any], Any]):
    s = 2.0
    q = quaternion(*Q1)

    res = op(q, s)
    assert isinstance(res, quaternion)
    assert res.values == approx([op(x, s) for x in Q1])

    if op is not operator.truediv:  # scalar / quaternion is not defined element-wise
        res = op(s, q)
        assert isinstance(res, quaternion)
        assert res.values == approx([op(s, x) for x in Q1])


@pytest.mark.parametrize("op", [operator.add, operator.sub], ids=['add', 'sub'])
def test_quaternion_add_sub(op: Callable[[Any, Any], Any]):
    res = op(quaternion(*Q1), quaternion(*Q2))
    assert isinstance(res, quaternion)
    assert res.values == approx([op(a, b) for a, b in zip(Q1, Q2)])


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        ((0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),    # i j = k
        ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0)),    # j k = i
        ((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),    # k i = j
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, -1.0)),   # j i = -k
        ((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)),   # i i = -1
        ((1.0, 0.0, 0.0, 0.0), Q1, Q1),                                        # identity
        (Q1, Q2, hamilton_ref(Q1, Q2)),
        (Q2, Q1, hamilton_ref(Q2, Q1)),
    ],
    ids=['ij', 'jk', 'ki', 'ji', 'ii', 'identity', 'q1q2', 'q2q1'],
)
def test_hamilton_product(p: tuple[float, ...], q: tuple[float, ...], expected: tuple[float, ...]):
    assert (quaternion(*p) @ quaternion(*q)).values == approx(expected)


def test_hamilton_product_norm_is_multiplicative():
    p, q = quaternion(*Q1), quaternion(*Q2)
    assert (p @ q).norm() == pytest.approx(p.norm() * q.norm())  # pyright: ignore[reportUnknownMemberType]


def test_comparison():
    q = quaternion(*Q1)
    assert (q == quaternion(*Q1)).values == (True, True, True, True)
    assert (q == quaternion(4.0, 1.0, 0.0, 3.0)).values == (True, True, False, True)
    assert (q == Q1).values == (True, True, True, True)
    assert (q != quaternion(4.0, 1.0, 0.0, 3.0)).values == (False, False, True, False)

    with pytest.raises(TypeError):
        bool(q == quaternion(*Q1))

    res, = evaluate(variable_quaternion(*Q1) == quaternion(4.0, 1.0, 0.0, 3.0))
    assert res.values == (True, True, False, True)


def test_map():
    assert quaternion(*Q1).map(lambda x: x * 2 + 1).values == (9.0, 3.0, 5.0, 7.0)


@pytest.mark.parametrize(
    ("euler", "axis", "angle"),
    [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),
        ((math.pi / 2, 0.0, 0.0), (1.0, 0.0, 0.0), math.pi / 2),
        ((0.0, math.pi / 2, 0.0), (0.0, 1.0, 0.0), math.pi / 2),
        ((0.0, 0.0, math.pi / 2), (0.0, 0.0, 1.0), math.pi / 2),
        ((0.0, 0.0, -math.pi / 3), (0.0, 0.0, -1.0), math.pi / 3),
        ((2.5, 0.0, 0.0), (1.0, 0.0, 0.0), 2.5),
    ],
)
def test_to_axis_angle(euler: tuple[float, float, float], axis: tuple[float, ...], angle: float):
    res_axis, res_angle = quaternion.from_euler(*euler).toAxisAngle()
    assert res_angle == pytest.approx(angle, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
    assert list(res_axis) == approx(axis)


def test_to_rotation_matrix_identity():
    m = quaternion.identity().toRotationMatrix()
    assert m.shape == (4, 4)
    assert m.values == approx((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


def test_to_rotation_matrix_yaw_90():
    m = quaternion.from_euler(0.0, 0.0, math.pi / 2).toRotationMatrix()
    assert m.values == approx((0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


def test_to_rotation_matrix_is_orthonormal_and_matches_rotate_vector():
    q = quaternion.from_euler(0.4, -0.7, 2.1)
    m = q.toRotationMatrix()
    r = m[0:3, 0:3]

    assert (r @ r.T).values == approx((1, 0, 0, 0, 1, 0, 0, 0, 1))
    for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -2.0, 1.5]):
        assert list(q.rotate_vector(vector(v))) == approx((r @ cp.tensor(v)).values)


@pytest.mark.parametrize(
    ("euler", "v", "expected"),
    [
        ((0.0, 0.0, 0.0), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ((math.pi / 2, 0.0, 0.0), [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ((0.0, math.pi / 2, 0.0), [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ((0.0, 0.0, math.pi / 2), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ((0.0, 0.0, math.pi), [1.0, 2.0, 3.0], [-1.0, -2.0, 3.0]),
    ],
)
def test_rotate_vector(euler: tuple[float, float, float], v: list[float], expected: list[float]):
    rotated = quaternion.from_euler(*euler).rotate_vector(vector(v))
    assert isinstance(rotated, vector)
    assert list(rotated) == approx(expected)


def test_rotate_vector_matches_cp_rotate_vector():
    q = quaternion.from_euler(0.0, 0.0, 0.8)
    v = vector([1.0, 2.0, 3.0])
    assert list(q.rotate_vector(v)) == approx(cp.rotate_vector(v, vector([0.0, 0.0, 1.0]), 0.8).values)


def test_rotate_vector_properties():
    p = quaternion.from_euler(math.pi / 4, math.pi / 6, math.pi / 3)
    q = quaternion.from_euler(-0.3, 1.1, 0.2)
    v = vector([1.0, 0.5, 0.25])
    rotated = p.rotate_vector(v)

    # Length is preserved
    assert rotated.magnitude() == pytest.approx(v.magnitude())  # pyright: ignore[reportUnknownMemberType]
    # Inverse rotation restores the vector
    assert list(p.inverse().rotate_vector(rotated)) == approx(v.values)
    # Product of quaternions composes rotations
    assert list((p @ q).rotate_vector(v)) == approx(p.rotate_vector(q.rotate_vector(v)).values)
    # q and -q describe the same rotation
    assert list((-p).rotate_vector(v)) == approx(rotated.values)


def quaternion_ops(p: quaternion, q: quaternion) -> dict[str, Any]:
    """Operations evaluated once with constant and once with variable quaternions"""
    pn = p.normalize()
    return {
        'add': p + q,
        'sub': p - q,
        'scale': 2.0 * p / 4.0 - 1.0,
        'neg': -p,
        'conjugate': p.conjugate(),
        'inverse': p.inverse(),
        'matmul': p @ q,
        'norm': p.norm(),
        'normalize': pn,
        'rotate_vector': pn.rotate_vector(vector([1.0, -2.0, 0.5])),
        'to_euler': pn.toEulerAngles(),
        'to_rotation_matrix': pn.toRotationMatrix(),
    }


@pytest.mark.parametrize("name", list(quaternion_ops(quaternion(*Q1), quaternion(*Q2))))
def test_compiled_matches_python(name: str):
    ref = quaternion_ops(quaternion(*Q1), quaternion(*Q2))[name]
    res, = evaluate(quaternion_ops(variable_quaternion(*Q1), variable_quaternion(*Q2))[name])

    if isinstance(ref, (quaternion, vector, cp.tensor)):
        assert type(res) is type(ref)
        assert res.values == pytest.approx(ref.values, rel=1e-5, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]
    else:
        assert res == pytest.approx(ref, rel=1e-5, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_satellite_attitude_correction():
    current_q = quaternion.from_euler(math.pi / 8, math.pi / 6, 0.0)
    desired_q = quaternion.from_euler(cp.value(-math.pi / 8), cp.value(math.pi / 3), cp.value(math.pi / 4))
    solar_panel_normal = vector([0.0, 0.0, 1.0])

    rotation_q = desired_q @ current_q.inverse()
    rotated_normal = rotation_q.rotate_vector(solar_panel_normal)

    expected_desired = quaternion.from_euler(-math.pi / 8, math.pi / 3, math.pi / 4)
    expected_rotation = expected_desired @ current_q.inverse()
    expected_rotated = expected_rotation.rotate_vector(solar_panel_normal)

    result_q, result_normal = evaluate(rotation_q, rotated_normal)

    assert list(result_q) == approx(expected_rotation.values, abs=1e-6)
    assert list(result_normal) == approx(expected_rotated.values, abs=1e-6)

    # The rotation takes the current attitude to the desired one
    assert (expected_rotation @ current_q).values == approx(expected_desired.values)


def madgwick_imu_update(q: quaternion, gyro: vector[float], accel: vector[float], dt: float = 0.01) -> quaternion:
    """Update an orientation quaternion with a single Madgwick IMU step.

    Arguments:
        q: Current orientation quaternion.
        gyro: Gyroscope measurement vector in rad/s.
        accel: Accelerometer measurement vector.
        dt: Integration time step in seconds.

    Returns:
        The updated, normalized orientation quaternion.
    """
    BETA: float = 0.1

    # Compute the cost function and its gradient
    objective = q.rotate_vector(vector([0.0, 0.0, 1.0])) - accel.normalize()
    cost = 0.5 * objective.dot(objective)
    gradient = cp.grad(cost, q).normalize()

    # Quaternion derivative from gyroscope measurements
    gyro_quat = cp.quaternion(0.0, *gyro)
    q_dot_gyro = 0.5 * (q @ gyro_quat)

    # Update quaternion using gradient descent
    q_dot = q_dot_gyro - BETA * gradient

    return (q + q_dot * dt).normalize()


def test_sensor_fusion():
    q: quaternion = quaternion(cp.value(0.7071), cp.value(0.7071), cp.value(0.0), cp.value(0.0))  # Initial orientation (90 degrees around X-axis)
    gyro = vector([0.01, 0.02, 0.015])
    accel = vector([0.0, 0.0, 1.0])

    new_q = madgwick_imu_update(q, gyro, accel)

    new_q_value, = evaluate(new_q)

    assert list(new_q_value) == approx((0.7077782144891159, 0.7064346986591282, 1.7677661414353056e-05, 0.00012374362990047135), abs=1e-6)
    assert math.sqrt(sum(x * x for x in new_q_value)) == pytest.approx(1.0, abs=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_sensor_fusion_converges_to_gravity():
    """Repeated updates with a static accelerometer reading align the z-axis with gravity"""
    q = quaternion.from_euler(cp.value(0.4), cp.value(-0.3), cp.value(0.0))
    gyro = vector([0.0, 0.0, 0.0])
    accel = vector([0.0, 0.0, 1.0])

    for _ in range(40):
        q = madgwick_imu_update(q, gyro, accel, dt=0.1)

    z_axis = q.rotate_vector(vector([0.0, 0.0, 1.0]))
    res, = evaluate(z_axis)
    assert list(res) == approx((0.0, 0.0, 1.0), abs=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
