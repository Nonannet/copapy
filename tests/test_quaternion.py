import math
from copapy import quaternion, tensor, vector, Target
import copapy as cp
import pytest


def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    if isinstance(a, tensor) and isinstance(b, tensor):
        return all(isclose(av, bv, rel_tol=rel_tol, abs_tol=abs_tol) for av, bv in zip(a.values, b.values))
    if isinstance(a, tensor):
        return all(isclose(av, b, rel_tol=rel_tol, abs_tol=abs_tol) for av in a.values)
    if isinstance(b, tensor):
        return all(isclose(a, bv, rel_tol=rel_tol, abs_tol=abs_tol) for bv in b.values)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def test_identity():
    q = quaternion.identity()
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    assert q.w == 1.0


def test_constructor_default():
    q = quaternion()
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    assert q.w == 1.0


def test_constructor_with_values():
    q = quaternion(1.0, 2.0, 3.0, 4.0)
    assert q.x == 2.0
    assert q.y == 3.0
    assert q.z == 4.0
    assert q.w == 1.0


def test_from_euler_90_roll():
    q = quaternion.from_euler(math.pi / 2, 0.0, 0.0)
    assert isclose(q.w, math.sqrt(2) / 2)
    assert isclose(q.x, math.sqrt(2) / 2)


def test_from_euler_90_pitch():
    q = quaternion.from_euler(0.0, math.pi / 2, 0.0)
    assert isclose(q.w, math.sqrt(2) / 2)
    assert isclose(q.y, math.sqrt(2) / 2)


def test_from_euler_90_yaw():
    q = quaternion.from_euler(0.0, 0.0, math.pi / 2)
    assert isclose(q.w, math.sqrt(2) / 2)
    assert isclose(q.z, math.sqrt(2) / 2)


def test_normalize():
    q = quaternion(0.0, 2.0, 0.0, 0.0)
    n = q.normalize()
    assert isclose(n.x, 1.0)
    assert n.y == 0.0
    assert n.z == 0.0
    assert n.w == 0.0


def test_normalize_identity():
    q = quaternion.identity().normalize()
    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == 0.0
    assert isclose(q.w, 1.0)


def test_to_rotation_matrix_identity():
    q = quaternion.identity()
    m = q.toRotationMatrix()
    expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for i in range(4):
        for j in range(4):
            assert isclose(m[i, j], expected[i][j])


def test_to_euler_roundtrip():
    roll, pitch, yaw = math.pi / 4, math.pi / 6, math.pi / 3
    q = quaternion.from_euler(roll, pitch, yaw)
    r, p, y = q.toEulerAngles()
    assert isclose(r, roll)
    assert isclose(p, pitch)
    assert isclose(y, yaw)


def test_conjugate():
    q = quaternion(4.0, 1.0, 2.0, 3.0)
    c = q.conjugate()
    assert c.x == -1.0
    assert c.y == -2.0
    assert c.z == -3.0
    assert c.w == 4.0


def test_inverse_identity():
    q = quaternion.identity()
    inv = q.inverse()
    assert isclose(inv.x, 0.0)
    assert isclose(inv.y, 0.0)
    assert isclose(inv.z, 0.0)
    assert isclose(inv.w, 1.0)


def test_inverse_product():
    q = quaternion(4.0, 1.0, 2.0, 3.0)
    inv = q.inverse()
    result = q @ inv
    assert isclose(result.x, 0.0, abs_tol=1e-6)
    assert isclose(result.y, 0.0, abs_tol=1e-6)
    assert isclose(result.z, 0.0, abs_tol=1e-6)
    assert isclose(result.w, 1.0, abs_tol=1e-6)


def test_norm_identity():
    q = quaternion.identity()
    assert isclose(abs(q), 1.0)


def test_norm_unit():
    q = quaternion(0.0, 1.0, 0.0, 0.0)
    assert isclose(abs(q), 1.0)


def test_negation():
    q = quaternion(4.0, 1.0, 2.0, 3.0)
    assert (-q).x == -1.0
    assert (-q).y == -2.0
    assert (-q).z == -3.0
    assert (-q).w == -4.0


def test_add():
    q1 = quaternion(0.0, 1.0, 0.0, 0.0)
    q2 = quaternion(0.0, 0.0, 1.0, 0.0)
    s = q1 + q2
    assert s.x == 1.0
    assert s.y == 1.0
    assert s.z == 0.0
    assert s.w == 0.0


def test_add_scalar():
    q = quaternion(0.0, 1.0, 0.0, 0.0)
    s = q + 1.0
    assert s.x == 2.0
    assert s.y == 1.0
    assert s.z == 1.0
    assert s.w == 1.0


def test_sub():
    q1 = quaternion(0.0, 1.0, 1.0, 0.0)
    q2 = quaternion(0.0, 0.0, 1.0, 0.0)
    s = q1 - q2
    assert s.x == 1.0
    assert s.y == 0.0


def test_sub_scalar():
    q = quaternion(2.0, 2.0, 2.0, 2.0)
    s = q - 1.0
    assert s.x == 1.0
    assert s.y == 1.0
    assert s.z == 1.0
    assert s.w == 1.0


def test_mul_scalar():
    q = quaternion(4.0, 1.0, 2.0, 3.0)
    m = q * 2.0
    assert m.x == 2.0
    assert m.y == 4.0
    assert m.z == 6.0
    assert m.w == 8.0


def test_rmul_scalar():
    q = quaternion(4.0, 1.0, 2.0, 3.0)
    m = 2.0 * q
    assert m.x == 2.0
    assert m.y == 4.0
    assert m.z == 6.0
    assert m.w == 8.0


def test_matmul():
    q1 = quaternion(0.0, 1.0, 0.0, 0.0)  # i
    q2 = quaternion(0.0, 0.0, 1.0, 0.0)  # j
    m = q1 @ q2

    assert isclose(m.x, 0.0)
    assert isclose(m.y, 0.0)
    assert isclose(m.z, 1.0)
    assert isclose(m.w, 0.0)


def test_div():
    q = quaternion(8.0, 2.0, 4.0, 6.0)
    d = q / 2.0
    assert d.x == 1.0
    assert d.y == 2.0
    assert d.z == 3.0
    assert d.w == 4.0


def test_to_axis_angle_identity():
    q = quaternion.identity()
    axis, angle = q.toAxisAngle()
    assert isclose(angle, 0.0)
    assert isclose(axis[0], 1.0)
    assert isclose(axis[1], 0.0)
    assert isclose(axis[2], 0.0)


def test_to_axis_angle_90_degrees():
    q = quaternion.from_euler(math.pi / 2, 0.0, 0.0)
    axis, angle = q.toAxisAngle()
    assert isclose(angle, math.pi / 2)
    assert isclose(axis[0], 1.0)
    assert isclose(axis[1], 0.0)
    assert isclose(axis[2], 0.0)


def test_rotate_vector_identity():
    from copapy import vector
    q = quaternion.identity()
    v = vector([1.0, 2.0, 3.0])
    rotated = q.rotate_vector(v)
    assert isclose(rotated[0], 1.0)
    assert isclose(rotated[1], 2.0)
    assert isclose(rotated[2], 3.0)


def test_rotate_vector_90_degrees_x():
    from copapy import vector
    q = quaternion.from_euler(math.pi / 2, 0.0, 0.0)
    v = vector([0.0, 1.0, 0.0])
    rotated = q.rotate_vector(v)
    assert isclose(rotated[0], 0.0, abs_tol=1e-9)
    assert isclose(rotated[1], 0.0, abs_tol=1e-9)
    assert isclose(rotated[2], 1.0, abs_tol=1e-9)


def test_rotate_vector_roundtrip():
    from copapy import vector
    q = quaternion.from_euler(math.pi / 4, math.pi / 6, math.pi / 3)
    v = vector([1.0, 0.5, 0.25])
    rotated = q.rotate_vector(v)
    q_inv = q.inverse()
    restored = q_inv.rotate_vector(rotated)
    for i in range(3):
        assert isclose(restored[i], v[i], abs_tol=1e-9)


def test_satellite_attitude_correction():
    current_q = quaternion.from_euler(math.pi / 8, math.pi / 6, 0.0)
    desired_q = quaternion.from_euler(cp.value(-math.pi / 8), cp.value(math.pi / 3), cp.value(math.pi / 4))
    solar_panel_normal = vector([0.0, 0.0, 1.0])

    rotation_q = desired_q @ current_q.inverse()
    rotated_normal = rotation_q.rotate_vector(solar_panel_normal)

    expected_current = quaternion.from_euler(math.pi / 8, math.pi / 6, 0.0)
    expected_desired = quaternion.from_euler(-math.pi / 8, math.pi / 3, math.pi / 4)
    expected_rotation = expected_desired @ expected_current.inverse()
    expected_rotated = expected_rotation.rotate_vector(solar_panel_normal)

    tg = Target()
    tg.compile(rotation_q, rotated_normal)
    tg.run()

    result_q = tg.read_value(rotation_q)
    result_normal = tg.read_value(rotated_normal)

    print(rotation_q,result_q)

    assert isclose(result_q[0], expected_rotation.w, abs_tol=1e-6)
    assert isclose(result_q[1], expected_rotation.x, abs_tol=1e-6)
    assert isclose(result_q[2], expected_rotation.y, abs_tol=1e-6)
    assert isclose(result_q[3], expected_rotation.z, abs_tol=1e-6)
    assert isclose(result_normal[0], expected_rotated[0], abs_tol=1e-6)
    assert isclose(result_normal[1], expected_rotated[1], abs_tol=1e-6)
    assert isclose(result_normal[2], expected_rotated[2], abs_tol=1e-6)


def test_sensor_fusion():
    # Based on Sebastian O. H. Madgwick's sensor fusion algorithm for orientation estimation.
    # https://x-io.co.uk/open-source-imu-and-ahrs-algorithms

    def update_orientation(q: quaternion, gyro: vector[float], accel: vector[float], dt: float = 0.01):
        # Compute the cost function and its gradient
        objective = q.rotate_vector(vector([0.0, 0.0, 1.0])) - accel.normalize()
        cost = 0.5 * objective.dot(objective)
        gradient = cp.grad(cost, q).normalize()

        # Quaternion derivative from gyroscope measurements
        gyro_quat = cp.quaternion(0.0, *gyro)
        q_dot_gyro = 0.5 * (q @ gyro_quat)

        # Update quaternion using gradient descent
        q_dot = q_dot_gyro - 0.1 * gradient
        
        return (q + q_dot * dt).normalize()
    
    q: quaternion = quaternion(cp.value(0.7071), cp.value(0.7071), cp.value(0.0), cp.value(0.0))  # Initial orientation (45 degrees around X-axis)
    gyro = vector([0.01, 0.02, 0.015])
    accel = vector([0.0, 0.0, 1.0])

    new_q = update_orientation(q, gyro, accel)

    tg = Target()
    tg.compile(new_q)
    tg.run()

    new_q_value = tg.read_value(new_q)

    assert pytest.approx(new_q_value[0], abs=1e-4) == 0.7072948217391968
    assert pytest.approx(new_q_value[1], abs=1e-4) == 0.7069186568260193
    assert pytest.approx(new_q_value[2], abs=1e-4) == 0.7660913727013394e-05
    assert pytest.approx(new_q_value[3], abs=1e-4) == 0.00012362639245111495