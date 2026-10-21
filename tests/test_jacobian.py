import copapy as cp
import pytest
import math

from runner_helpers import trig_tol


def flat(x):
    return [v for row in x for v in (flat(row) if isinstance(row, list) else [row])]


def test_jacobian_vector():
    x = cp.vector([cp.value(0.5), cp.value(-1.2)])
    u = cp.value(0.3)

    # Discrete pendulum-like dynamics
    dt = 0.1
    f = cp.vector([x[0] + dt * x[1],
                   x[1] + dt * (-9.81 * cp.sin(x[0]) + u)])

    a = cp.jacobian(f, x)
    b = cp.jacobian(f, u)
    assert a.shape == (2, 2)
    assert b.shape == (2,)

    tg = cp.Target()
    tg.compile(a, b)
    tg.run()

    ref_a = [[1.0, dt], [-dt * 9.81 * math.cos(0.5), 1.0]]
    assert tg.read_value(a).values == pytest.approx(flat(ref_a), rel=1e-5, abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(b).values == pytest.approx([0.0, dt], rel=1e-5, abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]


def test_jacobian_scalar_is_grad():
    a = cp.value(2.0)
    b = cp.value(3.0)
    y = a * b + cp.sin(a)

    j = cp.jacobian(y, [a, b])

    tg = cp.Target()
    tg.compile(j)
    tg.run()

    assert tg.read_value(j) == pytest.approx([3.0 + math.cos(2.0), 2.0], rel=1e-5, abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]


def test_jacobian_lists_and_constants():
    a = cp.value(2.0)
    b = cp.value(3.0)

    j = cp.jacobian([a * b, a ** 2, 1.0], [a, b])
    assert len(j) == 3 and all(len(row) == 2 for row in j)

    tg = cp.Target()
    tg.compile(j)
    tg.run()

    assert tg.read_value(j[0]) == pytest.approx([3.0, 2.0])  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(j[1]) == pytest.approx([4.0, 0.0])  # pyright: ignore[reportUnknownMemberType]
    assert j[2] == [0.0, 0.0]


def test_jacobian_tensor_shape():
    m = cp.tensor([[cp.value(1.0), cp.value(2.0)], [cp.value(3.0), cp.value(4.0)]])
    v = cp.vector([cp.value(0.5), cp.value(-0.5)])

    j = cp.jacobian(m @ v, m)
    assert j.shape == (2, 2, 2)

    tg = cp.Target()
    tg.compile(j)
    tg.run()

    # d(m @ v)_i / dm_kl = delta_ik * v_l
    ref = [[[0.5, -0.5], [0.0, 0.0]], [[0.0, 0.0], [0.5, -0.5]]]
    assert tg.read_value(j).values == pytest.approx(flat(ref))  # pyright: ignore[reportUnknownMemberType]


def test_jacobian_array_backed():
    x = cp.tensor(cp.array([cp.value(1.0), cp.value(2.0), cp.value(3.0)]))
    f = x * x

    j = cp.jacobian(f, x)
    assert isinstance(j, cp.tensor)
    assert j.shape == (3, 3)

    tg = cp.Target()
    tg.compile(j)
    tg.run()

    ref = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]
    assert tg.read_value(j).values == pytest.approx(flat(ref))  # pyright: ignore[reportUnknownMemberType]


def test_jacobian_array_raises():
    x = cp.array([cp.value(1.0), cp.value(2.0)])
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.jacobian(x * x, x)
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.jacobian(x.sum(), x)
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.jacobian(cp.vector([x.sum()]), x)


def test_hessian():
    x = cp.vector([cp.value(1.5), cp.value(-0.5)])
    cost = x[0] ** 2 * x[1] + cp.exp(x[1])

    h = cp.jacobian(cp.grad(cost, x), x)

    tg = cp.Target()
    tg.compile(h)
    tg.run()

    ref = [[2 * -0.5, 2 * 1.5], [2 * 1.5, math.exp(-0.5)]]
    assert tg.read_value(h).values == pytest.approx(flat(ref), rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_grad_unconnected_and_identity():
    a = cp.value(2.0)
    b = cp.value(3.0)

    # b has no influence on a ** 2, a is derived in respect to itself
    g = cp.grad(a ** 2, [a, b])
    assert g[1] == 0.0
    assert cp.grad(a, a) == 1.0
    assert cp.grad(a, [a, b]) == [1.0, 0.0]

    tg = cp.Target()
    tg.compile(g[0])
    tg.run()
    assert tg.read_value(g[0]) == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]
