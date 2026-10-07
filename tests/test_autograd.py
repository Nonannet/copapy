from copapy import value, grad
import copapy as cp
import pytest
import math


def test_autograd():
    # Validated against micrograd results from Andrej Karpathy
    # https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
    a = value(-4.0)
    b = value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + cp.nn.relu(b + a)
    d += 3 * d + cp.nn.relu(-a + b)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    dg = grad(g, (a, b))

    tg = cp.Target()
    tg.compile(g, dg)
    tg.run()


    print(f"g = {tg.read_value(g)}")
    print(f"dg/da = {tg.read_value(dg[0])}   grad:{dg[0]}    val:{a} = {tg.read_value(a)}")
    print(f"dg/db = {tg.read_value(dg[1])}   grad:{dg[1]}    val:{b} = {tg.read_value(b)}")

    assert tg.read_value(dg[0]) == pytest.approx(138.83381, abs=1e-3)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(dg[1]) == pytest.approx(645.57725, abs=1e-3)  # pyright: ignore[reportUnknownMemberType]


def test_autograd_tanh():
    a = value(0.7)
    b = value(-1.5)
    y = cp.tanh(a * b) + cp.tanh(a)

    dy = grad(y, (a, b))

    tg = cp.Target()
    tg.compile(dy)
    tg.run()

    ref_a = -1.5 * (1 - math.tanh(-1.05) ** 2) + (1 - math.tanh(0.7) ** 2)
    ref_b = 0.7 * (1 - math.tanh(-1.05) ** 2)
    assert tg.read_value(dy[0]) == pytest.approx(ref_a, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(dy[1]) == pytest.approx(ref_b, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def extended_function(a, b, relu, sin, abs):
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + relu(b + a)
    d += 3 * d + relu(b - a)
    e = c - sin(-d)
    f = abs(e**2)
    g = f / 2.0
    g += 10.0 / f
    return g


def test_autograd_extended():
    a = value(-4.0)
    b = value(2.0)
    g = extended_function(a, b, cp.nn.relu, cp.sin, cp.abs)

    dg = grad(g, (a, b))

    tg = cp.Target()
    tg.compile(g, dg)
    tg.run()

    # Reference: central differences in double precision
    def ref(x: float, y: float) -> float:
        return extended_function(x, y, lambda v: max(v, 0.0), math.sin, abs)

    h = 1e-6
    ref_a = (ref(-4.0 + h, 2.0) - ref(-4.0 - h, 2.0)) / (2 * h)
    ref_b = (ref(-4.0, 2.0 + h) - ref(-4.0, 2.0 - h)) / (2 * h)
    assert tg.read_value(dg[0]) == pytest.approx(ref_a, rel=1e-4)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(dg[1]) == pytest.approx(ref_b, rel=1e-4)  # pyright: ignore[reportUnknownMemberType]


def test_autograd_neg():
    x = value(1.5)
    y = value(-2.0)
    terms = [-x, (-x) * (-x), cp.sin(-x), -(-x) * 3.0]
    grads = [grad(t, x) for t in terms]
    grad_xy = grad(-(x * y), (x, y))

    tg = cp.Target()
    tg.compile(grads, grad_xy)
    tg.run()

    expected = [-1.0, 3.0, -math.cos(-1.5), 3.0]
    for dg, ref in zip(grads, expected):
        assert tg.read_value(dg) == pytest.approx(ref, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(grad_xy) == pytest.approx([2.0, -1.5], rel=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_autograd_atan2_min_max():
    a, b = value(0.7), value(-1.3)
    terms = [cp.atan2(a, b), cp.maximum(a, b) * 2.0 + cp.minimum(a, b), cp.sign(a) * b]
    grads = [grad(t, (a, b)) for t in terms]

    tg = cp.Target()
    tg.compile(*grads)
    tg.run()

    denom = 0.7 ** 2 + 1.3 ** 2
    expected = [[-1.3 / denom, -0.7 / denom], [2.0, 1.0], [0.0, 1.0]]
    for dg, ref in zip(grads, expected):
        assert tg.read_value(dg) == pytest.approx(ref, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_autograd()
