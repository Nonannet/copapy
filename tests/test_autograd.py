from copapy import value, grad
import copapy as cp
import pytest


def test_autograd():
    # Validate against micrograd results from Andrej Karpathy
    # https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
    a = value(-4.0)
    b = value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + cp.relu(b + a)
    d += 3 * d + cp.relu(b - a)
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

    assert pytest.approx(dg[0], abs=1e-4) == 138.83381  # pyright: ignore[reportUnknownMemberType]
    assert pytest.approx(dg[1], abs=1e-4) == 645.57725  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    test_autograd()
