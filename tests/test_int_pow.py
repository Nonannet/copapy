"""Power with int operands: an exact int result (wrapping around like int
multiplication on overflow), with float operands a float result"""
from typing import Any

import pytest

import copapy as cp


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def int_pow_ref(x: int, y: int) -> int:
    """1 / x^-y truncated towards zero for negative exponents, 32 bit wrap around"""
    if y < 0:
        r = 1 if x == 1 else (-1 if y % 2 else 1) if x == -1 else 0
    else:
        r = x ** y
    return (r + 2**31) % 2**32 - 2**31


CASES = [(2, 10), (3, 17), (7, 10), (11, 8), (-3, 9), (5, 0), (0, 0), (0, 5), (2, 31), (3, 40),
         (2, -1), (1, -5), (-1, -3), (-1, -4), (0, -2), (-7, -1)]


@pytest.mark.parametrize(('x', 'y'), CASES)
def test_value_int_pow(x: int, y: int) -> None:
    exprs = [cp.value(x) ** cp.value(y), cp.value(x) ** y, x ** cp.value(y)]
    assert all(e.dtype == 'int' for e in exprs)
    assert evaluate(*exprs) == [int_pow_ref(x, y)] * 3


@pytest.mark.parametrize(('x', 'y'), [(x, y) for x, y in CASES if y < 0 or abs(x ** y) < 2**31])
def test_constants_like_stencil(x: int, y: int) -> None:
    """Constant folding gives the result of the stencil (without overflow)"""
    assert cp.pow(x, y) == int_pow_ref(x, y)
    assert isinstance(cp.pow(x, y), int)


def test_float_operands_give_float() -> None:
    exprs = [cp.value(2.0) ** cp.value(3), cp.value(2) ** cp.value(0.5), cp.value(2) ** 1.5, cp.value(2.0) ** 0]
    assert all(e.dtype == 'float' for e in exprs)
    assert evaluate(*exprs) == pytest.approx([8.0, 2 ** 0.5, 2 ** 1.5, 1.0], rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(cp.pow(2.0, 9), float) and isinstance(cp.pow(2, 0.5), float)


def test_arrays_tensors_vectors() -> None:
    bases, exps = [2, 3, -3, 7], [10, 17, 9, -1]
    ref = [int_pow_ref(x, y) for x, y in zip(bases, exps)]
    ai, ae = cp.array([cp.value(b) for b in bases]), cp.array([cp.value(e) for e in exps])
    t = cp.tensor([cp.value(b) for b in bases], packed=True)
    v = cp.vector([cp.value(b) for b in bases], packed=True)
    exprs: list[Any] = [ai ** ae, ai ** 9, 3 ** ae, t ** 9, v ** 9, t ** cp.tensor(exps), ai ** 2.0]
    assert [e.dtype for e in exprs] == ['int'] * 6 + ['float']
    res = evaluate(*exprs)
    assert res[0] == ref
    assert list(res[1]) == list(res[3].values) == list(res[4].values) == [int_pow_ref(b, 9) for b in bases]
    assert res[2] == [int_pow_ref(3, e) for e in exps]
    assert list(res[5].values) == ref
    assert res[6] == pytest.approx([float(b * b) for b in bases], rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_grad_with_int_exponent() -> None:
    """The derivative of x**n uses a float power, x**(n - 1) is not truncated"""
    x = cp.value(2.0)
    y = x ** cp.value(-1)
    dx, = cp.grad(y, [x])
    assert evaluate(dx) == pytest.approx([-0.25], rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
