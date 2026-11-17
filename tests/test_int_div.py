"""Int floor division and modulo: rounded towards -inf like in Python. The
division instruction is never executed with the operands it traps for (x / 0
and INT_MIN / -1 on x86): a constant divisor of 0 raises ZeroDivisionError at
trace time, with a variable divisor x // 0 is 0 and x % 0 is x."""
from typing import Any

import pytest

import copapy as cp
from copapy._basic_types import value_from_number

INT_MIN, INT_MAX = -2**31, 2**31 - 1

DIVIDENDS = [0, 1, -1, 7, -7, 8, -8, 100, -100, 12345, -12345, INT_MAX, INT_MIN, INT_MIN + 1]
DIVISORS = [1, -1, 2, -2, 3, -3, 7, -7, 8, -8, 1024, 100, -100, INT_MAX, INT_MIN]


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def wrap(x: int) -> int:
    return (x + 2**31) % 2**32 - 2**31


def floordiv_ref(a: int, b: int) -> int:
    return 0 if b == 0 else wrap(a // b)


def mod_ref(a: int, b: int) -> int:
    return a if b == 0 else a % b


@pytest.mark.parametrize('b', DIVISORS + [0])
def test_variable_divisor(b: int) -> None:
    """Divisor written by the host: no trap for 0 and INT_MIN / -1"""
    exprs = [e for a in DIVIDENDS for e in (cp.value(a) // cp.value(b), cp.value(a) % cp.value(b))]
    assert [e.net.source.name for e in exprs[:2]] == ['floordiv_int_int', 'mod_int_int']
    ref = [r for a in DIVIDENDS for r in (floordiv_ref(a, b), mod_ref(a, b))]
    assert evaluate(*exprs) == ref


@pytest.mark.parametrize('b', DIVISORS)
def test_constant_divisor(b: int) -> None:
    exprs = [e for a in DIVIDENDS for e in (cp.value(a) // b, cp.value(a) % b)]
    assert all(e.dtype == 'int' for e in exprs)
    ref = [r for a in DIVIDENDS for r in (floordiv_ref(a, b), mod_ref(a, b))]
    assert evaluate(*exprs) == ref


def test_reflected_operators() -> None:
    """Python number as dividend, the divisor is a variable"""
    b = cp.value(-3)
    assert evaluate(7 // b, 7 % b, 7 // cp.value(0), 7 % cp.value(0)) == [-3, -2, 0, 7]


def test_constant_divisor_ops() -> None:
    """Divisions by 1, -1 and powers of 2 need no division"""
    x = cp.value(-7)

    def ops(d: int) -> tuple[str, str]:
        return (x // d).net.source.name, (x % d).net.source.name

    assert ops(1) == ('const_int', 'const_int')
    assert ops(-1) == ('neg_int', 'const_int')
    assert ops(8) == ('rshift_int_int', 'bwand_int_int')
    assert ops(3) == ('floordivc_int_int', 'modc_int_int')
    assert ops(-8) == ('floordivc_int_int', 'modc_int_int')
    assert (x // value_from_number(4)).net.source.name == 'rshift_int_int'  # Anonymous constant


@pytest.mark.parametrize('divisor', [0, value_from_number(0)])
def test_constant_zero_divisor_raises(divisor: Any) -> None:
    x = cp.value(7)
    with pytest.raises(ZeroDivisionError):
        x // divisor
    with pytest.raises(ZeroDivisionError):
        x % divisor


def test_float_operands_unchanged() -> None:
    exprs = [cp.value(7.5) // 2, cp.value(7) // 2.0, cp.value(-7) // cp.value(2.0)]
    assert all(e.dtype == 'float' for e in exprs)
    assert evaluate(*exprs) == [3.0, 3.0, -4.0]


def test_grad_through_int_division() -> None:
    """Int division results are constant for the derivative"""
    x = cp.value(2.0)
    i = cp.value(9)
    y = x * (i // 4) + x * (i // 3) + x * (i % 4)
    dx, = cp.grad(y, [x])
    assert evaluate(dx) == pytest.approx([2 + 3 + 1])  # pyright: ignore[reportUnknownMemberType]
