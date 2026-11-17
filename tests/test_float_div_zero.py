"""Float division by zero never traps at runtime: x / 0 and x // 0 give inf,
-inf or nan as in IEEE 754 (Python would raise ZeroDivisionError). This also
holds for int operands of a true division and for a constant zero divisor."""
import math
from typing import Any

import pytest

import copapy as cp

DIVIDENDS = [1.0, -1.0, 0.0, 2.5, -1e30, 1, -1, 0, 2**31 - 1, -2**31]


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def div_ref(a: float, b: float) -> float:
    """IEEE 754 result of a / b and a // b for b == 0"""
    if a == 0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def check(results: list[Any], refs: list[float]) -> None:
    assert len(results) == len(refs)
    for res, ref in zip(results, refs):
        assert isinstance(res, float)
        assert math.isnan(res) if math.isnan(ref) else res == ref


@pytest.mark.parametrize('zero', [0.0, -0.0, 0])
def test_variable_zero_divisor(zero: float) -> None:
    """Divisor written by the host"""
    exprs = [e for a in DIVIDENDS for e in (cp.value(a) / cp.value(zero), cp.value(a) // cp.value(float(zero)))]
    assert all(e.dtype == 'float' for e in exprs)
    check(evaluate(*exprs), [div_ref(a, zero) for a in DIVIDENDS for _ in range(2)])


@pytest.mark.parametrize('zero', [0.0, -0.0, 0])
def test_constant_zero_divisor(zero: float) -> None:
    """A constant zero divisor of a float result is no error at trace time"""
    exprs = [cp.value(a) / zero for a in DIVIDENDS] + [cp.value(float(a)) // zero for a in DIVIDENDS]
    check(evaluate(*exprs), [div_ref(a, zero) for a in DIVIDENDS] * 2)


@pytest.mark.parametrize('zero', [0.0, -0.0, 0])
def test_reflected_zero_divisor(zero: float) -> None:
    """Python number as dividend, the zero divisor is a variable"""
    exprs = [a / cp.value(zero) for a in DIVIDENDS] + [float(a) // cp.value(zero) for a in DIVIDENDS]
    check(evaluate(*exprs), [div_ref(a, zero) for a in DIVIDENDS] * 2)


def test_zero_divisor_computed_at_runtime() -> None:
    """The divisor becomes zero only at runtime, the run continues after it"""
    x, y = cp.value(3.0), cp.value(3)
    d = x - 3.0
    exprs = [x / d, x // d, d / d, y / (y - 3), -x / d + 1.0, (d / d) * 2.0]
    check(evaluate(*exprs), [math.inf, math.inf, math.nan, math.inf, -math.inf, math.nan])
