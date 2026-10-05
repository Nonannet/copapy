"""Scalar math functions: compiled results and direct Python evaluation."""
import math
from typing import Any, Callable

import pytest

import copapy as cp
from copapy import value


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sign_ref(x: float) -> int:
    return (x > 0) - (x < 0)


TRIG_VALS = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0,
             -0.0001, -0.1, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -5.5, -6.0, -6.28318530718, -100.0, -1000.0, -100000.0]

ARC_VALS = [-1.0, -0.95, -0.9, -0.7, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

UNARY_CASES: list[tuple[Callable[..., Any], Callable[..., Any], list[float]]] = [
    (cp.sqrt, math.sqrt, [0.0, 0.0001, 0.1, 0.5, 1.0, 2.0, 2.5, 6.25, 100.0, 1000.0, 100000.0]),
    (cp.exp, math.exp, [-10.0, -2.5, -1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.5, 10.0]),
    (cp.log, math.log, [0.0001, 0.1, 0.5, 0.9, 0.999, 1.0, 2.5, math.e, 1000.0, 100000.0]),
    (cp.sin, math.sin, TRIG_VALS),
    (cp.cos, math.cos, TRIG_VALS),
    (cp.tan, math.tan, TRIG_VALS),
    (cp.tanh, math.tanh, [-20.0, -2.5, -1.0, -0.26, -0.25, -0.24, -0.1, -0.0001, 0.0, 0.0001, 0.1, 0.24, 0.25, 0.26, 1.0, 2.5, 20.0]),
    (cp.asin, math.asin, ARC_VALS),
    (cp.acos, math.acos, ARC_VALS),
    (cp.atan, math.atan, ARC_VALS + [-1000.0, -10.0, -2.0, 2.0, 10.0, 1000.0]),
    (cp.abs, abs, [-1000.5, -2.5, -0.0, 0.0, 2.5, 1000.5]),
    (cp.sign, sign_ref, [-1000.5, -2.5, 0.0, 2.5, 1000.5]),
    (cp.relu, lambda x: max(x, 0.0), [-1000.5, -2.5, 0.0, 2.5, 1000.5]),
    (cp.sigmoid, lambda x: 1.0 / (1.0 + math.exp(-x)), [-20.0, -2.5, -1.0, 0.0, 1.0, 2.5, 20.0]),
]


@pytest.mark.parametrize(("cp_func", "ref_func", "args"), UNARY_CASES, ids=[c[0].__name__ for c in UNARY_CASES])
def test_unary_function(cp_func: Callable[..., Any], ref_func: Callable[..., Any], args: list[float]):
    refs = [ref_func(a) for a in args]

    # Called with Python numbers the function is evaluated directly
    for a, ref in zip(args, refs):
        res = cp_func(a)
        assert not isinstance(res, value)
        assert res == pytest.approx(ref, rel=1e-9, abs=1e-12), f"{cp_func.__name__}({a})"  # pyright: ignore[reportUnknownMemberType]

    # Called with a cp.value the function is compiled (float32 precision)
    ret = [cp_func(value(a)) for a in args]
    for a, res, ref in zip(args, evaluate(*ret), refs):
        assert res == pytest.approx(ref, rel=1e-5, abs=1e-5), f"compiled {cp_func.__name__}({a})"  # pyright: ignore[reportUnknownMemberType]


BINARY_CASES: list[tuple[Callable[..., Any], Callable[..., Any], list[tuple[float, float]]]] = [
    (cp.atan2, math.atan2, [(1.0, 3.0), (1.0, -3.0), (-1.0, -3.0), (-1.0, 3.0), (1.0, 0.0), (-1.0, 0.0),
                            (0.0, 3.0), (0.0, -3.0), (0.95, 0.01), (-100.0, 0.5)]),
    (cp.pow, math.pow, [(2.0, 3.0), (2.5, 2.0), (2.5, 0.5), (9.0, -1.0), (2.5, 2.111), (0.5, -2.0), (10.0, 0.0), (0.0, 2.0)]),
    (cp.minimum, min, [(1.0, 2.0), (2.0, 1.0), (-1.5, -1.5), (-3.0, 5.0), (0.0, -0.5)]),
    (cp.maximum, max, [(1.0, 2.0), (2.0, 1.0), (-1.5, -1.5), (-3.0, 5.0), (0.0, -0.5)]),
]


@pytest.mark.parametrize(("cp_func", "ref_func", "args"), BINARY_CASES, ids=[c[0].__name__ for c in BINARY_CASES])
def test_binary_function(cp_func: Callable[..., Any], ref_func: Callable[..., Any], args: list[tuple[float, float]]):
    refs = [ref_func(a, b) for a, b in args]

    for (a, b), ref in zip(args, refs):
        res = cp_func(a, b)
        assert not isinstance(res, value)
        assert res == pytest.approx(ref, rel=1e-9, abs=1e-12), f"{cp_func.__name__}({a}, {b})"  # pyright: ignore[reportUnknownMemberType]

    # Both arguments variable, only the first or only the second one
    ret = [r for a, b in args for r in (cp_func(value(a), value(b)), cp_func(value(a), b), cp_func(a, value(b)))]
    results = evaluate(*ret)
    for i, res in enumerate(results):
        a, b = args[i // 3]
        assert res == pytest.approx(refs[i // 3], rel=1e-5, abs=1e-5), f"compiled {cp_func.__name__}({a}, {b}), variant {i % 3}"  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(("x", "lo", "hi"), [(-2.5, 0.0, 1.0), (0.3, 0.0, 1.0), (2.5, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (-7.0, -5.0, -1.0)])
def test_clamp(x: float, lo: float, hi: float):
    ref = min(max(x, lo), hi)
    assert cp.clamp(x, lo, hi) == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]

    res, = evaluate(cp.clamp(value(x), lo, hi))
    assert res == pytest.approx(ref, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_power_operator():
    a_i = 9
    a_f = 2.5
    c_i = value(a_i)
    c_f = value(a_f)

    ret_test = (c_f ** 2, c_i ** 2, c_i ** -1, c_i ** 0.5, c_i ** 2.111, c_f ** 2.111, c_f ** -2, 2 ** c_f)
    ret_refe = (a_f ** 2, a_i ** 2, a_i ** -1, a_i ** 0.5, a_i ** 2.111, a_f ** 2.111, a_f ** -2, 2 ** a_f)

    for test, ref in zip(ret_test, ret_refe):
        assert isinstance(test, value)
        assert test.dtype == type(ref).__name__

    for test, res, ref in zip(ret_test, evaluate(*ret_test), ret_refe):
        assert res == pytest.approx(ref, rel=1e-5), f"{test}: {res} != {ref}"  # pyright: ignore[reportUnknownMemberType]


def test_result_dtypes():
    """Integer arguments keep the integer type where Python's math does"""
    c_i = value(-9)

    cases = [
        (cp.abs(c_i), int, 9),
        (cp.sign(c_i), int, -1),
        (cp.sign(value(0)), int, 0),
        (cp.minimum(c_i, 5), int, -9),
        (cp.maximum(c_i, 5), int, 5),
        (cp.sqrt(-c_i), float, 3.0),
        (cp.exp(value(0)), float, 1.0),
        (cp.sin(value(0)), float, 0.0),
    ]

    for test, typ, _ in cases:
        assert isinstance(test, value)
        assert test.dtype == typ.__name__

    for (test, typ, ref), res in zip(cases, evaluate(*(c[0] for c in cases))):
        assert isinstance(res, typ), f"{test}: result {res!r} is not of type {typ.__name__}"
        assert res == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]


def test_identities():
    """Combinations of functions that must cancel out"""
    x = 0.7
    c = value(x)

    ret_test = (cp.sin(c) ** 2 + cp.cos(c) ** 2,
                cp.tan(c) - cp.sin(c) / cp.cos(c),
                cp.exp(cp.log(c)),
                cp.log(cp.exp(c)),
                cp.asin(cp.sin(c)),
                cp.acos(cp.cos(c)),
                cp.atan(cp.tan(c)),
                cp.atan2(cp.sin(c), cp.cos(c)),
                cp.sqrt(c) ** 2,
                cp.abs(c) * cp.sign(c))
    ret_refe = (1.0, 0.0, x, x, x, x, x, x, x, x)

    for res, ref in zip(evaluate(*ret_test), ret_refe):
        assert res == pytest.approx(ref, abs=1e-5)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
