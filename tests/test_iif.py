"""Tests for cp.iif: branch-free selection by the mask and masknot stencils."""
import math
from typing import Any

import pytest

import copapy as cp
from copapy.backend import get_dag_stats


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def op_names(result: Any) -> set[str]:
    return {name.split('_')[0] for name in get_dag_stats([result.net]) if not name.startswith('const')}


@pytest.mark.parametrize('cond', [0, 1, -3, 7])
def test_float_results(cond: int) -> None:
    c, x, y = cp.value(cond), cp.value(2.5), cp.value(-4.25)
    res = evaluate(cp.iif(c, x, y), cp.iif(c, x, 8.5), cp.iif(c, 1.5, y), cp.iif(c, 1.5, 8.5))
    assert res == ([2.5, 2.5, 1.5, 1.5] if cond else [-4.25, 8.5, -4.25, 8.5])


@pytest.mark.parametrize('cond', [0.0, 0.5, -2.0])
def test_int_results(cond: float) -> None:
    c, x, y = cp.value(cond), cp.value(7), cp.value(-9)
    results = [cp.iif(c, x, y), cp.iif(c, x, 3), cp.iif(c, 4, y), cp.iif(c, 4, 3)]
    assert all(r.dtype == 'int' for r in results)
    res = evaluate(*results)
    assert res == ([7, 7, 4, 4] if cond else [-9, 3, -9, 3])
    assert all(isinstance(v, int) for v in res)


@pytest.mark.parametrize('cond', [0, 1])
def test_mixed_results_are_float(cond: int) -> None:
    c, i, f = cp.value(cond), cp.value(7), cp.value(2.5)
    results = [cp.iif(c, i, f), cp.iif(c, f, i), cp.iif(c, i, 0.5), cp.iif(c, 3, f)]
    assert all(r.dtype == 'float' for r in results)
    assert evaluate(*results) == ([7.0, 2.5, 7.0, 3.0] if cond else [2.5, 7.0, 0.5, 2.5])


def test_comparison_and_bool_results() -> None:
    a, b = cp.value(3.0), cp.value(5.0)
    res = evaluate(cp.iif(a > b, a, b), cp.iif(a < b, a, b), cp.iif(a == b, 1, 2), cp.iif(a != b, a > b, a < b))
    assert res == [5.0, 3.0, 2, 0]


def test_number_of_operations() -> None:
    a, b = cp.value(3.0), cp.value(5.0)
    # Comparison: negated to the mask, the final add is not counted
    assert get_dag_stats([cp.iif(a > b, a, b).net]) == {
        'const_float': 2, 'gt_float_float': 1, 'neg_int': 1, 'mask_float_int': 1, 'masknot_float_int': 1}
    # Other conditions are compared with zero first
    assert op_names(cp.iif(a, a, b)) == {'ne', 'neg', 'mask', 'masknot'}
    # A constant zero is not masked: the mask stencil is the result
    assert cp.iif(a > b, a, 0.0).net.source.name == 'mask_float_int'
    assert cp.iif(a > b, 0.0, b).net.source.name == 'masknot_float_int'


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_unselected_result_has_no_effect(bad: float) -> None:
    c, x = cp.value(1), cp.value(2.5)
    zero = cp.value(0.0)
    computed = x / zero if not math.isnan(bad) else zero / zero  # inf or nan at runtime
    res = evaluate(cp.iif(c, x, cp.value(bad)), cp.iif(c == 0, cp.value(bad), x), cp.iif(c, x, computed),
                   cp.iif(c, cp.value(bad), x))
    assert res[:3] == [2.5, 2.5, 2.5]
    assert math.isnan(res[3]) if math.isnan(bad) else res[3] == bad


def test_exact_result() -> None:
    """The selected value is passed bit by bit"""
    data = [1e-38, -3.4e38, 0.1, 1.0000001]
    vals = [cp.value(v) for v in data]
    c = cp.value(1)
    res = evaluate(*(cp.iif(c, v, 123.0) for v in vals), *(cp.iif(c == 0, 123.0, v) for v in vals))
    assert res == evaluate(*vals) * 2


def test_condition_known_at_trace_time() -> None:
    x = cp.value(2.5)
    assert cp.iif(1, x, 3.0) is x
    assert cp.iif(0, 3.0, x) is x
    assert cp.iif(3 > 5, 2, 7) == 7 and isinstance(cp.iif(3 > 5, 2, 7), int)
    assert cp.iif(0.0, 2.5, 7) == 7.0 and isinstance(cp.iif(0.0, 2.5, 7), float)
    assert cp.iif(1, 2.5, 7) == 2.5


def test_condition_written_at_runtime() -> None:
    c, x, y = cp.value(0.0), cp.value(2.5), cp.value(-1.5)
    result = cp.iif(c, x, y)
    tg = cp.Target()
    tg.compile(result)
    for cond, expected in [(0.0, -1.5), (1.0, 2.5), (-0.25, 2.5), (0.0, -1.5)]:
        tg.write_value(c, cond)
        tg.run()
        assert tg.read_value(result) == expected


@pytest.mark.parametrize('cond', [0, 1])
def test_grad(cond: int) -> None:
    """The gradient passes to the selected result only"""
    c, x, y = cp.value(cond), cp.value(1.5), cp.value(-2.0)
    result = cp.iif(c, x * x, cp.sin(y)) * 3.0
    res = evaluate(result, *cp.grad(result, [x, y]))
    expected = [6.75, 9.0, 0.0] if cond else [3.0 * math.sin(-2.0), 0.0, 3.0 * math.cos(-2.0)]
    assert res == pytest.approx(expected, rel=1e-5)
