"""Tests for cp.interp: table lookup by the lerp and bsearch array stencils."""
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


def interp_ref(x: float, xp: list[float], fp: list[float], method: str = 'linear') -> float:
    """Reference in double precision, clamped outside of the grid"""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    i = max(k for k in range(len(xp) - 1) if xp[k] <= x)
    f = (x - xp[i]) / (xp[i + 1] - xp[i])
    if method == 'previous':
        return fp[i]
    if method == 'nearest':
        return fp[i + 1] if f >= 0.5 else fp[i]
    return fp[i] + f * (fp[i + 1] - fp[i])


def op_names(*results: Any) -> set[str]:
    stats = get_dag_stats([r.net for r in results if isinstance(r, cp.value)])
    return {name.split('_')[0] for name in stats}


UNIFORM = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
GRID = [-2.0, -1.5, 0.0, 0.25, 1.0, 3.0]
TABLE = [1.0, -1.0, 4.0, 2.0, 2.0, 0.5]

# Positions below, inside (not at points of the grids, no ties for 'nearest') and above the grids
POSITIONS = [-5.0, -2.0, -1.9, -1.2, -0.3, 0.1, 0.2, 0.3, 0.9, 1.3, 2.9, 3.0, 4.5]


@pytest.mark.parametrize('xp', [UNIFORM, GRID])
@pytest.mark.parametrize('method', ['linear', 'previous', 'nearest'])
def test_scalar(xp: list[float], method: str) -> None:
    results = [cp.interp(cp.value(x), xp, TABLE, method) for x in POSITIONS]
    assert all(isinstance(r, cp.value) and r.dtype == 'float' for r in results)
    assert evaluate(*results) == pytest.approx([interp_ref(x, xp, TABLE, method) for x in POSITIONS], rel=1e-5, abs=1e-6)


def test_uniform_grid_needs_no_search() -> None:
    x = cp.value(0.3)
    uniform, searched = cp.interp(x, UNIFORM, TABLE), cp.interp(x, GRID, TABLE)
    assert uniform.net.source.name == 'lerp_floatarr_float'
    assert op_names(uniform) == {'const', 'sub', 'mul'}
    assert op_names(searched) == {'const', 'bsearch'}
    # No scalar operation for a grid of the element indices
    assert op_names(cp.interp(x, [0.0, 1.0, 2.0], [5.0, 6.0, 8.0])) == {'const'}


def test_exact_at_points() -> None:
    """The values of the table are returned bit by bit at the points of the grid"""
    table = [0.1, 1.0000001, -3.4e38, 1e-38, 7.7, 0.3]
    for xp in (UNIFORM, GRID):
        results = [cp.interp(cp.value(x), xp, table) for x in xp]
        assert evaluate(*results) == evaluate(*(cp.value(v) for v in table))


@pytest.mark.parametrize('xp', [UNIFORM, GRID])
def test_nan_and_inf(xp: list[float]) -> None:
    x = [cp.value(v) for v in (float('nan'), float('inf'), float('-inf'), 1e38, -1e38)]
    res = evaluate(*(cp.interp(v, xp, TABLE) for v in x))
    assert math.isnan(res[0])
    assert res[1:] == [TABLE[-1], TABLE[0], TABLE[-1], TABLE[0]]
    for method in ('previous', 'nearest'):
        res = evaluate(*(cp.interp(v, xp, TABLE, method) for v in x[1:]))
        assert res == [TABLE[-1], TABLE[0], TABLE[-1], TABLE[0]]


@pytest.mark.parametrize('n', [2, 3, 4, 5, 8, 9, 33])
def test_table_sizes(n: int) -> None:
    """Number of steps of the binary search for different numbers of points"""
    xp = [0.3 * i + 0.05 * i * i for i in range(n)]
    fp = [((i * 7) % 11 - 5) * 0.37 for i in range(n)]
    positions = [xp[0] - 1.0] + [0.5 * (a + b) for a, b in zip(xp, xp[1:])] + [0.25 * a + 0.75 * b for a, b in zip(xp, xp[1:])] + [xp[-1] + 1.0]
    result = cp.interp(cp.array(positions), xp, fp)
    assert evaluate(result)[0] == pytest.approx([interp_ref(x, xp, fp) for x in positions], rel=1e-4, abs=1e-5)


@pytest.mark.parametrize('xp', [UNIFORM, GRID])
@pytest.mark.parametrize('method', ['linear', 'previous', 'nearest'])
def test_array(xp: list[float], method: str) -> None:
    result = cp.interp(cp.array(POSITIONS), xp, TABLE, method)
    assert isinstance(result, cp.array) and result.shape == (len(POSITIONS),)
    assert result.net.source.name == 'lerp_floatarr_floatarr'
    assert evaluate(result)[0] == pytest.approx([interp_ref(x, xp, TABLE, method) for x in POSITIONS], rel=1e-5, abs=1e-6)


def test_array_shape_and_int_positions() -> None:
    positions = [[-3, -1, 0], [1, 2, 5]]
    result = cp.interp(cp.array(positions), GRID, TABLE)
    assert result.shape == (2, 3)
    res = evaluate(result)[0]
    assert [v for row in res for v in row] == pytest.approx([interp_ref(x, GRID, TABLE) for row in positions for x in row])


def test_vector_and_tensor() -> None:
    ref = [interp_ref(x, GRID, TABLE) for x in POSITIONS[:6]]
    vec = cp.interp(cp.vector(cp.value(x) for x in POSITIONS[:6]), GRID, TABLE)
    assert isinstance(vec, cp.vector) and op_names(*vec.values) == {'const', 'bsearch'}
    packed = cp.interp(cp.vector((cp.value(x) for x in POSITIONS[:6]), packed=True), GRID, TABLE)
    assert isinstance(packed, cp.vector) and 'lerp' in op_names(*packed.values)
    ten = cp.interp(cp.tensor([cp.value(x) for x in POSITIONS[:6]], (2, 3)), GRID, TABLE)
    assert isinstance(ten, cp.tensor) and ten.shape == (2, 3)
    for result in (vec, packed, ten):
        assert evaluate(*result.values) == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_table_of_values_changed_at_runtime() -> None:
    table = [cp.value(v) for v in TABLE]
    x = cp.value(0.1)
    result = cp.interp(x, GRID, table)
    tg = cp.Target()
    tg.compile(result)
    tg.run()
    assert tg.read_value(result) == pytest.approx(interp_ref(0.1, GRID, TABLE))
    new_table = [v * -2.0 + 1.0 for v in TABLE]
    for tv, v in zip(table, new_table):
        tg.write_value(tv, v)
    tg.write_value(x, 2.0)
    tg.run()
    assert tg.read_value(result) == pytest.approx(interp_ref(2.0, GRID, new_table))


def test_tables_as_arrays_and_vectors() -> None:
    x = cp.value(0.6)
    ref = interp_ref(0.6, GRID, TABLE)
    results = [cp.interp(x, cp.array(GRID), cp.array(TABLE)),
               cp.interp(x, cp.vector(GRID), cp.tensor(TABLE)),
               cp.interp(x, [cp.value(v) for v in GRID], cp.vector(cp.value(v) for v in TABLE)),
               cp.interp(x, cp.array(UNIFORM), [1, 2, 3, 4, 5, 6]),
               cp.interp(0.6, GRID, cp.array(TABLE))]
    res = evaluate(*results)
    assert res[:3] == pytest.approx([ref] * 3) and res[4] == pytest.approx(ref)
    assert res[3] == pytest.approx(interp_ref(0.6, UNIFORM, [1, 2, 3, 4, 5, 6]))
    # A grid in an array is not known at trace time: searched
    assert 'bsearch' in op_names(results[3])


@pytest.mark.parametrize('method', ['linear', 'previous', 'nearest'])
def test_numbers_at_trace_time(method: str) -> None:
    for xp in (UNIFORM, GRID):
        for x in POSITIONS:
            result = cp.interp(x, xp, TABLE, method)
            assert isinstance(result, float)
            assert result == pytest.approx(interp_ref(x, xp, TABLE, method))
    vec = cp.interp(cp.vector(POSITIONS), GRID, TABLE, method)
    assert isinstance(vec, cp.vector) and all(isinstance(v, float) for v in vec.values)
    assert math.isnan(cp.interp(float('nan'), GRID, TABLE, method))


@pytest.mark.parametrize('args', [
    (GRID, TABLE[:-1]),                       # different numbers of elements
    ([1.0], [2.0]),                           # one point
    ([0.0, 1.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0]),  # not strictly increasing
    ([0.0, 2.0, 1.0], [1.0, 2.0, 3.0]),
    (cp.array([[0.0, 1.0], [2.0, 3.0]]), cp.array([[0.0, 1.0], [2.0, 3.0]]))])
def test_invalid_tables(args: tuple[Any, Any]) -> None:
    with pytest.raises(ValueError):
        cp.interp(cp.value(0.5), *args)


def test_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        cp.interp(cp.value(0.5), GRID, TABLE, 'cubic')


# --- Gradients ---

def slope_ref(x: float, xp: list[float], fp: list[float]) -> float:
    if x <= xp[0] or x >= xp[-1]:
        return 0.0
    i = max(k for k in range(len(xp) - 1) if xp[k] <= x)
    return (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])


@pytest.mark.parametrize('xp', [UNIFORM, GRID])
def test_grad_scalar(xp: list[float]) -> None:
    positions = [-5.0, -1.2, -0.3, 0.1, 0.3, 0.9, 1.3, 4.5]
    x = [cp.value(v) for v in positions]
    grads = [cp.grad(cp.interp(v * 1.0, xp, TABLE), v) for v in x]
    assert evaluate(*grads) == pytest.approx([slope_ref(v, xp, TABLE) for v in positions], rel=1e-4, abs=1e-5)


@pytest.mark.parametrize('xp', [UNIFORM, GRID])
def test_grad_array(xp: list[float]) -> None:
    positions = [-5.0, -1.2, -0.3, 0.1, 0.3, 0.9, 1.3, 4.5]
    weights = [0.5 + 0.25 * i for i in range(len(positions))]
    x = cp.array([cp.value(v) for v in positions])
    result = (cp.interp(x, xp, TABLE) * cp.array(weights)).sum()
    grads = evaluate(cp.grad(result, x))[0]
    assert grads == pytest.approx([w * slope_ref(v, xp, TABLE) for w, v in zip(weights, positions)], rel=1e-4, abs=1e-5)


def test_grad_chain() -> None:
    """Gradient through the interpolation of a function of x"""
    x = cp.value(0.4)
    result = cp.interp(cp.sin(x) * 2.0, GRID, TABLE) ** 2
    u = math.sin(0.4) * 2.0
    expected = 2.0 * interp_ref(u, GRID, TABLE) * slope_ref(u, GRID, TABLE) * math.cos(0.4) * 2.0
    assert evaluate(cp.grad(result, x))[0] == pytest.approx(expected, rel=1e-4)


def test_grad_to_table_is_not_supported() -> None:
    table = [cp.value(v) for v in TABLE]
    result = cp.interp(cp.value(0.1), GRID, table)
    with pytest.raises(NotImplementedError):
        cp.grad(result, table[2])
