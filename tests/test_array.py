"""Tests for the array class: array stencils for element-wise operations,
reductions, matrix-vector products and the interaction with scalar values."""
import operator
from typing import Any, Callable

import pytest

import copapy as cp


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample_values(n: int, dtype: str, offset: int = 0) -> list[Any]:
    if dtype == 'int':
        return [(i * 7 + offset) % 23 - 11 for i in range(n)]
    return [((i * 7 + offset) % 23 - 11) * 0.37 + 0.5 for i in range(n)]


OPS: dict[str, Callable[[Any, Any], Any]] = {
    'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': operator.truediv}

# Lengths below, equal and above the SIMD width and the unrolling of the reductions
SIZES = [1, 3, 8, 17, 100]


@pytest.mark.parametrize('n', SIZES)
@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('t1,t2', [('int', 'int'), ('int', 'float'), ('float', 'int'), ('float', 'float')])
def test_elementwise(n: int, op: str, t1: str, t2: str) -> None:
    va = sample_values(n, t1)
    vb = [v if v != 0 else 3 for v in sample_values(n, t2, 5)]  # no division by zero
    a, b = cp.array(va, t1), cp.array(vb, t2)
    s = cp.value(vb[0])
    f = OPS[op]

    results = evaluate(f(a, b), f(a, s), f(s, a if op != 'div' else b), f(a, 3), f(2.5, b))
    expected = [
        [f(x, y) for x, y in zip(va, vb)],
        [f(x, vb[0]) for x in va],
        [f(vb[0], y) for y in (va if op != 'div' else vb)],
        [f(x, 3) for x in va],
        [f(2.5, y) for y in vb]]

    for res, exp in zip(results, expected):
        assert res == pytest.approx(exp, rel=1e-5)


@pytest.mark.parametrize('n', SIZES)
@pytest.mark.parametrize('dtype', ['int', 'float'])
def test_reductions(n: int, dtype: str) -> None:
    va = sample_values(n, dtype)
    vb = sample_values(n, dtype, 3)
    a, b = cp.array(va), cp.array(vb)

    res_sum, res_dot, res_matmul = evaluate(a.sum(), a.dot(b), a @ b)

    assert res_sum == pytest.approx(sum(va), rel=1e-5)
    assert res_dot == pytest.approx(sum(x * y for x, y in zip(va, vb)), rel=1e-5)
    assert res_matmul == pytest.approx(res_dot)
    if dtype == 'int':
        assert isinstance(res_sum, int) and isinstance(res_dot, int)


@pytest.mark.parametrize('m,n', [(1, 1), (3, 5), (8, 17), (20, 64)])
def test_matvec(m: int, n: int) -> None:
    rows = [sample_values(n, 'float', r) for r in range(m)]
    vx = sample_values(n, 'float', 11)
    mat, x = cp.array(rows), cp.array(vx)

    res, = evaluate(mat @ x)

    assert mat.shape == (m, n)
    assert res == pytest.approx([sum(a * b for a, b in zip(r, vx)) for r in rows], rel=1e-5)


def test_chained_ops_and_elements() -> None:
    va = [1.0, 2.0, 3.0, 4.0, 5.0]
    vb = [10, 20, 30, 40, 50]
    a, b = cp.array(va), cp.array(vb)
    s = cp.value(2.5)

    d = (a + b) * s - 1
    m = cp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
    h = m @ a
    x = d[2] * 2 + h[1]  # scalar ops on array elements
    k = -(a * x)  # array op with a computed scalar
    e = cp.sqrt(d.sum()) + a[-1]  # scalar op on reduction result

    ref_d = [(p + q) * 2.5 - 1 for p, q in zip(va, vb)]
    ref_x = ref_d[2] * 2 + 6.0

    res_d, res_x, res_k, res_e, res_h = evaluate(d, x, k, e, h)

    assert res_d == pytest.approx(ref_d)
    assert res_x == pytest.approx(ref_x)
    assert res_k == pytest.approx([-p * ref_x for p in va])
    assert res_e == pytest.approx(sum(ref_d) ** 0.5 + 5.0)
    assert res_h == pytest.approx([55.0, 6.0])


def test_write_and_rerun() -> None:
    a = cp.array([1.0, 2.0, 3.0])
    w = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    s = cp.value(2.0)
    y = w @ (a * s)

    tg = cp.Target()
    tg.compile(y)
    tg.run()
    assert tg.read_value(y) == pytest.approx([2.0, 12.0])

    tg.write_value(a, [1, 1, 1])
    tg.write_value(w, [[0, 0, 1], [2, 0, 0]])
    tg.write_value(s, 3.0)
    tg.run()
    assert tg.read_value(y) == pytest.approx([3.0, 6.0])
    assert tg.read_value(w) == [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]]


def test_shared_subexpression() -> None:
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])

    c1 = a + b
    c2 = a + b  # identical operation is computed once

    r1, r2, r3 = evaluate(c1, c2, c1 * c2)

    assert r1 == r2 == [5, 7, 9]
    assert r3 == [25, 49, 81]


def test_code_size_independent_of_length() -> None:
    from copapy.backend import compile_to_dag

    def program_size(n: int) -> int:
        a = cp.array([1.0] * n)
        b = cp.array([2.0] * n)
        c = ((a + b) * 3.0 - a) / b
        dw, _ = compile_to_dag([c.net.source], cp.generic_sdb)
        return len(dw.get_data()) - 2 * 4 * n  # without the data of a, b

    assert program_size(1000) == program_size(10)


def test_errors() -> None:
    a = cp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        a + cp.array([1.0, 2.0])
    with pytest.raises(IndexError):
        a[3]
    with pytest.raises(ValueError):
        cp.array([[1, 2], [3]])
    with pytest.raises(NotImplementedError):
        cp.array([cp.value(1.0), 2.0])
