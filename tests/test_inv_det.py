"""Tests for cp.inv and cp.det: the solve and det array stencils for arrays and
packed tensors, the unrolled elimination (inv) and the Laplace expansion (det)
for small tensors."""
from typing import Any, Callable

import pytest

import copapy as cp
from copapy.backend import get_dag_stats


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def eliminate(a: list[list[float]], b: list[list[float]]) -> tuple[list[list[float]], float]:
    """Solution of a x = b and determinant of a by Gaussian elimination with
    partial pivoting in double precision"""
    n = len(a)
    rows = [list(ra) + list(rb) for ra, rb in zip(a, b)]
    det = 1.0
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(rows[i][k]))
        if rows[p][k] == 0:
            return [], 0.0
        if p != k:
            rows[k], rows[p] = rows[p], rows[k]
            det = -det
        det *= rows[k][k]
        for i in range(k + 1, n):
            f = rows[i][k] / rows[k][k]
            rows[i] = [x - f * y for x, y in zip(rows[i], rows[k])]
    for i in reversed(range(n)):
        for q in range(i + 1, n):
            rows[i] = [x - rows[i][q] * y for x, y in zip(rows[i], rows[q])]
        rows[i] = [x / rows[i][i] for x in rows[i]]
    return [row[n:] for row in rows], det


def inv_ref(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return eliminate(a, [[float(i == j) for j in range(n)] for i in range(n)])[0]


def det_ref(a: list[list[float]]) -> float:
    return eliminate(a, [[] for _ in a])[1]


def sample_matrix(n: int) -> list[list[float]]:
    """Well conditioned matrix with the largest elements beside the diagonal:
    each column needs a row swap"""
    m = [[((i * 7 + j * 13 + 3) % 11 - 5) * 0.37 for j in range(n)] for i in range(n)]
    for i in range(n):
        m[i][(i + 1) % n] += 1.5
    return m


def flat(m: list[list[float]]) -> list[float]:
    return [v for row in m for v in row]


def values(m: list[list[float]]) -> list[list[Any]]:
    return [[cp.value(v) for v in row] for row in m]


def stats(*results: Any) -> dict[str, int]:
    return get_dag_stats([v.net for v in results if isinstance(v, cp.value)])


# A zero on the diagonal: not invertible without pivoting
ZERO_DIAG = [[0.0, 2.0, 1.0], [1.0, 1.0, 0.5], [4.0, -1.0, 3.0]]
SINGULAR = [[1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [-2.0, 0.0, 5.0]]


# --- inv ---

@pytest.mark.parametrize('n', [1, 2, 3, 5, 8])
def test_array_inv(n: int) -> None:
    a = sample_matrix(n)
    x = cp.inv(cp.array(a))
    assert isinstance(x, cp.array) and x.shape == (n, n)
    assert get_dag_stats([x.element(0).net]).get('solve_floatarr_floatarr') == 1
    assert flat(evaluate(x)[0]) == pytest.approx(flat(inv_ref(a)), rel=1e-3, abs=1e-4)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
@pytest.mark.parametrize('pivot', [True, False])
def test_unrolled_inv(n: int, pivot: bool) -> None:
    a = sample_matrix(n)
    if not pivot:
        a = a[-1:] + a[:-1]  # largest elements on the diagonal
    x = cp.inv(cp.tensor(values(a)), pivot=pivot)
    assert isinstance(x, cp.tensor) and x.shape == (n, n)
    assert 'solve_floatarr_floatarr' not in stats(*x.values)
    assert evaluate(*x.values) == pytest.approx(flat(inv_ref(a)), rel=1e-3, abs=1e-4)


def test_large_tensor_inv_uses_array_stencil() -> None:
    a = sample_matrix(5)
    x = cp.inv(cp.tensor(values(a)))
    assert isinstance(x, cp.tensor)
    assert stats(*x.values).get('solve_floatarr_floatarr') == 1
    assert evaluate(*x.values) == pytest.approx(flat(inv_ref(a)), rel=1e-3, abs=1e-4)


def test_inv_requires_pivoting() -> None:
    x = cp.inv(cp.tensor(values(ZERO_DIAG)))
    assert evaluate(*x.values) == pytest.approx(flat(inv_ref(ZERO_DIAG)), rel=1e-4, abs=1e-5)


def test_constant_inv_at_trace_time() -> None:
    x = cp.inv(cp.tensor(ZERO_DIAG))
    assert all(isinstance(v, float) for v in x.values)
    assert list(x.values) == pytest.approx(flat(inv_ref(ZERO_DIAG)))
    x2 = cp.inv(ZERO_DIAG)  # type: ignore[call-overload]
    assert list(x2.values) == list(x.values)


def test_inv_times_matrix_is_identity() -> None:
    a = cp.tensor(values(ZERO_DIAG))
    product = cp.inv(a) @ a
    assert isinstance(product, cp.tensor)
    assert evaluate(*product.values) == pytest.approx(flat([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), abs=1e-5)


# --- det ---

@pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 17])
def test_array_det(n: int) -> None:
    a = sample_matrix(n)
    d = cp.det(cp.array(a))
    assert isinstance(d, cp.value)
    assert d.net.source.name == 'det_floatarr'
    assert evaluate(d)[0] == pytest.approx(det_ref(a), rel=1e-3)


def test_array_det_sign_and_singular() -> None:
    swapped = [ZERO_DIAG[1], ZERO_DIAG[0], ZERO_DIAG[2]]
    res = evaluate(cp.det(cp.array(ZERO_DIAG)), cp.det(cp.array(swapped)), cp.det(cp.array(SINGULAR)),
                   cp.det(cp.array([[2, 1], [1, 3]])))
    assert res == pytest.approx([-7.0, 7.0, 0.0, 5.0], rel=1e-5)


def test_array_det_does_not_change_operand() -> None:
    a = cp.array(ZERO_DIAG)
    res = evaluate(cp.det(a), a)
    assert flat(res[1]) == flat(ZERO_DIAG)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_unrolled_det(n: int) -> None:
    a = sample_matrix(n)
    d = cp.det(cp.tensor(values(a)))
    assert isinstance(d, cp.value)
    used = stats(d)
    assert not any(op.split('_')[0] in ('det', 'div', 'gtabs', 'mask', 'masknot') for op in used)
    assert evaluate(d)[0] == pytest.approx(det_ref(a), rel=1e-4)


def test_unrolled_det_number_of_operations() -> None:
    """2 x 2: two products and a difference, the last operation is not counted"""
    (a, b), (c, d) = values([[1.0, 2.0], [3.0, 5.0]])
    result = cp.det(cp.tensor([[a, b], [c, d]]))
    assert stats(result) == {'const_float': 4, 'mul_float_float': 2}
    assert evaluate(result)[0] == pytest.approx(-1.0)


def test_unrolled_det_singular() -> None:
    assert evaluate(cp.det(cp.tensor(values(SINGULAR))))[0] == pytest.approx(0.0, abs=1e-6)


def test_det_zeros_are_eliminated() -> None:
    """Triangular matrix: the product of the diagonal elements"""
    a = [[cp.value(2.0), cp.value(7.0), cp.value(-1.0)], [0.0, cp.value(3.0), cp.value(4.0)], [0.0, 0.0, cp.value(0.5)]]
    d = cp.det(cp.tensor(a))
    assert stats(d) == {'const_float': 3, 'mul_float_float': 1}
    assert evaluate(d)[0] == pytest.approx(3.0)


@pytest.mark.parametrize('n', [2, 4, 5, 9])
def test_constant_det_at_trace_time(n: int) -> None:
    a = sample_matrix(n)
    d = cp.det(cp.tensor(a))
    assert isinstance(d, float)
    assert d == pytest.approx(det_ref(a))
    assert cp.det(a) == d  # type: ignore[call-overload]


def test_constant_det_singular() -> None:
    assert cp.det(cp.tensor(SINGULAR)) == 0.0
    five = [[float(i + j) for j in range(5)] for i in range(5)]  # rank 2
    assert cp.det(cp.tensor(five)) == pytest.approx(0.0, abs=1e-9)


def test_int_det() -> None:
    a = [[cp.value(2), cp.value(1)], [cp.value(1), cp.value(3)]]
    assert evaluate(cp.det(cp.tensor(a)))[0] == pytest.approx(5.0)


@pytest.mark.parametrize('kwargs,n', [({}, 5), ({'packed': True}, 3)])
def test_tensor_det_uses_array_stencil(kwargs: dict[str, Any], n: int) -> None:
    a = sample_matrix(n)
    d = cp.det(cp.tensor(values(a), **kwargs))
    assert isinstance(d, cp.value) and d.net.source.name == 'det_floatarr'
    assert evaluate(d)[0] == pytest.approx(det_ref(a), rel=1e-3)


@pytest.mark.parametrize('func', [cp.inv, cp.det])
@pytest.mark.parametrize('a', [
    cp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    cp.array([1.0, 2.0])])
def test_requires_square_matrix(func: Callable[[Any], Any], a: Any) -> None:
    with pytest.raises(ValueError, match="square matrix"):
        func(a)


# --- Gradients ---

def numeric_grad(func: Callable[[list[float]], float], data: list[float], h: float = 1e-6) -> list[float]:
    def shifted(i: int, d: float) -> list[float]:
        return [v + d if j == i else v for j, v in enumerate(data)]
    return [(func(shifted(i, h)) - func(shifted(i, -h))) / (2 * h) for i in range(len(data))]


def as_matrix(d: list[float], n: int) -> list[list[float]]:
    return [d[i * n:(i + 1) * n] for i in range(n)]


@pytest.mark.parametrize('mode', ['unrolled', 'packed', 'array'])
def test_det_grad(mode: str) -> None:
    n = 3
    data = flat(ZERO_DIAG)
    inputs = [cp.value(v) for v in data]
    if mode == 'array':
        result = cp.det(cp.array(inputs).reshape(n, n))
    else:
        result = cp.det(cp.tensor(inputs, (n, n), packed=True if mode == 'packed' else None))
    out = evaluate(result, *cp.grad(result, inputs))
    assert out[0] == pytest.approx(-7.0, rel=1e-4)
    assert out[1:] == pytest.approx(numeric_grad(lambda d: det_ref(as_matrix(d, n)), data), rel=2e-3, abs=2e-3)


@pytest.mark.parametrize('mode', ['unrolled', 'packed', 'array'])
def test_inv_grad(mode: str) -> None:
    """Gradient of a weighted sum of the elements of the inverse"""
    n = 3
    data = flat(ZERO_DIAG)
    weights = [0.7 + 0.3 * i for i in range(n * n)]

    def ref(d: list[float]) -> float:
        return sum(w * v for w, v in zip(weights, flat(inv_ref(as_matrix(d, n)))))

    inputs = [cp.value(v) for v in data]
    result: Any
    if mode == 'array':
        result = (cp.inv(cp.array(inputs).reshape(n, n)).reshape(n * n) * cp.array(weights)).sum()
    else:
        x = cp.inv(cp.tensor(inputs, (n, n), packed=True if mode == 'packed' else None))
        result = (x * cp.tensor(weights, (n, n))).sum()
    out = evaluate(result, *cp.grad(result, inputs))
    assert out[0] == pytest.approx(ref(data), rel=1e-4, abs=1e-4)
    assert out[1:] == pytest.approx(numeric_grad(ref, data), rel=2e-3, abs=2e-3)
