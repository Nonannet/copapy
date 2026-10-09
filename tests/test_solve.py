"""Tests for cp.solve: the solve array stencil for arrays and packed tensors and
the elimination unrolled into scalar operations with the gtabs, mask and masknot
stencils for pivoting."""
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


def solve_ref(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Gaussian elimination with partial pivoting in double precision"""
    n = len(a)
    rows = [list(ra) + list(rb) for ra, rb in zip(a, b)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(rows[i][k]))
        rows[k], rows[p] = rows[p], rows[k]
        for i in range(k + 1, n):
            f = rows[i][k] / rows[k][k]
            rows[i] = [x - f * y for x, y in zip(rows[i], rows[k])]
    for i in reversed(range(n)):
        for q in range(i + 1, n):
            rows[i] = [x - rows[i][q] * y for x, y in zip(rows[i], rows[q])]
        rows[i] = [x / rows[i][i] for x in rows[i]]
    return [row[n:] for row in rows]


def sample_matrix(n: int) -> list[list[float]]:
    """Well conditioned matrix with the largest elements beside the diagonal:
    each column needs a row swap"""
    m = [[((i * 7 + j * 13 + 3) % 11 - 5) * 0.37 for j in range(n)] for i in range(n)]
    for i in range(n):
        m[i][(i + 1) % n] += 3.0 * n
    return m


def sample_rhs(n: int, r: int) -> list[list[float]]:
    return [[((i * 5 + j * 3) % 13 - 6) * 0.61 + 0.2 for j in range(r)] for i in range(n)]


def column(m: list[list[float]]) -> list[float]:
    return [row[0] for row in m]


def flat(m: list[list[float]]) -> list[float]:
    return [v for row in m for v in row]


def values(m: list[list[float]]) -> list[list[Any]]:
    return [[cp.value(v) for v in row] for row in m]


# A zero on the diagonal: not solvable without pivoting
ZERO_DIAG = [[0.0, 2.0, 1.0], [1.0, 1.0, 0.5], [4.0, -1.0, 3.0]]


@pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 17])
@pytest.mark.parametrize('r', [1, 3])
def test_array_solve(n: int, r: int) -> None:
    a, b = sample_matrix(n), sample_rhs(n, r)
    x = cp.solve(cp.array(a), cp.array(b))
    assert isinstance(x, cp.array) and x.shape == (n, r)
    assert get_dag_stats([x.element(0).net]).get('solve_floatarr_floatarr') == 1
    result = evaluate(x)[0]
    assert flat(result) == pytest.approx(flat(solve_ref(a, b)), rel=1e-3, abs=1e-4)


def test_array_solve_vector_rhs() -> None:
    b = sample_rhs(3, 1)
    x = cp.solve(cp.array(ZERO_DIAG), cp.array(column(b)))
    assert x.shape == (3,)
    assert evaluate(x)[0] == pytest.approx(column(solve_ref(ZERO_DIAG, b)), rel=1e-4)


def test_array_solve_int_operands() -> None:
    a = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
    b = [1, 2, 3]
    x = cp.solve(cp.array(a), cp.array(b))
    assert evaluate(x)[0] == pytest.approx(column(solve_ref(a, [[v] for v in b])), rel=1e-4)  # type: ignore[arg-type]


def test_array_solve_does_not_change_operands() -> None:
    a, b = cp.array(ZERO_DIAG), cp.array([1.0, 2.0, 3.0])
    x = cp.solve(a, b)
    res = evaluate(x, a, b)
    assert flat(res[1]) == flat(ZERO_DIAG)
    assert res[2] == [1.0, 2.0, 3.0]


def test_array_and_tensor_operands() -> None:
    b = sample_rhs(3, 1)
    ref = column(solve_ref(ZERO_DIAG, b))
    x1 = cp.solve(cp.array(ZERO_DIAG), cp.vector(cp.value(v) for v in column(b)))
    x2 = cp.solve(cp.tensor(values(ZERO_DIAG)), cp.array(column(b)))
    assert isinstance(x1, cp.array) and isinstance(x2, cp.array)
    res = evaluate(x1, x2)
    assert res[0] == pytest.approx(ref, rel=1e-4)
    assert res[1] == pytest.approx(ref, rel=1e-4)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
@pytest.mark.parametrize('pivot', [True, False])
def test_unrolled_solve(n: int, pivot: bool) -> None:
    a, b = sample_matrix(n), sample_rhs(n, 1)
    if not pivot:
        a = a[-1:] + a[:-1]  # largest elements on the diagonal
    x = cp.solve(cp.tensor(values(a)), cp.vector(cp.value(v) for v in column(b)), pivot=pivot)
    assert isinstance(x, cp.vector)
    stats = get_dag_stats([v.net for v in x.values if isinstance(v, cp.value)])
    assert 'solve_floatarr_floatarr' not in stats
    assert ('gtabs_float_float' in stats) == (pivot and n > 1)
    assert evaluate(*x.values) == pytest.approx(column(solve_ref(a, b)), rel=1e-3, abs=1e-4)


def test_unrolled_solve_requires_pivoting() -> None:
    b = sample_rhs(3, 2)
    x = cp.solve(cp.tensor(values(ZERO_DIAG)), cp.tensor(values(b)))
    assert isinstance(x, cp.tensor) and x.shape == (3, 2)
    assert evaluate(*x.values) == pytest.approx(flat(solve_ref(ZERO_DIAG, b)), rel=1e-4)


def test_unrolled_solve_selects_rows_at_runtime() -> None:
    """The same compiled program swaps different rows for different values"""
    a = cp.tensor(values(ZERO_DIAG))
    b = cp.vector(cp.value(v) for v in [1.0, 2.0, 3.0])
    x = cp.solve(a, b)
    tg = cp.Target()
    tg.compile(*x.values)
    for m in (ZERO_DIAG, [[5.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.5, 3.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]):
        for av, v in zip(a.values, flat(m)):
            tg.write_value(av, v)  # type: ignore[arg-type]
        tg.run()
        assert [tg.read_value(v) for v in x.values] == pytest.approx(column(solve_ref(m, [[1.0], [2.0], [3.0]])), rel=1e-4)


def test_constants_are_solved_at_trace_time() -> None:
    b = sample_rhs(3, 1)
    x = cp.solve(cp.tensor(ZERO_DIAG), cp.vector(column(b)))
    assert all(isinstance(v, float) for v in x.values)
    assert list(x.values) == pytest.approx(column(solve_ref(ZERO_DIAG, b)))

    x2 = cp.solve(ZERO_DIAG, column(b))  # type: ignore[call-overload]
    assert isinstance(x2, cp.tensor) and list(x2.values) == list(x.values)


def test_constant_matrix_generates_no_pivoting_code() -> None:
    """Pivot rows of a constant matrix are selected at trace time"""
    x = cp.solve(cp.tensor(ZERO_DIAG), cp.vector(cp.value(v) for v in [1.0, 2.0, 3.0]))
    stats = get_dag_stats([v.net for v in x.values if isinstance(v, cp.value)])
    assert not any(op.split('_')[0] in ('gtabs', 'mask', 'masknot', 'div') for op in stats)
    assert evaluate(*x.values) == pytest.approx(column(solve_ref(ZERO_DIAG, [[1.0], [2.0], [3.0]])), rel=1e-4)


def test_zeros_are_eliminated() -> None:
    """A diagonal matrix of values: one division and one multiplication (the
    result, not counted by the statistic) per row"""
    d = [2.0, -4.0, 0.5]
    a = cp.diagonal(cp.vector(cp.value(v) for v in d))
    x = cp.solve(a, cp.vector(cp.value(v) for v in [1.0, 2.0, 3.0]))
    stats = get_dag_stats([v.net for v in x.values if isinstance(v, cp.value)])
    assert stats == {'const_float': 7, 'div_float_float': 3}
    assert evaluate(*x.values) == pytest.approx([0.5, -0.5, 6.0])


def test_select_ignores_inf_in_unselected_rows() -> None:
    """Swapped rows are selected by their bit pattern: no 0 * inf = nan"""
    a = [[1.0, float('inf')], [2.0, 1.0]]
    x = cp.solve(cp.tensor(values(a)), cp.vector(cp.value(v) for v in [1.0, 4.0]))
    res = evaluate(*x.values)
    assert res == pytest.approx([2.0, 0.0])


def test_packed_tensor_uses_array_stencil() -> None:
    n = 4
    a, b = sample_matrix(n), sample_rhs(n, 1)
    x = cp.solve(cp.tensor(values(a), packed=True), cp.vector(cp.value(v) for v in column(b)))
    assert isinstance(x, cp.vector)
    assert get_dag_stats([v.net for v in x.values]).get('solve_floatarr_floatarr') == 1  # type: ignore[union-attr]
    assert evaluate(*x.values) == pytest.approx(column(solve_ref(a, b)), rel=1e-3)


def uses_array_stencil(x: Any) -> bool:
    return 'solve_floatarr_floatarr' in get_dag_stats([v.net for v in x.values if isinstance(v, cp.value)])


def test_unroll_threshold() -> None:
    """The array stencil is used for more than 16 copapy values in the matrix"""
    def rhs(n: int) -> Any:
        return cp.vector(cp.value(v) for v in column(sample_rhs(n, 1)))

    assert not uses_array_stencil(cp.solve(cp.tensor(values(sample_matrix(4))), rhs(4)))
    assert uses_array_stencil(cp.solve(cp.tensor(values(sample_matrix(5))), rhs(5)))
    assert uses_array_stencil(cp.solve(cp.tensor(values(sample_matrix(5))), rhs(5), pivot=False))

    # Constants in the matrix and the elements of the right-hand side are not counted
    assert not uses_array_stencil(cp.solve(cp.tensor(sample_matrix(6)), rhs(6)))
    a = sample_matrix(5)
    mixed = [[cp.value(v) if i < 3 else v for v in row] for i, row in enumerate(a)]  # 15 values
    assert not uses_array_stencil(cp.solve(cp.tensor(mixed), rhs(5)))

    # packed=False always unrolls
    x = cp.solve(cp.tensor(values(a), packed=False), rhs(5))
    assert not uses_array_stencil(x)
    assert evaluate(*x.values) == pytest.approx(column(solve_ref(a, sample_rhs(5, 1))), rel=1e-3, abs=1e-4)


def test_singular_constant_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        cp.solve(cp.tensor([[1.0, 2.0], [2.0, 4.0]]), cp.vector([1.0, 2.0]))
    with pytest.raises(ValueError, match="pivoting is required"):
        cp.solve(cp.tensor([[0.0, 1.0], [1.0, 0.0]]), cp.vector([1.0, 2.0]), pivot=False)


@pytest.mark.parametrize('a,b', [
    (cp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), cp.vector([1.0, 2.0])),
    (cp.tensor([[1.0, 2.0], [3.0, 4.0]]), cp.vector([1.0, 2.0, 3.0])),
    (cp.array([[1.0, 2.0], [3.0, 4.0]]), cp.array([1.0, 2.0, 3.0])),
    (cp.array([1.0, 2.0]), cp.array([1.0, 2.0]))])
def test_shape_mismatch(a: Any, b: Any) -> None:
    with pytest.raises(ValueError):
        cp.solve(a, b)


def numeric_grad(func: Callable[[list[float]], float], data: list[float], h: float = 1e-6) -> list[float]:
    def shifted(i: int, d: float) -> list[float]:
        return [v + d if j == i else v for j, v in enumerate(data)]
    return [(func(shifted(i, h)) - func(shifted(i, -h))) / (2 * h) for i in range(len(data))]


@pytest.mark.parametrize('mode', ['unrolled', 'unrolled_no_pivot', 'packed', 'array'])
@pytest.mark.parametrize('r', [1, 2])
def test_grad(mode: str, r: int) -> None:
    """Gradient of a weighted sum of the solution to the matrix and the right-hand side"""
    n = 3
    a = ZERO_DIAG if mode != 'unrolled_no_pivot' else [ZERO_DIAG[2], ZERO_DIAG[0], ZERO_DIAG[1]]
    data = flat(a) + flat(sample_rhs(n, r))
    weights = [0.7 + 0.3 * i for i in range(n * r)]

    def ref(d: list[float]) -> float:
        x = solve_ref([d[i * n:(i + 1) * n] for i in range(n)],
                      [d[n * n + i * r:n * n + (i + 1) * r] for i in range(n)])
        return sum(w * v for w, v in zip(weights, flat(x)))

    inputs = [cp.value(v) for v in data]
    result: Any
    if mode == 'array':
        x = cp.solve(cp.array(inputs[:n * n]).reshape(n, n), cp.array(inputs[n * n:]).reshape(n, r))
        result = (x.reshape(n * r) * cp.array(weights)).sum()
    else:
        packed = True if mode == 'packed' else None
        x = cp.solve(cp.tensor(inputs[:n * n], (n, n), packed=packed), cp.tensor(inputs[n * n:], (n, r)),
                     pivot=mode != 'unrolled_no_pivot')
        result = (x * cp.tensor(weights, (n, r))).sum()

    grads = cp.grad(result, inputs)
    out = evaluate(result, *grads)
    assert out[0] == pytest.approx(ref(data), rel=1e-4, abs=1e-4)
    assert out[1:] == pytest.approx(numeric_grad(ref, data), rel=2e-3, abs=2e-3)
