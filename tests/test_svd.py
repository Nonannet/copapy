"""Tests for cp.svd: the svd array stencil (one-sided Jacobi), the trace-time
decomposition of constant tensors and the automatic differentiation.

Singular vectors are only defined up to their sign (and up to a rotation for
equal singular values), so the results are checked by the properties that
define the decomposition: u @ diag(s) @ vh reconstructs the matrix, u and vh
have orthonormal columns and rows, s is sorted descending and not negative.
Singular values are compared to matrices constructed with known values."""
import math
from typing import Any, Callable

import pytest

import copapy as cp
from copapy.backend import get_dag_stats

Matrix = list[list[float]]


# --- Helpers (pure Python, double precision) ---

def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][q] * b[q][j] for q in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def eye(n: int) -> Matrix:
    return [[float(i == j) for j in range(n)] for i in range(n)]


def householder(n: int, seed: int) -> Matrix:
    """Orthogonal n x n matrix: product of two Householder reflections"""
    q = eye(n)
    for h in range(2):
        v = [math.sin(seed * 1.7 + i * 2.3 + h * 0.9) + 0.1 for i in range(n)]
        vv = sum(x * x for x in v)
        refl = [[float(i == j) - 2 * v[i] * v[j] / vv for j in range(n)] for i in range(n)]
        q = matmul(q, refl)
    return q


def with_singular_values(m: int, n: int, s: list[float], seed: int = 1) -> Matrix:
    """m x n matrix with the given singular values: q1 @ diag(s) @ q2^T"""
    d = [[s[i] if i == j and i < len(s) else 0.0 for j in range(n)] for i in range(m)]
    return matmul(matmul(householder(m, seed), d), transpose(householder(n, seed + 5)))


def random_matrix(m: int, n: int, seed: int = 0) -> Matrix:
    return [[math.sin(seed * 3.1 + i * 1.3 + j * 2.9 + i * j * 0.7) * 2.0 for j in range(n)] for i in range(m)]


def as_values(a: Matrix) -> cp.tensor[float]:
    return cp.tensor([[cp.value(float(x)) for x in row] for row in a])


def consts(t: Any) -> list[float]:
    """Elements of a tensor or vector computed at trace time"""
    ret = list(t.values)
    assert all(isinstance(v, float) for v in ret)
    return ret


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def run_svd(a: Matrix, **kwargs: Any) -> tuple[Matrix, list[float], Matrix]:
    """SVD by the array stencil, results as nested lists"""
    u, s, vh = cp.svd(as_values(a), **kwargs)
    ru, rs, rvh = evaluate(list(u.values), list(s.values), list(vh.values))
    m, n = len(a), len(a[0])
    k = min(m, n)
    assert u.shape == (m, k) and s.shape == (k,) and vh.shape == (k, n)
    return [ru[i * k:(i + 1) * k] for i in range(m)], rs, [rvh[i * n:(i + 1) * n] for i in range(k)]


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def check_decomposition(a: Matrix, u: Matrix, s: list[float], vh: Matrix, tol: float = 2e-6) -> None:
    """u @ diag(s) @ vh == a, orthonormal u columns and vh rows, s sorted descending"""
    m, n = len(a), len(a[0])
    k = min(m, n)
    assert all(math.isfinite(x) for row in u + vh for x in row)
    assert all(x >= 0 for x in s)
    assert all(s[i] >= s[i + 1] for i in range(k - 1)), f"Not sorted: {s}"
    scale = max(max(abs(x) for row in a for x in row), 1e-300)
    us = [[u[i][j] * s[j] for j in range(k)] for i in range(m)]
    assert max_abs_diff(matmul(us, vh), a) <= tol * scale * max(m, n), "u @ diag(s) @ vh != a"
    assert max_abs_diff(matmul(transpose(u), u), eye(k)) <= tol * max(m, n), "u columns not orthonormal"
    assert max_abs_diff(matmul(vh, transpose(vh)), eye(k)) <= tol * max(m, n), "vh rows not orthonormal"


def projector(u: Matrix, vh: Matrix, i: int) -> Matrix:
    """u_i v_i^T: independent of the sign of the singular vectors"""
    return [[ui[i] * vj for vj in vh[i]] for ui in u]


# --- Decomposition by the array stencil ---

SHAPES = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (4, 4), (4, 3), (3, 4), (7, 2), (2, 7), (8, 8), (12, 5), (5, 12), (17, 17)]


@pytest.mark.parametrize('m,n', SHAPES)
def test_decomposition(m: int, n: int) -> None:
    a = random_matrix(m, n, seed=m * 31 + n)
    u, s, vh = run_svd(a)
    check_decomposition(a, u, s, vh)


@pytest.mark.parametrize('m,n', [(3, 3), (6, 4), (4, 6), (10, 10)])
def test_known_singular_values(m: int, n: int) -> None:
    k = min(m, n)
    s_ref = [5.0 / (i + 1) ** 1.5 for i in range(k)]
    a = with_singular_values(m, n, s_ref, seed=m + n)
    u, s, vh = run_svd(a)
    assert s == pytest.approx(s_ref, rel=1e-5, abs=1e-6 * s_ref[0])
    check_decomposition(a, u, s, vh)


def test_singular_values_sorted_from_any_column_order() -> None:
    # Diagonal matrix with ascending values: the columns are already
    # orthogonal (no rotation), only the sort orders the values
    a = [[0.5 if i == j == 0 else 0.0 for j in range(4)] for i in range(4)]
    for i in range(1, 4):
        a[i][i] = float(i + 1)
    a[2][2] = -3.0
    u, s, vh = run_svd(a)
    assert s == pytest.approx([4.0, 3.0, 2.0, 0.5], rel=1e-6)
    check_decomposition(a, u, s, vh)


def test_permutation_and_negative_entries() -> None:
    a = [[0.0, 0.0, 2.0], [-5.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    u, s, vh = run_svd(a)
    assert s == pytest.approx([5.0, 2.0, 1.0], rel=1e-6)
    check_decomposition(a, u, s, vh)


def test_graded_matrix() -> None:
    # Condition number 1e5: the small singular values have a relative error of
    # about eps * cond, the decomposition is accurate relative to the largest
    s_ref = [10.0 ** -i for i in range(6)]
    a = with_singular_values(6, 6, s_ref, seed=3)
    u, s, vh = run_svd(a)
    assert s == pytest.approx(s_ref, abs=2e-6)
    check_decomposition(a, u, s, vh)


def test_clustered_singular_values() -> None:
    s_ref = [1.0 + 1e-4 * i for i in range(8)][::-1]
    a = with_singular_values(8, 8, s_ref, seed=7)
    u, s, vh = run_svd(a)
    assert s == pytest.approx(s_ref, abs=2e-6)
    check_decomposition(a, u, s, vh)


# --- Rank deficient and degenerate matrices ---

@pytest.mark.parametrize('m,n', [(3, 3), (4, 2), (2, 4)])
def test_zero_matrix(m: int, n: int) -> None:
    a = [[0.0] * n for _ in range(m)]
    u, s, vh = run_svd(a)
    assert s == [0.0] * min(m, n)
    check_decomposition(a, u, s, vh)  # u and vh orthonormal, not nan


@pytest.mark.parametrize('m,n', [(4, 4), (5, 3), (3, 5)])
def test_rank_one(m: int, n: int) -> None:
    x, y = [i + 1.0 for i in range(m)], [(-1.0) ** j * (j + 0.5) for j in range(n)]
    a = [[xi * yj for yj in y] for xi in x]
    u, s, vh = run_svd(a)
    s_ref = math.sqrt(sum(v * v for v in x)) * math.sqrt(sum(v * v for v in y))
    assert s[0] == pytest.approx(s_ref, rel=1e-6)
    assert max(s[1:]) <= 1e-6 * s_ref
    check_decomposition(a, u, s, vh)


def test_rank_deficient() -> None:
    s_ref = [3.0, 2.0, 0.0, 0.0]
    a = with_singular_values(6, 4, s_ref, seed=11)
    u, s, vh = run_svd(a)
    assert s == pytest.approx(s_ref, abs=2e-6)
    check_decomposition(a, u, s, vh)


def test_zero_column_and_equal_rows() -> None:
    a = [[1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [1.0, 0.0, 2.0], [-2.0, 0.0, 5.0]]
    u, s, vh = run_svd(a)
    assert s[2] == pytest.approx(0.0, abs=1e-6)
    check_decomposition(a, u, s, vh)


@pytest.mark.parametrize('a', [eye(4), [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], householder(5, 2)],
                         ids=['identity', 'diagonal_ties', 'orthogonal'])
def test_equal_singular_values(a: Matrix) -> None:
    u, s, vh = run_svd(a)
    s_ref = sorted((math.sqrt(sum(x * x for x in col)) for col in zip(*a)), reverse=True)  # orthogonal columns
    assert s == pytest.approx(s_ref, rel=1e-6)
    check_decomposition(a, u, s, vh)


# --- Range of magnitudes ---

@pytest.mark.parametrize('scale', [1e30, 1e-30, 2.0 ** 100, 3e-38])
def test_large_and_small_magnitudes(scale: float) -> None:
    # Squared column norms would overflow or underflow without the scaling
    base = random_matrix(4, 3, seed=2)
    a = [[x * scale for x in row] for row in base]
    u, s, vh = run_svd(a)
    _, s_base, _ = run_svd(base)
    assert s == pytest.approx([x * scale for x in s_base], rel=1e-6)
    check_decomposition(a, u, s, vh)


def test_mixed_magnitudes() -> None:
    a = [[1e20, 1.0, 0.0], [0.0, 1e-20, 2.0], [3.0, 0.0, 1e10]]
    u, s, vh = run_svd(a)
    assert s[0] == pytest.approx(1e20, rel=1e-6)
    assert s[1] == pytest.approx(1e10, rel=1e-6)
    check_decomposition(a, u, s, vh)


def test_nan_input() -> None:
    a = [[float('nan'), 1.0], [2.0, 3.0]]
    _, s, _ = run_svd(a)
    assert all(math.isnan(x) for x in s)


# --- Consistency ---

def test_transposed_matrix() -> None:
    a = random_matrix(5, 3, seed=4)
    u, s, vh = run_svd(a)
    ut, st, vht = run_svd(transpose(a))
    assert st == pytest.approx(s, rel=1e-6)
    for i in range(3):
        # The singular vectors of a^T are the swapped ones of a
        assert max_abs_diff(projector(ut, vht, i), transpose(projector(u, vh, i))) <= 1e-5


def test_constant_matrix_at_trace_time() -> None:
    a = random_matrix(5, 4, seed=8)
    u, s, vh = cp.svd(cp.tensor(a))
    k = 4
    uc = [consts(u)[i * k:(i + 1) * k] for i in range(5)]  # numbers, no code
    vhc = [consts(vh)[i * 4:(i + 1) * 4] for i in range(k)]
    check_decomposition(a, uc, consts(s), vhc, tol=1e-14)

    # Same results as the array stencil (up to the signs of the vectors)
    ru, rs, rvh = run_svd(a)
    assert rs == pytest.approx(consts(s), rel=1e-6)
    for i in range(k):
        assert max_abs_diff(projector(ru, rvh, i), projector(uc, vhc, i)) <= 1e-5


@pytest.mark.parametrize('a', [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], [[1e300, 1.0], [1.0, 1e-300]]],
                         ids=['zero', 'rank_one', 'huge'])
def test_constant_degenerate_matrices(a: Matrix) -> None:
    u, s, vh = cp.svd(cp.tensor(a))
    m, n = len(a), len(a[0])
    k = min(m, n)
    check_decomposition(a, [consts(u)[i * k:(i + 1) * k] for i in range(m)], consts(s),
                        [consts(vh)[i * n:(i + 1) * n] for i in range(k)], tol=1e-14)


def test_matches_numpy() -> None:
    np = pytest.importorskip('numpy')
    a = random_matrix(6, 4, seed=12)
    u, s, vh = run_svd(a)
    nu, ns, nvh = np.linalg.svd(np.array(a), full_matrices=False)
    assert s == pytest.approx(list(ns), rel=1e-5)
    for i in range(4):
        assert np.allclose(np.outer(np.array(u)[:, i], np.array(vh)[i]), np.outer(nu[:, i], nvh[i]), atol=1e-5)


# --- Program structure and runtime behavior ---

def test_single_stencil() -> None:
    u, s, vh = cp.svd(as_values(random_matrix(6, 6)))
    stats = get_dag_stats([v.net for v in u.values + s.values + vh.values if isinstance(v, cp.value)])
    assert stats.get('svd_floatarr') == 1
    s_only = cp.svd(as_values(random_matrix(6, 6)), compute_uv=False)
    assert isinstance(s_only, cp.vector) and s_only.shape == (6,)
    assert get_dag_stats([v.net for v in s_only.values if isinstance(v, cp.value)]).get('svd_floatarr') == 1


def test_new_input_values_on_each_run() -> None:
    # The work memory of the stencil is reused: results must not depend on the previous run
    a1, a2 = random_matrix(4, 4, seed=1), with_singular_values(4, 4, [4.0, 3.0, 0.0, 0.0])
    inputs = [[cp.value(x) for x in row] for row in a1]
    u, s, vh = cp.svd(cp.tensor(inputs))
    tg = cp.Target()
    tg.compile(u, s, vh)
    for a in (a1, a2, a1):
        for row_in, row in zip(inputs, a):
            for v, x in zip(row_in, row):
                tg.write_value(v, x)
        tg.run()
        ru, rs, rvh = (tg.read_value(list(t.values)) for t in (u, s, vh))
        check_decomposition(a, [ru[i * 4:(i + 1) * 4] for i in range(4)], rs, [rvh[i * 4:(i + 1) * 4] for i in range(4)])


def test_two_decompositions_in_one_program() -> None:
    # Equal sizes: the constant parameter and work arrays may be shared
    a1, a2 = random_matrix(3, 3, seed=5), random_matrix(3, 3, seed=6)
    s1 = cp.svd(as_values(a1), compute_uv=False)
    s2 = cp.svd(as_values(a2), compute_uv=False)
    r1, r2 = evaluate(list(s1.values), list(s2.values))
    assert r1 == pytest.approx(run_svd(a1)[1], rel=1e-6)
    assert r2 == pytest.approx(run_svd(a2)[1], rel=1e-6)
    assert r1 != pytest.approx(r2, rel=1e-3)


def test_input_types() -> None:
    a = [[3, 1], [1, 3], [0, 2]]  # int values
    ref = run_svd([[float(x) for x in row] for row in a])[1]
    s_int = cp.svd(cp.tensor([[cp.value(x) for x in row] for row in a]), compute_uv=False)
    s_packed = cp.svd(cp.tensor(cp.array([[float(x) for x in row] for row in a]), packed=True), compute_uv=False)
    s_mixed = cp.svd(cp.tensor([[cp.value(3.0), 1.0], [1.0, 3.0], [0.0, cp.value(2.0)]]), compute_uv=False)  # type: ignore[arg-type]
    for res in evaluate(*(list(s.values) for s in (s_int, s_packed, s_mixed))):
        assert res == pytest.approx(ref, rel=1e-6)


def test_sweeps_parameter() -> None:
    a = with_singular_values(8, 8, [8.0 - i for i in range(8)], seed=9)
    u, s, vh = run_svd(a, sweeps=8)
    check_decomposition(a, u, s, vh)
    # Too few sweeps: u and vh stay orthonormal, the columns are not orthogonal yet
    u1, s1, _ = run_svd(a, sweeps=1)
    assert max_abs_diff(matmul(transpose(u1), u1), eye(8)) <= 1e-5
    assert s1 != pytest.approx([8.0 - i for i in range(8)], rel=1e-3)


def test_zero_sweeps() -> None:
    # Without rotations the column norms are only ordered by the final sort and the
    # nearly parallel columns of a graded matrix are only orthogonalized by Gram-Schmidt
    a = [[1.0 + 1e-3 * math.sin(i * 1.1 + j * 2.3) for j in range(5)] for i in range(6)]
    for i in range(6):
        a[i][0] *= 0.1
        a[i][3] *= 10.0
    u, s, vh = run_svd(a, sweeps=0)
    assert all(s[i] >= s[i + 1] for i in range(4)), f"Not sorted: {s}"
    assert max_abs_diff(matmul(transpose(u), u), eye(5)) <= 1e-5
    assert max_abs_diff(matmul(vh, transpose(vh)), eye(5)) <= 1e-6


def test_invalid_arguments() -> None:
    with pytest.raises(TypeError):
        cp.svd(cp.array([[1.0, 2.0], [3.0, 4.0]]))  # type: ignore[call-overload]
    with pytest.raises(ValueError):
        cp.svd(cp.tensor([cp.value(1.0), 2.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cp.svd(as_values(random_matrix(2, 2)), sweeps=-1)
    # Square matrices: the full and the reduced decomposition are identical
    u, s, vh = cp.svd(as_values(random_matrix(2, 2)))
    assert u.shape == (2, 2) and vh.shape == (2, 2)


# --- Automatic differentiation ---

def numeric_grad(func: Callable[[Matrix], float], a: Matrix, h: float = 1e-6) -> list[float]:
    """Central differences of func, evaluated by the trace-time decomposition (double precision)"""
    m, n = len(a), len(a[0])
    ret = []
    for i in range(m):
        for j in range(n):
            ap = [row[:] for row in a]
            am = [row[:] for row in a]
            ap[i][j] += h
            am[i][j] -= h
            ret.append((func(ap) - func(am)) / (2 * h))
    return ret


def check_grad(loss: Callable[[Any, Any, Any], Any], a: Matrix, rel: float = 2e-3, abs_tol: float = 2e-4) -> None:
    """Gradient by autograd on the target vs. central differences.
    loss(u, s, vh) must not depend on the signs of the singular vectors."""
    inputs = [[cp.value(x) for x in row] for row in a]
    flat_inputs = [v for row in inputs for v in row]
    result = loss(*cp.svd(cp.tensor(inputs)))
    grads = cp.grad(result, flat_inputs)

    def ref(b: Matrix) -> float:
        value = loss(*cp.svd(cp.tensor(b)))
        assert isinstance(value, float)
        return value

    assert evaluate(grads)[0] == pytest.approx(numeric_grad(ref, a), rel=rel, abs=abs_tol)


def sum_s(u: Any, s: Any, vh: Any) -> Any:
    return sum(s.values)


def weighted_s(u: Any, s: Any, vh: Any) -> Any:
    return sum(x * (i + 1) for i, x in enumerate(s.values)) + s.values[0] * s.values[-1]


def projector_loss(u: Any, s: Any, vh: Any) -> Any:
    # Elements of u_0 v_0^T and u_1 u_1^T: independent of the signs
    m, k = u.shape
    n = vh.shape[1]
    ret = 0.0
    for i in range(m):
        for j in range(n):
            ret = ret + u.values[i * k] * vh.values[j] * ((i + 1) * 0.3 - j * 0.2)
    for i in range(m):
        ret = ret + u.values[i * k + 1] * u.values[i * k + 1] * (i - 1.5)
    return ret


def mixed_loss(u: Any, s: Any, vh: Any) -> Any:
    n = vh.shape[1]
    return projector_loss(u, s, vh) * s.values[0] + sum(vh.values[n + j] * vh.values[n + j] * j for j in range(n))


@pytest.mark.parametrize('loss', [sum_s, weighted_s, projector_loss, mixed_loss])
@pytest.mark.parametrize('m,n', [(3, 3), (5, 3), (3, 5)])
def test_grad(loss: Callable[[Any, Any, Any], Any], m: int, n: int) -> None:
    s_ref = [3.0, 1.7, 0.6]
    check_grad(loss, with_singular_values(m, n, s_ref, seed=m * 2 + n))


def test_grad_of_singular_values_for_equal_values() -> None:
    # d sum(s) / d a = u @ vh: defined (finite) also for repeated singular values
    a = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    inputs = [[cp.value(x) for x in row] for row in a]
    _, s, _ = cp.svd(cp.tensor(inputs))
    grads = cp.grad(sum(s.values), [v for row in inputs for v in row])
    assert evaluate(grads)[0] == pytest.approx([float(i == j) for i in range(3) for j in range(3)], abs=1e-5)


def test_grad_of_packed_tensor() -> None:
    a = with_singular_values(4, 3, [2.0, 1.0, 0.5], seed=4)
    arr = cp.array([[cp.value(x) for x in row] for row in a])
    _, s, _ = cp.svd(cp.tensor(arr, packed=True))
    g = cp.grad(s.values[0] * 2.0 + s.values[2], arr)
    assert isinstance(g, cp.array)

    def loss(b: Matrix) -> float:
        sv = consts(cp.svd(cp.tensor(b), compute_uv=False))
        return sv[0] * 2.0 + sv[2]

    ref = numeric_grad(loss, a)
    assert [x for row in evaluate(g)[0] for x in row] == pytest.approx(ref, rel=2e-3, abs=2e-4)
