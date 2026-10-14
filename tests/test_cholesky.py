from typing import Any

import pytest

import copapy as cp

np = pytest.importorskip('numpy')


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def variables(m: Any) -> cp.tensor[float]:
    """Tensor of copapy values with the elements of a numpy matrix"""
    return cp.tensor([cp.value(float(x)) for x in m.ravel()], m.shape)


def flat(x: Any) -> list[float]:
    """Flat element list of a result read from the target"""
    if isinstance(x, list):
        return [v for row in x for v in flat(row)]
    if isinstance(x, cp.vector | cp.tensor):
        values: Any = x.values
        return [float(v) for v in values]
    return [float(x)]


def spd_matrix(n: int, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1.0, 1.0, (n, n))
    return m @ m.T + n * np.eye(n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_cholesky(n: int) -> None:
    a = spd_matrix(n, n)
    ref = np.linalg.cholesky(a)

    l_var = cp.cholesky(variables(a))
    l_const = cp.cholesky(cp.tensor(a.tolist()))
    assert l_var.shape == (n, n)

    # Constant matrices are factorized at trace time
    assert flat(l_const) == pytest.approx(ref.ravel().tolist(), rel=1e-12)

    (res,) = evaluate(l_var)
    assert flat(res) == pytest.approx(ref.ravel().tolist(), rel=1e-4, abs=1e-6)


def test_cholesky_uses_lower_triangle() -> None:
    a = spd_matrix(4, 7)
    garbage = np.tril(a) + np.triu(np.full((4, 4), 99.0), 1)

    (res,) = evaluate(cp.cholesky(variables(garbage)))
    assert flat(res) == pytest.approx(np.linalg.cholesky(a).ravel().tolist(), rel=1e-4, abs=1e-6)


def test_cholesky_array_backed() -> None:
    a = spd_matrix(4, 3)
    l_arr = cp.cholesky(cp.tensor(cp.array([[cp.value(float(x)) for x in row] for row in a])))
    assert isinstance(l_arr, cp.tensor)
    assert l_arr.shape == (4, 4)

    (res,) = evaluate(l_arr)
    assert flat(res) == pytest.approx(np.linalg.cholesky(a).ravel().tolist(), rel=1e-4, abs=1e-6)


def test_cholesky_banded_keeps_zeros() -> None:
    # The factor of a tridiagonal matrix is bidiagonal: no fill-in
    n = 6
    a = 4.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    t = cp.tensor([[cp.value(float(a[i, j])) if abs(i - j) <= 1 else 0.0 for j in range(n)] for i in range(n)])
    lo = cp.cholesky(t)

    for i in range(n):
        for j in range(n):
            if not 0 <= i - j <= 1:
                assert not isinstance(lo.get_scalar(i, j), cp.value)

    (res,) = evaluate(lo)
    assert flat(res) == pytest.approx(np.linalg.cholesky(a).ravel().tolist(), rel=1e-4, abs=1e-6)


def test_cholesky_not_positive_definite() -> None:
    a = [[1.0, 2.0], [2.0, 1.0]]
    with pytest.raises(ValueError):
        cp.cholesky(cp.tensor(a))

    # Not known at trace time: nan instead of an exception
    (res,) = evaluate(cp.cholesky(variables(np.array(a))))
    assert np.isnan(flat(res)[3])


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("unit_diagonal", [False, True])
def test_solve_triangular(lower: bool, trans: bool, unit_diagonal: bool) -> None:
    n = 5
    rng = np.random.default_rng(11)
    full = rng.uniform(-1.0, 1.0, (n, n)) + 3.0 * np.eye(n)
    tri = np.tril(full) if lower else np.triu(full)
    if unit_diagonal:
        tri = tri - np.diag(np.diag(tri)) + np.eye(n)
    b = rng.uniform(-2.0, 2.0, n)
    ref = np.linalg.solve(tri.T if trans else tri, b)

    # The other triangle and, for unit_diagonal, the diagonal are not used
    x_var = cp.solve_triangular(variables(full), cp.vector([cp.value(float(v)) for v in b]),
                                lower=lower, trans=trans, unit_diagonal=unit_diagonal)
    x_const = cp.solve_triangular(cp.tensor(full.tolist()), cp.vector(b.tolist()),
                                  lower=lower, trans=trans, unit_diagonal=unit_diagonal)
    assert isinstance(x_var, cp.vector)
    assert flat(x_const) == pytest.approx(ref.tolist(), rel=1e-12)

    (res,) = evaluate(x_var)
    assert flat(res) == pytest.approx(ref.tolist(), rel=1e-4, abs=1e-6)


def test_solve_triangular_matrix_rhs() -> None:
    n, r = 4, 3
    rng = np.random.default_rng(5)
    tri = np.tril(rng.uniform(-1.0, 1.0, (n, n))) + 2.0 * np.eye(n)
    b = rng.uniform(-1.0, 1.0, (n, r))

    x = cp.solve_triangular(variables(tri), variables(b), lower=True)
    assert isinstance(x, cp.tensor) and x.shape == (n, r)

    (res,) = evaluate(x)
    assert flat(res) == pytest.approx(np.linalg.solve(tri, b).ravel().tolist(), rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("n", [1, 3, 6])
def test_cho_solve(n: int) -> None:
    a = spd_matrix(n, 20 + n)
    rng = np.random.default_rng(n)
    b = rng.uniform(-1.0, 1.0, n)
    b2 = rng.uniform(-1.0, 1.0, (n, 2))

    lo = cp.cholesky(variables(a))
    x = cp.cho_solve(lo, cp.vector([cp.value(float(v)) for v in b]))
    x2 = cp.cho_solve(lo, variables(b2))
    assert isinstance(x, cp.vector)
    assert x2.shape == (n, 2)

    res, res2 = evaluate(x, x2)
    assert flat(res) == pytest.approx(np.linalg.solve(a, b).tolist(), rel=1e-4, abs=1e-6)
    assert flat(res2) == pytest.approx(np.linalg.solve(a, b2).ravel().tolist(), rel=1e-4, abs=1e-6)


def test_cho_solve_array_backed() -> None:
    a = spd_matrix(3, 9)
    b = [1.0, -2.0, 0.5]

    lo = cp.cholesky(cp.tensor(cp.array([[cp.value(float(x)) for x in row] for row in a])))
    x = cp.cho_solve(lo, cp.vector(cp.array([cp.value(v) for v in b])))
    assert isinstance(x, cp.vector) and x.shape == (3,)

    (res,) = evaluate(x)
    assert flat(res) == pytest.approx(np.linalg.solve(a, b).tolist(), rel=1e-4, abs=1e-6)


def test_array_operands_raise() -> None:
    a = cp.array([[4.0, 2.0], [2.0, 3.0]])
    b = cp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.cholesky(a)
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.solve_triangular(a, cp.vector([1.0, 2.0]))
    with pytest.raises(TypeError, match="cp.tensor"):
        cp.cho_solve(cp.tensor([[2.0, 0.0], [1.0, 1.0]]), b)


def test_shape_errors() -> None:
    with pytest.raises(ValueError):
        cp.cholesky(cp.tensor([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        cp.solve_triangular(cp.tensor([[1.0, 0.0], [1.0, 1.0]]), cp.vector([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        cp.solve_triangular(cp.tensor([[0.0, 0.0], [1.0, 1.0]]), cp.vector([1.0, 2.0]), lower=True)
