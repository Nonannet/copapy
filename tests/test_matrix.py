"""Linear algebra on 2D tensors (matrices). Element-wise operations, indexing,
reshaping and reductions of general tensors are covered in test_tensor_basic.py."""
import math
from typing import Any

import pytest

import copapy as cp
from runner_helpers import trig_tol


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def variables(data: list[list[float]]) -> cp.tensor[Any]:
    return cp.tensor([[cp.value(x) for x in row] for row in data])


def matmul_ref(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def flat(data: list[list[float]]) -> list[float]:
    return [x for row in data for x in row]


def test_matrix_alias():
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    assert isinstance(m, cp.tensor)
    assert m.shape == (2, 3)
    assert m.values == (1, 2, 3, 4, 5, 6)


MATMUL_CASES = [
    ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
    ([[1, 2], [3, 4], [5, 6]], [[1, 2, 3], [4, 5, 6]]),
    ([[1.5, -2.0, 0.5]], [[2.0], [1.0], [-4.0]]),
    ([[2.0], [3.0]], [[4.0, 5.0]]),
]


@pytest.mark.parametrize(("a", "b"), MATMUL_CASES)
def test_matrix_matrix_multiplication(a: list[list[float]], b: list[list[float]]):
    ref = matmul_ref(a, b)

    res = cp.tensor(a) @ cp.tensor(b)
    assert isinstance(res, cp.tensor)
    assert res.shape == (len(a), len(b[0]))
    assert res.values == pytest.approx(flat(ref))  # pyright: ignore[reportUnknownMemberType]
    assert cp.tensor(a).matmul(cp.tensor(b)).values == pytest.approx(flat(ref))  # pyright: ignore[reportUnknownMemberType]

    res, = evaluate(variables(a) @ cp.tensor(b))
    assert res.shape == (len(a), len(b[0]))
    assert res.values == pytest.approx(flat(ref), rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_matrix_vector_multiplication():
    m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    v = [7.0, -8.0, 9.5]
    ref = [sum(x * y for x, y in zip(row, v)) for row in m]
    ref_left = [sum(v2 * m[i][j] for i, v2 in enumerate([2.0, -1.0])) for j in range(3)]

    vv = cp.vector([cp.value(x) for x in v])
    exprs = (cp.tensor(m) @ cp.vector(v),  # matrix @ vector
             cp.tensor(m) @ cp.tensor(v),  # matrix @ 1D tensor
             variables(m) @ cp.vector(v),
             cp.tensor(m) @ vv,
             cp.tensor([2.0, -1.0]) @ variables(m))  # 1D tensor @ matrix
    results = evaluate(*exprs)

    for expr, res in zip(exprs[:4], results[:4]):
        assert expr.shape == (2,)
        assert res.values == pytest.approx(ref, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]

    assert exprs[4].shape == (3,)
    assert results[4].values == pytest.approx(ref_left, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_vector_dot_as_1d_matmul():
    assert cp.tensor([1, 2, 3]) @ cp.tensor([4, 5, 6]) == 32


@pytest.mark.parametrize(("a", "b"), [([[1, 2], [3, 4]], [[1, 2, 3]]), ([[1, 2, 3]], [[1, 2, 3]]), ([[1, 2], [3, 4]], [1, 2, 3])])
def test_matmul_shape_mismatch(a: Any, b: Any):
    with pytest.raises(ValueError):
        cp.tensor(a) @ cp.tensor(b)


def test_matmul_properties():
    a = [[1.0, 2.0], [3.0, -4.0]]
    b = [[0.5, -1.0], [2.0, 1.5]]
    c = [[-2.0, 1.0], [0.0, 3.0]]
    ta, tb, tc = cp.tensor(a), cp.tensor(b), cp.tensor(c)

    # Associativity
    assert ((ta @ tb) @ tc).values == pytest.approx((ta @ (tb @ tc)).values)  # pyright: ignore[reportUnknownMemberType]
    # Distributivity
    assert (ta @ (tb + tc)).values == pytest.approx((ta @ tb + ta @ tc).values)  # pyright: ignore[reportUnknownMemberType]
    # (AB)^T = B^T A^T
    assert (ta @ tb).T.values == pytest.approx((tb.T @ ta.T).values)  # pyright: ignore[reportUnknownMemberType]
    # Not commutative
    assert (ta @ tb).values != pytest.approx((tb @ ta).values)  # pyright: ignore[reportUnknownMemberType]


def test_identity_is_neutral_element():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    va = variables(a)
    left, right = evaluate(cp.identity(2) @ va, va @ cp.eye(3))
    assert left.values == pytest.approx(flat(a))  # pyright: ignore[reportUnknownMemberType]
    assert right.values == pytest.approx(flat(a))  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(("args", "expected"), [((1,), [[1]]), ((3,), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                ((2, 3), [[1, 0, 0], [0, 1, 0]]), ((3, 2), [[1, 0], [0, 1], [0, 0]])])
def test_eye(args: tuple[int, ...], expected: list[list[int]]):
    m = cp.eye(*args)
    assert m.shape == (len(expected), len(expected[0]))
    assert m.values == tuple(flat(expected))


def test_identity():
    m = cp.identity(3)
    assert m.shape == (3, 3)
    assert m.values == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_constant_matrices_are_not_variables():
    """Constant zeros and ones allow eliminating operations at trace time"""
    v = cp.vector([cp.value(1.0), cp.value(2.0)])
    for m in (cp.eye(2), cp.identity(2), cp.diagonal(cp.vector([2.0, 3.0]))):
        assert not any(isinstance(x, cp.value) for x in m.values)

    # Multiplications by 0 and 1 are removed, only the values of v remain
    res = cp.identity(2) @ v
    assert res.values[0] is v.values[0]
    assert res.values[1] is v.values[1]

    # Constant matrix @ constant vector is evaluated at trace time
    assert (cp.tensor([[1, 2], [3, 4]]) @ cp.vector([1, 1])).values == (3, 7)


@pytest.mark.parametrize("container", [cp.vector, cp.tensor])
@pytest.mark.parametrize("diag", [[1, 2, 3], [2.5], [-1.0, 0.0]])
def test_diagonal(container: Any, diag: list[float]):
    n = len(diag)
    expected = [diag[i] if i == j else 0 for i in range(n) for j in range(n)]

    m = cp.diagonal(container(diag))
    assert m.shape == (n, n)
    assert m.values == tuple(expected)

    res, = evaluate(cp.diagonal(container([cp.value(x) for x in diag])))
    assert res.values == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_diagonal_requires_1d():
    with pytest.raises(ValueError):
        cp.diagonal(cp.tensor([[1, 2], [3, 4]]))


def test_transpose():
    m = cp.tensor([[1, 2, 3], [4, 5, 6]])
    for mt in (m.transpose(), m.T):
        assert mt.shape == (3, 2)
        assert mt.values == (1, 4, 2, 5, 3, 6)
        assert mt[0].values == (1, 4)
        assert mt[:, 1].values == (4, 5, 6)


@pytest.mark.parametrize(("data", "expected"), [([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 15), ([[2.5]], 2.5),
                                                ([[1, -2], [3, -4]], -3)])
def test_trace(data: list[list[float]], expected: float):
    m = cp.tensor(data)
    assert m.trace() == expected
    assert m.T.trace() == expected

    res, = evaluate(variables(data).trace())
    assert res == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_trace_requires_square_matrix():
    with pytest.raises((AssertionError, ValueError)):
        cp.tensor([[1, 2, 3], [4, 5, 6]]).trace()


def test_homogenize():
    m = cp.tensor([[1, cp.value(2)], [3.5, 4]])
    m_homo = m.homogenize()

    assert m_homo.shape == (2, 2)
    assert all(isinstance(v, cp.value) for v in m_homo.values)
    for row in m_homo:
        for elem in row:
            assert isinstance(elem, cp.tensor) and elem.ndim == 0

    res, = evaluate(m_homo)
    assert res.values == pytest.approx((1, 2, 3.5, 4))  # pyright: ignore[reportUnknownMemberType]


def test_rotation_matrix():
    """2D rotation matrices: R(a) @ R(b) = R(a + b), R^T = R^-1"""
    def rot(a: Any) -> cp.tensor[Any]:
        return cp.tensor([[cp.cos(a), -cp.sin(a)], [cp.sin(a), cp.cos(a)]])

    a = cp.value(0.3)
    b = cp.value(1.1)
    p = cp.vector([1.0, 0.0])

    combined, direct, ortho, rotated = evaluate(rot(a) @ rot(b), rot(a + b), rot(a).T @ rot(a), rot(a + b) @ p)

    assert combined.values == pytest.approx(direct.values, abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]
    assert ortho.values == pytest.approx((1, 0, 0, 1), abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]
    assert rotated.values == pytest.approx((math.cos(1.4), math.sin(1.4)), abs=trig_tol(1e-6))  # pyright: ignore[reportUnknownMemberType]


def test_compiled_matrix_expression():
    """Mixed constant and variable entries in a longer expression"""
    m = cp.tensor([[cp.value(1), 2], [3, cp.value(4)]])
    v = cp.vector([cp.value(5), 6])

    result = (m @ m.T + cp.identity(2) * 2) @ v - m.sum()

    mm = [[1, 2], [3, 4]]
    mmt = matmul_ref(mm, [[1, 3], [2, 4]])
    inner = [[mmt[i][j] + (2 if i == j else 0) for j in range(2)] for i in range(2)]
    ref = [sum(inner[i][j] * [5, 6][j] for j in range(2)) - 10 for i in range(2)]

    res, = evaluate(result)
    assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
