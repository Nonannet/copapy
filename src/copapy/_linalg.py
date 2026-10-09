from typing import Any, Sequence, overload
from ._basic_types import value, ArrayNet, add_op, to_float
from ._arrays import array, _add_array_op
from ._vectors import vector
from ._tensors import tensor
from ._casts import _cast_array


def _gtabs(x: Any, y: Any) -> Any:
    """Int mask with all bits set (-1) if |x| > |y|, otherwise 0"""
    if not isinstance(x, value) and x == 0:
        return 0
    if isinstance(x, value) or isinstance(y, value):
        return add_op('gtabs', [x, y])
    return -int(abs(x) > abs(y))


def _select(m: value[int], x: Any, y: Any) -> Any:
    """x if the mask m is set, otherwise y. The unselected operand is replaced
    by 0.0 by its bit pattern, so the result is exact and inf or nan in the
    unselected operand have no effect."""
    kept_x = 0.0 if not isinstance(x, value) and x == 0 else add_op('mask', [x, m])
    kept_y = 0.0 if not isinstance(y, value) and y == 0 else add_op('masknot', [y, m])
    return kept_x + kept_y


def _cswap(m: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """(y, x) if the mask m is set, otherwise (x, y). Masks known at trace
    time and equal constants (e.g. zeros in both rows) generate no code."""
    if not isinstance(m, value):
        return (y, x) if m else (x, y)
    if not isinstance(x, value) and not isinstance(y, value) and x == y:
        return x, y
    return _select(m, y, x), _select(m, x, y)


def _solve_values(a: Sequence[Any], b: Sequence[Any], n: int, r: int, pivot: bool) -> list[Any]:
    """Gaussian elimination unrolled at trace time: one stencil per scalar
    operation, operations with constant zeros are eliminated.

    Arguments:
        a: Elements of the n x n matrix (row-major)
        b: Elements of the n x r right-hand sides (row-major)
        pivot: Partial pivoting by conditional swaps of rows

    Returns:
        Elements of the n x r solution (row-major)
    """
    def as_float(v: Any) -> Any:
        return to_float(v) if isinstance(v, value) else float(v)

    w = n + r
    rows = [[as_float(v) for v in (*a[i * n:(i + 1) * n], *b[i * r:(i + 1) * r])] for i in range(n)]
    inv_pivots: list[Any] = []

    for k in range(n):
        if pivot:
            # Move the row with the largest element of column k up to row k
            for i in range(k + 1, n):
                m = _gtabs(rows[i][k], rows[k][k])
                for j in range(k, w):
                    rows[k][j], rows[i][j] = _cswap(m, rows[k][j], rows[i][j])

        if not isinstance(rows[k][k], value) and rows[k][k] == 0:
            raise ValueError("Matrix is singular" if pivot else
                             f"Pivot element {k} is zero, pivoting is required to solve this system")

        inv = 1.0 / rows[k][k]
        inv_pivots.append(inv)
        for i in range(k + 1, n):
            if not isinstance(rows[i][k], value) and rows[i][k] == 0:
                continue
            f = rows[i][k] * inv
            for j in range(k + 1, w):
                rows[i][j] = rows[i][j] - f * rows[k][j]

    for i in reversed(range(n)):
        for c in range(n, w):
            s = rows[i][c]
            for q in range(i + 1, n):
                s = s - rows[i][q] * rows[q][c]
            rows[i][c] = s * inv_pivots[i]

    return [v if isinstance(v, value) else float(v) for i in range(n) for v in rows[i][n:]]


def _solve_arrays(a: array[Any], b: array[Any]) -> array[float]:
    """Solution of a x = b by the solve array stencil for an n x n array a
    and an array b with n or n x r elements. The result has the shape of b."""
    n = a.shape[0]
    r = b.size // n
    dims = array([n, r], 'int')
    work: array[float] = array([0.0] * (n * n), 'float')  # Memory for the eliminated matrix
    node = _add_array_op('solve_floatarr_floatarr',
                         [_cast_array(a, 'float').net, _cast_array(b, 'float').net, dims.net, work.net], n * r)
    assert isinstance(node.result, ArrayNet)
    return array._from_net(node.result, b.shape)


@overload
def solve(a: 'array[Any]', b: 'array[Any] | tensor[Any] | vector[Any]', pivot: bool = True) -> array[float]: ...
@overload
def solve(a: 'tensor[Any]', b: 'array[Any]', pivot: bool = True) -> array[float]: ...
@overload
def solve(a: 'tensor[Any]', b: 'vector[Any]', pivot: bool = True) -> vector[float]: ...
@overload
def solve(a: 'tensor[Any]', b: 'tensor[Any]', pivot: bool = True) -> tensor[float]: ...
def solve(a: Any, b: Any, pivot: bool = True) -> Any:
    """Solve the linear system a x = b for x (like numpy.linalg.solve).

    Arrays and packed tensors are solved by a single array stencil: Gaussian
    elimination with partial pivoting. Otherwise the elimination is unrolled
    into scalar operations, where operations with constant zeros are eliminated
    and pivot elements that are constants are selected without generating code.

    In both cases the execution time only depends on the size of the system,
    not on the values. A singular matrix results in inf or nan values.

    Arguments:
        a: Square matrix (2D tensor or array)
        b: Right-hand side: a vector, a 1D tensor or array with n elements
            or a 2D tensor or array with n rows (one system per column)
        pivot: Partial pivoting for the unrolled elimination. It can be
            disabled to reduce the number of operations for matrices that do
            not require it, like symmetric positive definite or diagonally
            dominant matrices. The array stencil does always pivot.

    Returns:
        Solution x with the shape of b: an array if a or b is an array,
        otherwise a vector if b is a vector, otherwise a tensor
    """
    if not isinstance(a, array | tensor):
        a = tensor(a)
    if not isinstance(b, array | tensor | vector):
        b = tensor(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"solve requires a square matrix, got shape {a.shape}")
    n: int = a.shape[0]
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError(f"Shape mismatch: matrix {a.shape} and right-hand side {b.shape}")

    if isinstance(a, array) or isinstance(b, array):
        return _solve_arrays(a if isinstance(a, array) else a._force_array(),
                             b if isinstance(b, array) else b._force_array())

    if not (a._is_constant() and b._is_constant()):
        arr_a, arr_b = a._get_array(), b._get_array()
        if arr_a is not None or arr_b is not None:
            packed = _solve_arrays(arr_a if arr_a is not None else a._force_array(),
                                   arr_b if arr_b is not None else b._force_array())
            return vector._from_array(packed) if isinstance(b, vector) else tensor._from_array(packed)

    values = _solve_values(a.values, b.values, n, len(b.values) // n, pivot)
    return vector(values) if isinstance(b, vector) else tensor(values, b.shape)
