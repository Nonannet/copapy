from typing import Any, Sequence, overload
from functools import lru_cache
from ._basic_types import value, ArrayNet, add_op, to_float, value_from_number, select_by_mask
from ._arrays import array, _add_array_op
from ._vectors import vector
from ._tensors import tensor
from ._casts import _cast_array

unroll_threshold: int = 16
"""Tensors are solved by the array stencil if more elements of the matrix than
unroll_threshold are copapy values, otherwise by the unrolled elimination"""

# Largest matrix size det expands into scalar operations
_DET_UNROLL_SIZE = 4


def _gtabs(x: Any, y: Any) -> Any:
    """Int mask with all bits set (-1) if |x| > |y|, otherwise 0"""
    if not isinstance(x, value) and x == 0:
        return 0
    if isinstance(x, value) or isinstance(y, value):
        return add_op('gtabs', [x, y])
    return -int(abs(x) > abs(y))


def _cswap(m: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """(y, x) if the mask m is set, otherwise (x, y). Masks known at trace
    time and equal constants (e.g. zeros in both rows) generate no code."""
    if not isinstance(m, value):
        return (y, x) if m else (x, y)
    if not isinstance(x, value) and not isinstance(y, value) and x == y:
        return x, y
    return select_by_mask(m, y, x), select_by_mask(m, x, y)


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


def _det_array(a: array[Any]) -> value[float]:
    """Determinant of an n x n array by the det array stencil"""
    n = a.shape[0]
    work: array[float] = array([0.0] * (n * n), 'float')  # Memory for the eliminated matrix
    node = _add_array_op('det_floatarr', [_cast_array(a, 'float').net, value_from_number(n).net, work.net])
    return value(node.result)


def _det_values(a: Sequence[Any], n: int) -> Any:
    """Determinant by the Laplace expansion along the rows unrolled at trace
    time. Each minor is computed once, products with constant zeros are
    eliminated.

    Arguments:
        a: Elements of the n x n matrix (row-major)

    Returns:
        Determinant as copapy value or as number if no value is involved
    """
    elements = [to_float(v) if isinstance(v, value) else float(v) for v in a]

    @lru_cache(maxsize=None)
    def minor(row: int, cols: tuple[int, ...]) -> Any:
        """Determinant of the rows from row on and the given columns"""
        if row == n:
            return 1.0
        acc: Any = 0.0
        for pos, col in enumerate(cols):
            x = elements[row * n + col]
            if not isinstance(x, value) and x == 0:
                continue
            term = x * minor(row + 1, cols[:pos] + cols[pos + 1:])
            if not pos % 2:
                acc = acc + term
            elif not isinstance(acc, value) and acc == 0:
                acc = -term
            else:
                acc = acc - term
        return acc

    return minor(0, tuple(range(n)))


def _det_constants(a: Sequence[Any], n: int) -> float:
    """Determinant of numbers by Gaussian elimination with partial pivoting"""
    rows = [[float(v) for v in a[i * n:(i + 1) * n]] for i in range(n)]
    result = 1.0
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(rows[i][k]))
        if rows[p][k] == 0:
            return 0.0
        if p != k:
            rows[k], rows[p] = rows[p], rows[k]
            result = -result
        result *= rows[k][k]
        for i in range(k + 1, n):
            f = rows[i][k] / rows[k][k]
            rows[i] = [x - f * y for x, y in zip(rows[i], rows[k])]
    return result


def _square_matrix(a: Any) -> Any:
    """The argument as array or tensor, checked to be a square matrix"""
    if not isinstance(a, array | tensor):
        a = tensor(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"requires a square matrix, got shape {a.shape}")
    return a


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

    Arrays and tensors with more than 16 copapy values in the matrix a are
    solved by a single array stencil: Gaussian elimination with partial
    pivoting. Otherwise the elimination is unrolled into scalar operations,
    where operations with constant zeros are eliminated and pivot elements
    that are constants are selected without generating code. A tensor created
    with packed=True is always solved by the array stencil, a matrix created
    with packed=False is always unrolled.

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
    a = _square_matrix(a)
    if not isinstance(b, array | tensor | vector):
        b = tensor(b)
    n: int = a.shape[0]
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError(f"Shape mismatch: matrix {a.shape} and right-hand side {b.shape}")

    if isinstance(a, array) or isinstance(b, array):
        return _solve_arrays(a if isinstance(a, array) else a._force_array(),
                             b if isinstance(b, array) else b._force_array())

    if not (a._is_constant() and b._is_constant()) and a._packed is not False:
        # The size of the unrolled elimination depends on the elements of the
        # matrix that are not known at trace time
        if a._packed or b._packed or sum(1 for v in a.values if isinstance(v, value)) > unroll_threshold:
            packed = _solve_arrays(a._force_array(), b._force_array())
            return vector._from_array(packed) if isinstance(b, vector) else tensor._from_array(packed)

    values = _solve_values(a.values, b.values, n, len(b.values) // n, pivot)
    return vector(values) if isinstance(b, vector) else tensor(values, b.shape)


@overload
def inv(a: 'array[Any]', pivot: bool = True) -> array[float]: ...
@overload
def inv(a: 'tensor[Any]', pivot: bool = True) -> tensor[float]: ...
def inv(a: Any, pivot: bool = True) -> Any:
    """Inverse of a square matrix (like numpy.linalg.inv).

    The inverse is the solution of a x = identity computed by solve: by the
    array stencil or by the unrolled elimination, selected like in solve. A
    singular matrix results in inf or nan values. To solve a linear system
    solve is more accurate and requires fewer operations than a multiplication
    with the inverse.

    Arguments:
        a: Square matrix (2D tensor or array)
        pivot: Partial pivoting for the unrolled elimination, see solve

    Returns:
        Inverse of a: an array if a is an array, otherwise a tensor
    """
    a = _square_matrix(a)
    n: int = a.shape[0]
    identity = [[float(i == j) for j in range(n)] for i in range(n)]
    if isinstance(a, array):
        return _solve_arrays(a, array(identity, 'float'))
    return solve(a, tensor(identity), pivot)


@overload
def det(a: 'array[Any]') -> value[float]: ...
@overload
def det(a: 'tensor[Any]') -> 'value[float] | float': ...
def det(a: Any) -> Any:
    """Determinant of a square matrix (like numpy.linalg.det).

    Tensors up to 4 x 4 are expanded into scalar operations: additions and
    multiplications only, products with constant zeros are eliminated. Arrays,
    larger tensors and tensors created with packed=True are computed by a
    single array stencil: Gaussian elimination with partial pivoting. A
    singular matrix results in 0. Tensors of numbers are evaluated at trace
    time.

    Arguments:
        a: Square matrix (2D tensor or array)

    Returns:
        Determinant as copapy value, as number if a is a tensor of numbers
    """
    a = _square_matrix(a)
    n: int = a.shape[0]
    if isinstance(a, array):
        return _det_array(a)
    if a._is_constant():
        return _det_values(a.values, n) if n <= _DET_UNROLL_SIZE else _det_constants(a.values, n)
    if n > _DET_UNROLL_SIZE or a._packed:
        return _det_array(a._force_array())
    return _det_values(a.values, n)
