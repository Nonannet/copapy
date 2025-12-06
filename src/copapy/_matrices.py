from . import value
from ._vectors import vector
from ._mixed import mixed_sum
from typing import TypeVar, Iterable, Any, overload, TypeAlias, Callable, Iterator, Generic
from ._helper_types import TNum

MatNumLike: TypeAlias = 'matrix[int] | matrix[float] | value[int] | value[float] | int | float'
MatIntLike: TypeAlias = 'matrix[int] | value[int] | int'
MatFloatLike: TypeAlias = 'matrix[float] | value[float] | float'
U = TypeVar("U", int, float)


class matrix(Generic[TNum]):
    """Mathematical matrix class supporting basic operations and interactions with values.
    """
    def __init__(self, values: Iterable[Iterable[TNum | value[TNum]]] | vector[TNum]):
        """Create a matrix with given values.

        Args:
            values: iterable of iterable of constant values
        """
        if isinstance(values, vector):
            rows = [values.values]
        else:
            rows = [tuple(row) for row in values]

        if rows:
            row_len = len(rows[0])
            assert all(len(row) == row_len for row in rows), "All rows must have the same length"
        self.values: tuple[tuple[value[TNum] | TNum, ...], ...] = tuple(rows)
        self.rows = len(self.values)
        self.cols = len(self.values[0]) if self.values else 0

    def __repr__(self) -> str:
        return f"matrix({self.values})"

    def __len__(self) -> int:
        """Return the number of rows in the matrix."""
        return self.rows

    @overload
    def __getitem__(self, key: int) -> vector[TNum]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> value[TNum] | TNum: ...
    def __getitem__(self, key: int | tuple[int, int]) -> Any:
        """Get a row as a vector or a specific element.
            Args:
                key: row index or (row, col) tuple
            
            Returns:
                vector if row index is given, else the element at (row, col)
        """
        if isinstance(key, tuple):
            assert len(key) == 2
            return self.values[key[0]][key[1]]
        else:
            return vector(self.values[key])

    def __iter__(self) -> Iterator[tuple[value[TNum] | TNum, ...]]:
        return iter(self.values)

    def __neg__(self) -> 'matrix[TNum]':
        return matrix((-a for a in row) for row in self.values)

    @overload
    def __add__(self: 'matrix[int]', other: MatFloatLike) -> 'matrix[float]': ...
    @overload
    def __add__(self: 'matrix[int]', other: MatIntLike) -> 'matrix[int]': ...
    @overload
    def __add__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __add__(self, other: MatNumLike) -> 'matrix[int] | matrix[float]': ...
    def __add__(self, other: MatNumLike) -> Any:
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(a + b for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(a + other for a in row)
            for row in self.values
        )

    @overload
    def __radd__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __radd__(self: 'matrix[int]', other: value[int] | int) -> 'matrix[int]': ...
    def __radd__(self, other: Any) -> Any:
        return self + other

    @overload
    def __sub__(self: 'matrix[int]', other: MatFloatLike) -> 'matrix[float]': ...
    @overload
    def __sub__(self: 'matrix[int]', other: MatIntLike) -> 'matrix[int]': ...
    @overload
    def __sub__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __sub__(self, other: MatNumLike) -> 'matrix[int] | matrix[float]': ...
    def __sub__(self, other: MatNumLike) -> Any:
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(a - b for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(a - other for a in row)
            for row in self.values
        )

    @overload
    def __rsub__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __rsub__(self: 'matrix[int]', other: value[int] | int) -> 'matrix[int]': ...
    def __rsub__(self, other: MatNumLike) -> Any:
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(b - a for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(other - a for a in row)
            for row in self.values
        )

    @overload
    def __mul__(self: 'matrix[int]', other: MatFloatLike) -> 'matrix[float]': ...
    @overload
    def __mul__(self: 'matrix[int]', other: MatIntLike) -> 'matrix[int]': ...
    @overload
    def __mul__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __mul__(self, other: MatNumLike) -> 'matrix[int] | matrix[float]': ...
    def __mul__(self, other: MatNumLike) -> Any:
        """Element-wise multiplication"""
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(a * b for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(a * other for a in row)
            for row in self.values
        )

    @overload
    def __rmul__(self: 'matrix[float]', other: MatNumLike) -> 'matrix[float]': ...
    @overload
    def __rmul__(self: 'matrix[int]', other: value[int] | int) -> 'matrix[int]': ...
    def __rmul__(self, other: MatNumLike) -> Any:
        return self * other

    def __truediv__(self, other: MatNumLike) -> 'matrix[float]':
        """Element-wise division"""
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(a / b for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(a / other for a in row)
            for row in self.values
        )

    def __rtruediv__(self, other: MatNumLike) -> 'matrix[float]':
        if isinstance(other, matrix):
            assert self.rows == other.rows and self.cols == other.cols, \
                "Matrices must have the same dimensions"
            return matrix(
                tuple(b / a for a, b in zip(row1, row2))
                for row1, row2 in zip(self.values, other.values)
            )
        return matrix(
            tuple(other / a for a in row)
            for row in self.values
        )

    @overload
    def __matmul__(self: 'matrix[TNum]', other: 'vector[TNum]') -> 'vector[TNum]': ...
    @overload
    def __matmul__(self: 'matrix[TNum]', other: 'matrix[TNum]') -> 'matrix[TNum]': ...
    def __matmul__(self: 'matrix[TNum]', other: 'matrix[TNum] | vector[TNum]') -> 'matrix[TNum] | vector[TNum]':
        """Matrix multiplication using @ operator"""
        if isinstance(other, vector):
            assert self.cols == len(other.values), \
                f"Matrix columns ({self.cols}) must match vector length ({len(other.values)})"
            vec_result = (mixed_sum(a * b for a, b in zip(row, other.values)) for row in self.values)
            return vector(vec_result)
        else:
            assert isinstance(other, matrix), "Cannot multiply matrix with {type(other)}"
            assert self.cols == other.rows, \
                f"Matrix columns ({self.cols}) must match other matrix rows ({other.rows})"
            result: list[list[TNum | value[TNum]]] = []
            for row in self.values:
                new_row: list[TNum | value[TNum]] = []
                for col_idx in range(other.cols):
                    col = tuple(other.values[i][col_idx] for i in range(other.rows))
                    element = sum(a * b for a, b in zip(row, col))
                    new_row.append(element)
                result.append(new_row)
            return matrix(result)

    def transpose(self) -> 'matrix[TNum]':
        """Return the transpose of the matrix."""
        if not self.values:
            return matrix([])
        return matrix(
            tuple(self.values[i][j] for i in range(self.rows))
            for j in range(self.cols)
        )
    
    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the matrix as (rows, cols)."""
        return (self.rows, self.cols)

    @property
    def T(self) ->  'matrix[TNum]':
        return self.transpose()

    def row(self, index: int) -> vector[TNum]:
        """Get a row as a vector."""
        assert 0 <= index < self.rows, f"Row index {index} out of bounds"
        return vector(self.values[index])

    def col(self, index: int) -> vector[TNum]:
        """Get a column as a vector."""
        assert 0 <= index < self.cols, f"Column index {index} out of bounds"
        return vector(self.values[i][index] for i in range(self.rows))

    @overload
    def trace(self: 'matrix[TNum]') -> TNum | value[TNum]: ...
    @overload
    def trace(self: 'matrix[int]') -> int | value[int]: ...
    @overload
    def trace(self: 'matrix[float]') -> float | value[float]: ...
    def trace(self) -> Any:
        """Calculate the trace (sum of diagonal elements)."""
        assert self.rows == self.cols, "Trace is only defined for square matrices"
        return mixed_sum(self.values[i][i] for i in range(self.rows))

    @overload
    def sum(self: 'matrix[TNum]') -> TNum | value[TNum]: ...
    @overload
    def sum(self: 'matrix[int]') -> int | value[int]: ...
    @overload
    def sum(self: 'matrix[float]') -> float | value[float]: ...
    def sum(self) -> Any:
        """Calculate the sum of all elements."""
        return mixed_sum(a for row in self.values for a in row)

    def map(self, func: Callable[[Any], value[U] | U]) -> 'matrix[U]':
        """Applies a function to each element of the matrix and returns a new matrix."""
        return matrix(
            tuple(func(a) for a in row)
            for row in self.values
        )

    def homogenize(self) -> 'matrix[TNum]':
        """Convert all elements to copapy values if any element is a copapy value."""
        if any(isinstance(val, value) for row in self.values for val in row):
            return matrix(
                tuple(value(val) if not isinstance(val, value) else val for val in row)
                for row in self.values
            )
        else:
            return self


def identity(size: int) -> matrix[int]:
    """Create an identity matrix of given size."""
    return matrix(
        tuple(1 if i == j else 0 for j in range(size))
        for i in range(size)
    )


def zeros(rows: int, cols: int) -> matrix[int]:
    """Create a zero matrix of given dimensions."""
    return matrix(
        tuple(0 for _ in range(cols))
        for _ in range(rows)
    )


def ones(rows: int, cols: int) -> matrix[int]:
    """Create a matrix of ones with given dimensions."""
    return matrix(
        tuple(1 for _ in range(cols))
        for _ in range(rows)
    )


def eye(rows: int, cols: int | None = None) -> matrix[int]:
    """Create a matrix with ones on the diagonal and zeros elsewhere."""
    cols = cols if cols else rows
    return matrix(
        tuple(1 if i == j else 0 for j in range(cols))
        for i in range(rows)
    )


@overload
def diagonal(vec: 'vector[int]') -> matrix[int]: ...
@overload
def diagonal(vec: 'vector[float]') -> matrix[float]: ...
def diagonal(vec: vector[Any]) -> matrix[Any]:
    """Create a diagonal matrix from a vector."""
    size = len(vec)

    return matrix(
        tuple(vec[i] if i == j else 0 for j in range(size))
        for i in range(size)
    )
