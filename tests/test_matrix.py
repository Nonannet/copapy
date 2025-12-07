import copapy as cp
import pytest


def test_matrix_init():
    """Test basic matrix initialization"""
    m1 = cp.matrix([[1, 2, 3], [4, 5, 6]])
    assert m1.rows == 2
    assert m1.cols == 3
    assert m1[0] == (1, 2, 3)
    assert m1[1] == (4, 5, 6)


def test_matrix_with_variables():
    """Test matrix initialization with variables"""
    m1 = cp.matrix([[cp.value(1), 2], [3, cp.value(4)]])
    assert m1.rows == 2
    assert m1.cols == 2
    assert isinstance(m1[0][0], cp.value)
    assert isinstance(m1[1][1], cp.value)


def test_matrix_addition():
    """Test matrix addition"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = cp.matrix([[5, 6], [7, 8]])
    m3 = m1 + m2

    assert m3[0] == (6, 8)
    assert m3[1] == (10, 12)


def test_matrix_scalar_addition():
    """Test matrix addition with scalar"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = m1 + 5

    assert m2[0] == (6, 7)
    assert m2[1] == (8, 9)


def test_matrix_subtraction():
    """Test matrix subtraction"""
    m1 = cp.matrix([[5, 6], [7, 8]])
    m2 = cp.matrix([[1, 2], [3, 4]])
    m3 = m1 - m2

    assert m3[0] == (4, 4)
    assert m3[1] == (4, 4)


def test_matrix_scalar_subtraction():
    """Test matrix subtraction with scalar"""
    m1 = cp.matrix([[5, 6], [7, 8]])
    m2 = m1 - 2

    assert m2[0] == (3, 4)
    assert m2[1] == (5, 6)


def test_matrix_negation():
    """Test matrix negation"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = -m1

    assert m2[0] == (-1, -2)
    assert m2[1] == (-3, -4)


def test_matrix_element_wise_multiplication():
    """Test element-wise matrix multiplication"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = cp.matrix([[5, 6], [7, 8]])
    m3 = m1 * m2

    assert m3[0] == (5, 12)
    assert m3[1] == (21, 32)


def test_matrix_scalar_multiplication():
    """Test matrix multiplication with scalar"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = m1 * 3

    assert m2[0] == (3, 6)
    assert m2[1] == (9, 12)


def test_matrix_element_wise_division():
    """Test element-wise matrix division"""
    m1 = cp.matrix([[6.0, 8.0], [12.0, 16.0]])
    m2 = cp.matrix([[2.0, 2.0], [3.0, 4.0]])
    m3 = m1 / m2

    assert m3[0][0] == pytest.approx(3.0)  # pyright: ignore[reportUnknownMemberType]
    assert m3[0][1] == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]
    assert m3[1][0] == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]
    assert m3[1][1] == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]


def test_matrix_scalar_division():
    """Test matrix division by scalar"""
    m1 = cp.matrix([[6.0, 8.0], [12.0, 16.0]])
    m2 = m1 / 2.0

    assert m2[0] == pytest.approx((3.0, 4.0))  # pyright: ignore[reportUnknownMemberType]
    assert m2[1] == pytest.approx((6.0, 8.0))  # pyright: ignore[reportUnknownMemberType]


def test_matrix_vector_multiplication():
    """Test matrix-vector multiplication using @ operator"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    v = cp.vector([7, 8, 9])
    result = m @ v

    assert isinstance(result, cp.vector)
    assert len(result.values) == 2
    assert result.values[0] == 1*7 + 2*8 + 3*9
    assert result.values[1] == 4*7 + 5*8 + 6*9


def test_matrix_matrix_multiplication():
    """Test matrix-matrix multiplication using @ operator"""
    m1 = cp.matrix([[1, 2], [3, 4]])
    m2 = cp.matrix([[5, 6], [7, 8]])
    result = m1 @ m2

    assert isinstance(result, cp.matrix)
    assert result.rows == 2
    assert result.cols == 2
    assert result[0][0] == 1*5 + 2*7
    assert result[0][1] == 1*6 + 2*8
    assert result[1][0] == 3*5 + 4*7
    assert result[1][1] == 3*6 + 4*8


def test_matrix_transpose():
    """Test matrix transpose"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    mt = m.transpose()

    assert mt.rows == 3
    assert mt.cols == 2
    assert mt[0] == (1, 4)
    assert mt[1] == (2, 5)
    assert mt[2] == (3, 6)


def test_matrix_transpose_property():
    """Test matrix transpose using .T property"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    mt = m.T

    assert mt.rows == 3
    assert mt.cols == 2
    assert mt[0] == (1, 4)


def test_matrix_row_access():
    """Test getting a row as a vector"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    row0 = m.row(0)

    assert isinstance(row0, cp.vector)
    assert row0.values == (1, 2, 3)


def test_matrix_col_access():
    """Test getting a column as a vector"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    col1 = m.col(1)

    assert isinstance(col1, cp.vector)
    assert col1.values == (2, 5)


def test_matrix_trace():
    """Test matrix trace (sum of diagonal elements)"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    trace = m.trace()

    assert trace == 1 + 5 + 9


def test_matrix_sum():
    """Test sum of all matrix elements"""
    m = cp.matrix([[1, 2, 3], [4, 5, 6]])
    total = m.sum()

    assert total == 1 + 2 + 3 + 4 + 5 + 6


def test_matrix_map():
    """Test mapping a function over matrix elements"""
    m = cp.matrix([[1, 2], [3, 4]])
    m_doubled = m.map(lambda x: x * 2)

    assert m_doubled[0] == (2, 4)
    assert m_doubled[1] == (6, 8)


def test_matrix_homogenize():
    """Test homogenizing matrix (converting to all variables)"""
    m = cp.matrix([[1, cp.value(2)], [3, 4]])
    m_homo = m.homogenize()

    for row in m_homo:
        for elem in row:
            assert isinstance(elem, cp.value)


def test_identity_matrix():
    """Test identity matrix creation"""
    m = cp.identity(3)

    assert m.rows == 3
    assert m.cols == 3
    assert m[0] == (1, 0, 0)
    assert m[1] == (0, 1, 0)
    assert m[2] == (0, 0, 1)


def test_zeros_matrix():
    """Test zeros matrix creation"""
    m = cp.zeros(2, 3)

    assert m.rows == 2
    assert m.cols == 3
    assert m[0] == (0, 0, 0)
    assert m[1] == (0, 0, 0)


def test_ones_matrix():
    """Test ones matrix creation"""
    m = cp.ones(2, 3)

    assert m.rows == 2
    assert m.cols == 3
    assert m[0] == (1, 1, 1)
    assert m[1] == (1, 1, 1)


def test_diagonal_matrix():
    """Test diagonal matrix creation from vector"""
    v = cp.vector([1, 2, 3])
    m = cp.diagonal(v)

    assert m.rows == 3
    assert m.cols == 3
    assert m[0] == (1, 0, 0)
    assert m[1] == (0, 2, 0)
    assert m[2] == (0, 0, 3)


def test_matrix_with_variables_compiled():
    """Test matrix operations with variables in compilation"""
    m = cp.matrix([[cp.value(1), 2], [3, cp.value(4)]])
    v = cp.vector([cp.value(5), 6])
    result = m @ v

    # result[0] = 1*5 + 2*6 = 17
    # result[1] = 3*5 + 4*6 = 39

    tg = cp.Target()
    tg.compile(result)
    tg.run()

    assert tg.read_value(result.values[0]) == pytest.approx(17)  # pyright: ignore[reportUnknownMemberType]
    assert tg.read_value(result.values[1]) == pytest.approx(39)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
