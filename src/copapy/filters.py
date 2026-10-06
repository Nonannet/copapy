from . import vector
from ._basic_types import unifloat
from ._arrays import array, ArrayType
from ._nn import _float_array, _conv
from ._helper_types import TNum
from typing import Any, Sequence, TypeVar, cast

TArray = TypeVar('TArray', bound=ArrayType[Any])


def _as_array(input_vector: ArrayType[Any]) -> array[Any]:
    """Elements of the input as array, packed independent of the pack threshold"""
    if input_vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector or tensor, got shape {input_vector.shape}")
    if not input_vector.shape[0]:
        raise ValueError("Empty vectors are not supported")
    return input_vector._force_array()


def sort(input_vector: TArray) -> TArray:
    """
    Sort the elements of a vector in ascending order.

    Arguments:
        input_vector: The input vector (or 1D tensor) containing numerical values.

    Returns:
        Sorted vector.
    """
    if input_vector._is_constant():
        constants: list[Any] = list(input_vector.values)
        array_type: Any = type(input_vector)
        return cast(TArray, array_type(sorted(constants)))
    return type(input_vector)._from_array(_as_array(input_vector).sort())


def argsort(input_vector: ArrayType[TNum]) -> vector[int]:
    """
    Perform an indirect sort. It returns a vector of indices that index data
    in sorted order. Equal elements are in order of their index (like a stable sort).

    Arguments:
        input_vector: The input vector (or 1D tensor) containing numerical values.

    Returns:
        Index vector.
    """
    if input_vector._is_constant():
        data: list[Any] = list(input_vector.values)
        indices: list[int] = sorted(range(len(data)), key=lambda i: data[i])
        return vector(indices)
    return vector._from_array(_as_array(input_vector).argsort())


def median(input_vector: ArrayType[TNum]) -> Any:
    """
    Median of the elements of a vector: the middle element of the sorted
    elements or the average of the two middle elements for an even number
    of elements.

    Arguments:
        input_vector: The input vector (or 1D tensor) containing numerical values.

    Returns:
        The median value of the input vector.
    """
    n = len(input_vector.values)
    if input_vector._is_constant():
        data: list[Any] = sorted(input_vector.values)
        return data[n // 2] if n % 2 else (data[n // 2 - 1] + data[n // 2]) / 2
    sorted_arr = _as_array(input_vector).sort()
    if n % 2:
        return sorted_arr.element(n // 2)
    return (sorted_arr.element(n // 2 - 1) + sorted_arr.element(n // 2)) * 0.5


def _constants(x: Any) -> list[Any] | None:
    """Elements of a constant 1D input (numbers only), otherwise None"""
    if isinstance(x, ArrayType):
        return list(x.values) if x._is_constant() and x.ndim == 1 else None
    if isinstance(x, Sequence) and all(isinstance(e, int | float) for e in x):
        return list(x)
    return None


def convolve(a: Any, v: Any, mode: str = 'full') -> Any:
    """
    Discrete linear convolution of two 1D sequences.

    Arguments:
        a: First 1D input
        v: Second 1D input
        mode: 'full' (length N + M - 1), 'same' (length max(M, N), centered)
            or 'valid' (length max(M, N) - min(M, N) + 1, only complete overlaps).

    Returns:
        Result of the convolution with the size determined by the mode.
    """
    if mode not in ('full', 'same', 'valid'):
        raise ValueError(f"mode must be 'full', 'same' or 'valid', not {mode!r}")

    const_a, const_v = _constants(a), _constants(v)
    if const_a is not None and const_v is not None:
        # Constants are evaluated at trace time
        data = _convolve_numbers(const_a, const_v, mode)
        array_type: Any = type(a)
        return array_type(data) if isinstance(a, ArrayType) else array(data, 'float')

    xa, xv = _float_array(a, 'a'), _float_array(v, 'v')
    if xa.ndim != 1 or xv.ndim != 1:
        raise ValueError(f"convolve requires 1D inputs, got shapes {xa.shape} and {xv.shape}")
    if not xa.size or not xv.size:
        raise ValueError("Empty inputs are not supported")
    if xv.size > xa.size:
        xa, xv, const_v = xv, xa, const_a  # Convolution is commutative
    n, m = xa.size, xv.size

    # Convolution = cross-correlation (computed by the stencil) with the reversed kernel
    kernel: array[Any] = array([float(c) for c in reversed(const_v)]) if const_v is not None else xv[::-1]
    left, length = {'full': (m - 1, n + m - 1), 'same': (m - 1 - (m - 1) // 2, n), 'valid': (0, n - m + 1)}[mode]
    ret = _conv(xa.reshape(1, 1, 1, n), kernel.reshape(1, 1, 1, m), None, (1, 1), (0, left), (1, 1),
                out_size=(1, length)).reshape(length)
    return type(a)._from_array(ret) if isinstance(a, ArrayType) else ret


def _convolve_numbers(a: list[Any], v: list[Any], mode: str) -> list[float]:
    if not a or not v:
        raise ValueError("Empty inputs are not supported")
    if len(v) > len(a):
        a, v = v, a
    n, m = len(a), len(v)
    full = [float(sum(a[i] * v[k - i] for i in range(n) if 0 <= k - i < m)) for k in range(n + m - 1)]
    start, length = {'full': (0, n + m - 1), 'same': ((m - 1) // 2, n), 'valid': (m - 1, n - m + 1)}[mode]
    return full[start:start + length]


def mean(input_vector: vector[Any]) -> unifloat:
    """
    Applies a mean filter to the input vector and returns the mean as a unifloat.

    Arguments:
        input_vector (vector): The input vector containing numerical values.

    Returns:
        unifloat: The mean value of the input vector.
    """
    return input_vector.sum() / len(input_vector)
