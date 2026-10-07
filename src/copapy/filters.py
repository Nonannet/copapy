from ._vectors import vector
from ._tensors import tensor
from ._basic_types import value, unifloat, NumLike
from ._arrays import array, ArrayType, _size
from ._nn import _float_array, _conv
from ._casts import to_float
from ._sorting import _as_array
from typing import Any, Sequence, TypeAlias, overload
import math

__all__ = ["median", "convolve", "gaussian_filter", "mean"]

ConvLike: TypeAlias = 'ArrayType[Any] | array[Any] | Sequence[float | value[Any]]'


def median(input_vector: ArrayType[float]) -> unifloat:
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
        return float(data[n // 2]) if n % 2 else (data[n // 2 - 1] + data[n // 2]) / 2
    sorted_arr = _as_array(input_vector).sort()
    if n % 2:
        return to_float(sorted_arr.element(n // 2))
    return (sorted_arr.element(n // 2 - 1) + sorted_arr.element(n // 2)) * 0.5


def _constants(x: Any) -> list[Any] | None:
    """Elements of a constant 1D input (numbers only), otherwise None"""
    if isinstance(x, ArrayType):
        return list(x.values) if x._is_constant() and x.ndim == 1 else None
    if isinstance(x, Sequence) and all(isinstance(e, int | float) for e in x):
        return list(x)
    return None


@overload
def convolve(a: vector[Any], v: ConvLike, mode: str = 'full') -> vector[float]: ...
@overload
def convolve(a: tensor[Any], v: ConvLike, mode: str = 'full') -> tensor[float]: ...
@overload
def convolve(a: 'array[Any] | Sequence[float | value[Any]]', v: ConvLike, mode: str = 'full') -> array[float]: ...
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


def _gaussian_kernel(sigma: float, truncate: float) -> list[float]:
    """Normalized samples of a Gaussian up to truncate standard deviations"""
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return [1.0]
    weights = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    total = sum(weights)
    return [w / total for w in weights]


@overload
def gaussian_filter(a: vector[Any], sigma: float | Sequence[float], truncate: float = 4.0) -> vector[float]: ...
@overload
def gaussian_filter(a: tensor[Any], sigma: float | Sequence[float], truncate: float = 4.0) -> tensor[float]: ...
@overload
def gaussian_filter(a: 'array[Any] | Sequence[Any]', sigma: float | Sequence[float], truncate: float = 4.0) -> array[float]: ...
def gaussian_filter(a: Any, sigma: float | Sequence[float], truncate: float = 4.0) -> Any:
    """
    Gaussian filter for 1D, 2D and 3D inputs. The filter is separable: one
    1D convolution is computed for each axis. The input is extended with zeros
    beyond its edges (like the mode 'constant' of scipy.ndimage.gaussian_filter).

    Arguments:
        a: Input vector, tensor, array or nested sequences
        sigma: Standard deviation of the Gaussian in elements, a single number
            for all axes or one number per axis. An axis with sigma 0 is not filtered.
        truncate: The kernel is truncated at this many standard deviations.

    Returns:
        Filtered values with the shape and type of the input.
    """
    xa = _float_array(a, 'a')
    shape = xa.shape
    sigmas = [float(sigma)] * xa.ndim if isinstance(sigma, int | float) else [float(s) for s in sigma]
    if len(sigmas) != xa.ndim:
        raise ValueError(f"Expected one sigma or one for each of the {xa.ndim} axes, got {len(sigmas)}")
    if any(s < 0 for s in sigmas) or truncate < 0:
        raise ValueError("sigma and truncate must not be negative")

    for axis, s in enumerate(sigmas):
        kernel = _gaussian_kernel(s, truncate)
        m = len(kernel)
        if m == 1:
            continue
        # The axis to filter is the height of the convolution, the axes in front
        # of it are the batch and the axes behind it the width
        n, width = shape[axis], _size(shape[axis + 1:])
        weight: array[Any] = array(kernel).reshape(1, 1, m, 1)
        xa = _conv(xa.reshape(-1, 1, n, width), weight, None, (1, 1), (m // 2, 0), (1, 1), out_size=(n, width))
    xa = xa.reshape(*shape)
    return type(a)._from_array(xa) if isinstance(a, ArrayType) else xa


def mean(input_vector: vector[Any]) -> unifloat:
    """
    Applies a mean filter to the input vector and returns the mean as a unifloat.

    Arguments:
        input_vector (vector): The input vector containing numerical values.

    Returns:
        unifloat: The mean value of the input vector.
    """
    return input_vector.sum() / len(input_vector)
