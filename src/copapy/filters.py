from ._vectors import vector
from ._tensors import tensor
from ._basic_types import value, unifloat, NumLike
from ._arrays import array, ArrayType, _size
from ._nn import _float_array, _conv
from ._casts import to_float
from ._sorting import _as_array
from typing import Any, Sequence, TypeAlias, overload
import math

__all__ = ["median", "convolve", "firwin", "lowpass", "highpass", "bandpass","gaussian_filter","mean"]

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


_WINDOWS: dict[str, tuple[float, ...]] = {
    # Coefficients a_k of the cosine sum: sum((-1)^k * a_k * cos(2 * pi * k * n / (N - 1)))
    'rectangular': (1.0,),
    'hann': (0.5, 0.5),
    'hamming': (0.54, 0.46),
    'blackman': (0.42, 0.5, 0.08),
}


def firwin(numtaps: int, cutoff: float | Sequence[float], pass_zero: str = 'lowpass',
           window: str = 'hamming', fs: float = 2.0) -> list[float]:
    """
    Coefficients of a linear phase FIR filter designed by the window method
    (like scipy.signal.firwin). The filter is applied with convolve:

        >>> taps = filters.firwin(31, 50.0, fs=1000.0)
        >>> filtered = filters.convolve(samples, taps, 'same')

    Arguments:
        numtaps: Number of coefficients, the delay of the filter is (numtaps - 1) / 2
            samples. It must be odd for a high-pass.
        cutoff: Cutoff frequency (half amplitude) in the unit of fs, two
            frequencies (lower, upper) for a band-pass.
        pass_zero: 'lowpass', 'highpass' or 'bandpass'
        window: 'hamming', 'hann', 'blackman' or 'rectangular'
        fs: Sampling frequency, cutoff is relative to the Nyquist frequency
            for the default of 2.

    Returns:
        Coefficients scaled to a gain of 1 at frequency 0 (low-pass), at the
        Nyquist frequency (high-pass) or at the center of the band (band-pass).
    """
    if pass_zero not in ('lowpass', 'highpass', 'bandpass'):
        raise ValueError(f"pass_zero must be 'lowpass', 'highpass' or 'bandpass', not {pass_zero!r}")
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {', '.join(_WINDOWS)}, not {window!r}")
    if numtaps < 1:
        raise ValueError("numtaps must be at least 1")
    if pass_zero == 'highpass' and numtaps % 2 == 0:
        raise ValueError("A high-pass requires an odd number of taps")

    # Band edges relative to the Nyquist frequency
    edges = [2.0 * c / fs for c in ([cutoff] if isinstance(cutoff, int | float) else cutoff)]
    if len(edges) != (2 if pass_zero == 'bandpass' else 1):
        raise ValueError(f"A {pass_zero} requires {'two cutoff frequencies' if pass_zero == 'bandpass' else 'one cutoff frequency'}")
    if not all(0.0 < e < 1.0 for e in edges) or edges != sorted(set(edges)):
        raise ValueError("Cutoff frequencies must be increasing and between 0 and fs / 2")
    left, right = {'lowpass': (0.0, edges[0]), 'highpass': (edges[0], 1.0), 'bandpass': (edges[0], edges[-1])}[pass_zero]

    def sinc(x: float) -> float:
        return math.sin(math.pi * x) / (math.pi * x) if x else 1.0

    center = 0.5 * (numtaps - 1)
    taps: list[float] = []
    for n in range(numtaps):
        w = sum((-1) ** k * a * math.cos(2 * math.pi * k * n / (numtaps - 1)) for k, a in enumerate(_WINDOWS[window])) if numtaps > 1 else 1.0
        taps.append(w * (right * sinc(right * (n - center)) - left * sinc(left * (n - center))))

    # Gain of 1 in the pass band
    f0 = 0.0 if left == 0.0 else 1.0 if right == 1.0 else 0.5 * (left + right)
    gain = sum(t * math.cos(math.pi * (n - center) * f0) for n, t in enumerate(taps))
    return [t / gain for t in taps]


# Width of the transition between pass band and stop band multiplied by numtaps / fs
_TRANSITION_WIDTHS = {'rectangular': 0.9, 'hann': 3.1, 'hamming': 3.3, 'blackman': 5.5}


def _estimate_numtaps(cutoff: float | Sequence[float], window: str, fs: float, width: float | None) -> int:
    """Odd number of coefficients for a transition of the given width, by default
    half the distance of the cutoff to the nearest band edge"""
    if window not in _TRANSITION_WIDTHS:
        raise ValueError(f"window must be one of {', '.join(_TRANSITION_WIDTHS)}, not {window!r}")
    if width is None:
        edges = [0.0, *([float(cutoff)] if isinstance(cutoff, int | float) else cutoff), 0.5 * fs]
        width = 0.5 * min(b - a for a, b in zip(edges, edges[1:]))
        if width <= 0:
            raise ValueError("Cutoff frequencies must be increasing and between 0 and fs / 2")
    elif width <= 0:
        raise ValueError("width must be positive")
    return math.ceil(_TRANSITION_WIDTHS[window] * fs / width - 1e-9) | 1


def _fir_filter(a: Sequence | array, cutoff: float | Sequence[float], pass_zero: str, numtaps: int | None,
                window: str, fs: float, width: float | None) -> Any:
    """Filtered samples aligned with the input (delay of the filter removed)"""
    if numtaps is None:
        numtaps = _estimate_numtaps(cutoff, window, fs, width)
    size = len(a.values) if isinstance(a, ArrayType) else a.size if isinstance(a, array) else len(a)
    if numtaps > size:
        raise ValueError(f"numtaps ({numtaps}) must not exceed the number of samples ({size}), "
                         "a smaller numtaps or a larger width gives a shorter filter")
    return convolve(a, firwin(numtaps, cutoff, pass_zero, window, fs), 'same')


@overload
def lowpass(a: vector[Any], cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> vector[float]: ...
@overload
def lowpass(a: tensor[Any], cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> tensor[float]: ...
@overload
def lowpass(a: 'array[Any] | Sequence[float | value[Any]]', cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> array[float]: ...
def lowpass(a: Any, cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> Any:
    """
    Low-pass filter (firwin FIR filter) with no delay, the input is extended
    with zeros beyond its edges, which affects the first and last
    (numtaps - 1) / 2 samples.

    Arguments:
        a: Samples as 1D vector, tensor, array or sequence
        cutoff: Cutoff frequency (halved amplitude) in the unit of fs
        numtaps: Number of filter coefficients, more coefficients give a
            steeper transition between pass band and stop band.
        window: 'hamming', 'hann', 'blackman' or 'rectangular'
        fs: Sampling frequency, cutoff is relative to the Nyquist frequency.
        width: Width of the transition between pass band and stop band in the
            unit of fs, used to estimate numtaps if it is not given. The default
            is half the distance of the cutoff to the nearest band edge.

    Returns:
        Filtered samples with the length and type of the input.
    """
    return _fir_filter(a, cutoff, 'lowpass', numtaps, window, fs, width)


@overload
def highpass(a: vector[Any], cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> vector[float]: ...
@overload
def highpass(a: tensor[Any], cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> tensor[float]: ...
@overload
def highpass(a: 'array[Any] | Sequence[float | value[Any]]', cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> array[float]: ...
def highpass(a: Any, cutoff: float, numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> Any:
    """
    High-pass filter (firwin FIR filter) with no delay, the input is extended
    with zeros beyond its edges, which affects the first and last
    (numtaps - 1) / 2 samples.

    Arguments:
        a: Samples as 1D vector, tensor, array or sequence
        cutoff: Cutoff frequency (halved amplitude) in the unit of fs
        numtaps: Odd number of filter coefficients, more coefficients give a
            steeper transition between stop band and pass band.
        window: 'hamming', 'hann', 'blackman' or 'rectangular'
        fs: Sampling frequency, cutoff is relative to the Nyquist frequency.
        width: Width of the transition between pass band and stop band in the
            unit of fs, used to estimate numtaps if it is not given. The default
            is half the distance of the cutoff to the nearest band edge.

    Returns:
        Filtered samples with the length and type of the input.
    """
    return _fir_filter(a, cutoff, 'highpass', numtaps, window, fs, width)


@overload
def bandpass(a: vector[Any], cutoff: Sequence[float], numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> vector[float]: ...
@overload
def bandpass(a: tensor[Any], cutoff: Sequence[float], numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> tensor[float]: ...
@overload
def bandpass(a: 'array[Any] | Sequence[float | value[Any]]', cutoff: Sequence[float], numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> array[float]: ...
def bandpass(a: Any, cutoff: Sequence[float], numtaps: int | None = None, window: str = 'hamming', fs: float = 2.0, width: float | None = None) -> Any:
    """
    Band-pass filter (firwin FIR filter) with no delay, the input is extended
    with zeros beyond its edges, which affects the first and last
    (numtaps - 1) / 2 samples.

    Arguments:
        a: Samples as 1D vector, tensor, array or sequence
        cutoff: Lower and upper cutoff frequency (halved amplitude) in the unit of fs
        numtaps: Number of filter coefficients, more coefficients give
            steeper transitions between pass band and stop bands.
        window: 'hamming', 'hann', 'blackman' or 'rectangular'
        fs: Sampling frequency, cutoff is relative to the Nyquist frequency.
        width: Width of the transition between pass band and stop band in the
            unit of fs, used to estimate numtaps if it is not given. The default
            is half the distance of the cutoff to the nearest band edge.

    Returns:
        Filtered samples with the length and type of the input.
    """
    return _fir_filter(a, cutoff, 'bandpass', numtaps, window, fs, width)


def mean(input_vector: vector[Any]) -> unifloat:
    """
    Applies a mean filter to the input vector and returns the mean as a unifloat.

    Arguments:
        input_vector (vector): The input vector containing numerical values.

    Returns:
        unifloat: The mean value of the input vector.
    """
    return input_vector.sum() / len(input_vector)
