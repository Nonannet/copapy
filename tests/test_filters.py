"""Tests for sort, argsort and the filters: median by a sorting network stencil, convolve"""
import math
import random
import statistics
from typing import Any

import pytest

import copapy as cp
from copapy import filters
from copapy.backend import get_dag_stats


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample(n: int, dtype: str, seed: int) -> list[Any]:
    rnd = random.Random(seed)
    if dtype == 'int':
        return [rnd.randint(-5, 5) for _ in range(n)]  # many equal elements
    return [round(rnd.uniform(-10, 10), 1) if rnd.random() > 0.3 else 1.5 for _ in range(n)]


@pytest.mark.parametrize('n', [1, 2, 3, 5, 8, 9, 16, 17, 31, 64, 70])
@pytest.mark.parametrize('dtype', ['int', 'float'])
def test_sort_argsort_median(n: int, dtype: str) -> None:
    data = sample(n, dtype, n)
    v = cp.vector(cp.value(x) for x in data)

    s, a, m = cp.sort(v), cp.argsort(v), filters.median(v)
    out_s, out_a, out_m = evaluate(s, a, m)

    assert list(out_s.values) == pytest.approx(sorted(data))  # float32
    assert list(out_a.values) == sorted(range(n), key=lambda i: data[i])  # stable order of equal elements
    assert out_m == pytest.approx(statistics.median(data))
    assert isinstance(s, cp.vector) and isinstance(a, cp.vector)


@pytest.mark.parametrize('data', [[4.0, 1.0, 3.0, 2.0], [5, 1, 4, 2], [3], [2.5, -1.0, 7.0], [1, 1, 1, 1]])
def test_constant_inputs(data: list[Any]) -> None:
    """Constant vectors are evaluated at trace time"""
    v = cp.vector(data)
    assert list(cp.sort(v).values) == sorted(data)
    assert list(cp.argsort(v).values) == sorted(range(len(data)), key=lambda i: data[i])
    assert filters.median(v) == statistics.median(data)


def test_median_even_int() -> None:
    """The average of the two middle elements is a float, also for int elements"""
    v = cp.vector(cp.value(x) for x in [7, 1, 4, 2])
    m = filters.median(v)
    out, = evaluate(m)
    assert out == 3.0 and isinstance(out, float)


def test_inputs_without_packing() -> None:
    """Tensors and vectors with packing disabled are sorted by the stencil as well"""
    data = [3.0, -2.0, 8.5, 0.0, 1.0]
    t = cp.tensor([cp.value(x) for x in data])
    v = cp.vector((cp.value(x) for x in data), packed=False)
    st, sv = cp.sort(t), cp.sort(v)
    assert isinstance(st, cp.tensor) and isinstance(sv, cp.vector)
    out_t, out_v = evaluate(st, sv)
    assert list(out_t.values) == list(out_v.values) == sorted(data)

    with pytest.raises(ValueError):
        cp.sort(cp.tensor([[cp.value(1.0), cp.value(2.0)]]))  # 2D
    with pytest.raises(ValueError):
        cp.sort(cp.tensor([[2.0, 1.0], [4.0, 3.0]]))  # 2D constants
    with pytest.raises(ValueError):
        cp.argsort(cp.tensor([[2.0, 1.0], [4.0, 3.0]]))


def test_sort_quaternion() -> None:
    """Sorting works for every 1D ArrayType and keeps its class"""
    q = cp.quaternion(cp.value(0.5), cp.value(-1.0), cp.value(2.0), cp.value(0.0))
    sq, aq = cp.sort(q), cp.argsort(q)
    assert isinstance(sq, cp.quaternion)
    out_s, out_a = evaluate(sq, aq)
    assert list(out_s.values) == [-1.0, 0.0, 0.5, 2.0]
    assert list(out_a.values) == [1, 3, 0, 2]
    assert list(cp.sort(cp.quaternion(0.5, -1.0, 2.0, 0.0)).values) == [-1.0, 0.0, 0.5, 2.0]


def test_code_size() -> None:
    """One sort stencil instead of O(n^2) comparisons"""
    v = cp.vector(cp.value(float(i % 7)) for i in range(63))
    stats = get_dag_stats([filters.median(v).net])
    assert stats.get('sort_floatarr') == 1
    assert sum(stats.values()) < 200  # O(n^2) implementation: 12475 operations


def convolve_ref(a: list[float], v: list[float], mode: str) -> list[float]:
    """Definition of the discrete convolution with the output ranges of numpy.convolve"""
    n, m = len(a), len(v)
    full = [sum(a[i] * v[k - i] for i in range(n) if 0 <= k - i < m) for k in range(n + m - 1)]
    if mode == 'full':
        return full
    big, small = max(n, m), min(n, m)
    if mode == 'same':
        start = (small - 1) // 2
        return full[start:start + big]
    return full[small - 1:big]  # valid


@pytest.mark.parametrize('n,m', [(1, 1), (5, 3), (6, 4), (7, 1), (4, 7), (12, 5), (3, 3)])
@pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
def test_convolve(n: int, m: int, mode: str) -> None:
    a = sample(n, 'float', n)
    v = [x * 0.5 for x in sample(m, 'float', m + 100)]
    ref = convolve_ref(a, v, mode)

    va = cp.vector(cp.value(x) for x in a)
    vv = cp.vector(cp.value(x) for x in v)
    results = [filters.convolve(va, v, mode),        # constant kernel (FIR filter)
               filters.convolve(va, vv, mode),       # computed kernel
               filters.convolve(cp.array(a), v, mode),
               filters.convolve(cp.tensor([cp.value(x) for x in a]), cp.array(v), mode)]
    assert [type(r) for r in results] == [cp.vector, cp.vector, cp.array, cp.tensor]

    outs = evaluate(*results)
    for out in outs:
        vals = list(out.values) if isinstance(out, cp.vector | cp.tensor) else out
        assert vals == pytest.approx(ref, abs=1e-4)

    # Constant inputs are evaluated at trace time
    assert list(filters.convolve(cp.vector(a), v, mode).values) == pytest.approx(ref)


def test_convolve_numpy_reference() -> None:
    np = pytest.importorskip('numpy')
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    v = [0.25, 0.5, 0.25, 1.0]
    for mode in ('full', 'same', 'valid'):
        assert convolve_ref(a, v, mode) == pytest.approx(list(np.convolve(a, v, mode)))
        assert convolve_ref(v, a, mode) == pytest.approx(list(np.convolve(v, a, mode)))


def test_convolve_errors() -> None:
    a = cp.vector([cp.value(1.0), cp.value(2.0)])
    with pytest.raises(ValueError):
        filters.convolve(a, [1.0], 'circular')
    with pytest.raises(ValueError):
        filters.convolve(cp.tensor([[cp.value(1.0), cp.value(2.0)]]), [1.0])


def gaussian_ref(flat: list[float], shape: tuple[int, ...], sigmas: list[float], truncate: float = 4.0) -> list[float]:
    """Separable Gaussian filter of row-major data with zeros beyond the edges"""
    for axis, s in enumerate(sigmas):
        r = int(truncate * s + 0.5)
        if r == 0:
            continue
        k = [math.exp(-0.5 * (i / s) ** 2) for i in range(-r, r + 1)]
        k = [w / sum(k) for w in k]
        n, stride = shape[axis], math.prod(shape[axis + 1:])
        flat = [sum(k[j + r] * flat[idx + j * stride] for j in range(-r, r + 1) if 0 <= (idx // stride) % n + j < n)
                for idx in range(len(flat))]
    return flat


def nest(flat: list[Any], shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0]
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def flatten(data: Any) -> list[Any]:
    return [x for d in data for x in flatten(d)] if isinstance(data, list) else [data]


@pytest.mark.parametrize('shape,sigma', [((9,), 1.0), ((3,), 2.0), ((5, 7), 0.8), ((6, 4), (1.5, 0.0)),
                                         ((3, 4, 5), 0.7), ((4, 3, 6), (0.5, 1.0, 1.5))])
def test_gaussian_filter(shape: tuple[int, ...], sigma: Any) -> None:
    data = sample(math.prod(shape), 'float', sum(shape))
    sigmas = [float(sigma)] * len(shape) if isinstance(sigma, float) else list(sigma)
    ref = gaussian_ref(data, shape, sigmas)

    t = cp.tensor([cp.value(x) for x in data], shape)
    results = [filters.gaussian_filter(t, sigma), filters.gaussian_filter(cp.array(nest(data, shape)), sigma),
               filters.gaussian_filter(nest(data, shape), sigma), filters.gaussian_filter(cp.tensor(data, shape), sigma)]
    assert [type(r) for r in results] == [cp.tensor, cp.array, cp.array, cp.tensor]
    assert all(r.shape == shape for r in results)

    for out in evaluate(*results):
        vals = list(out.values) if isinstance(out, cp.tensor) else flatten(out)
        assert vals == pytest.approx(ref, abs=1e-4)


def test_gaussian_filter_vector() -> None:
    data = sample(12, 'float', 3)
    g = filters.gaussian_filter(cp.vector(cp.value(x) for x in data), 1.2, truncate=2.0)
    assert isinstance(g, cp.vector)
    out, = evaluate(g)
    assert list(out.values) == pytest.approx(gaussian_ref(data, (12,), [1.2], 2.0), abs=1e-4)


def test_gaussian_filter_scipy_reference() -> None:
    np = pytest.importorskip('numpy')
    ndimage = pytest.importorskip('scipy.ndimage')
    for shape, sigmas in [((9,), [1.0]), ((5, 7), [0.8, 1.6]), ((3, 4, 5), [0.7, 0.0, 1.1])]:
        data = sample(math.prod(shape), 'float', 1)
        ref = ndimage.gaussian_filter(np.array(data).reshape(shape), sigmas, mode='constant')
        assert gaussian_ref(data, shape, sigmas) == pytest.approx(list(ref.flatten()))


def test_gaussian_filter_errors() -> None:
    t = cp.tensor([[cp.value(1.0), cp.value(2.0)]])
    with pytest.raises(ValueError):
        filters.gaussian_filter(t, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        filters.gaussian_filter(t, -1.0)


@pytest.mark.parametrize('numtaps,cutoff,pass_zero,window,fs', [
    (31, 50.0, 'lowpass', 'hamming', 1000.0), (8, 0.3, 'lowpass', 'hann', 2.0), (1, 0.5, 'lowpass', 'hamming', 2.0),
    (21, 0.4, 'highpass', 'blackman', 2.0), (33, 120.0, 'highpass', 'hamming', 1000.0),
    (41, (0.2, 0.5), 'bandpass', 'hamming', 2.0), (30, (60.0, 180.0), 'bandpass', 'rectangular', 800.0)])
def test_firwin_scipy_reference(numtaps: int, cutoff: Any, pass_zero: str, window: str, fs: float) -> None:
    signal = pytest.importorskip('scipy.signal')
    scipy_window = 'boxcar' if window == 'rectangular' else window
    ref = signal.firwin(numtaps, cutoff, pass_zero=pass_zero, window=scipy_window, fs=fs)
    assert filters.firwin(numtaps, cutoff, pass_zero, window, fs) == pytest.approx(list(ref), abs=1e-12)


def test_firwin_gain() -> None:
    def gain(taps: list[float], f: float) -> float:
        """Amplitude response at the frequency f relative to the Nyquist frequency"""
        return abs(sum(t * complex(math.cos(math.pi * f * n), -math.sin(math.pi * f * n)) for n, t in enumerate(taps)))

    low, high, band = filters.firwin(41, 0.3), filters.firwin(41, 0.3, 'highpass'), filters.firwin(41, (0.3, 0.6), 'bandpass')
    assert low == pytest.approx(low[::-1])  # linear phase
    assert [gain(low, f) for f in (0.0, 0.1, 0.3)] == pytest.approx([1.0, 1.0, 0.5], abs=0.01)
    assert [gain(high, f) for f in (1.0, 0.5, 0.3)] == pytest.approx([1.0, 1.0, 0.5], abs=0.01)
    assert [gain(band, f) for f in (0.45, 0.3, 0.6)] == pytest.approx([1.0, 0.5, 0.5], abs=0.01)
    assert max(gain(low, 0.5), gain(high, 0.1), gain(band, 0.1), gain(band, 0.8)) < 0.01


def test_firwin_filtering() -> None:
    """A low-pass and a high-pass separate two sine waves"""
    n = 96
    slow = [math.sin(2 * math.pi * 0.02 * i) for i in range(n)]
    fast = [0.5 * math.sin(2 * math.pi * 0.35 * i) for i in range(n)]
    x = cp.vector(cp.value(a + b) for a, b in zip(slow, fast))

    out_low, out_high = evaluate(filters.convolve(x, filters.firwin(31, 0.3), 'same'),
                                 filters.convolve(x, filters.firwin(31, 0.3, 'highpass'), 'same'))
    inner = slice(15, n - 15)  # without the edges, where the input is extended with zeros
    assert list(out_low.values)[inner] == pytest.approx(slow[inner], abs=0.01)
    assert list(out_high.values)[inner] == pytest.approx(fast[inner], abs=0.01)


def test_firwin_errors() -> None:
    for args in [(0, 0.3), (8, 0.3, 'highpass'), (9, 0.3, 'bandstop'), (9, 0.3, 'lowpass', 'kaiser'), (9, 1.2),
                 (9, 0.0), (9, (0.5, 0.2), 'bandpass'), (9, 0.3, 'bandpass'), (9, (0.2, 0.5), 'lowpass')]:
        with pytest.raises(ValueError):
            filters.firwin(*args)  # type: ignore[arg-type]


def test_lowpass_highpass_bandpass() -> None:
    """Three sine waves are separated by the filters"""
    n, fs = 128, 1000.0
    waves = [[amp * math.sin(2 * math.pi * f / fs * i) for i in range(n)] for f, amp in [(15.0, 1.0), (150.0, 0.7), (400.0, 0.5)]]
    data = [sum(s) for s in zip(*waves)]
    x = cp.vector(cp.value(v) for v in data)

    results = [filters.lowpass(x, 70.0, 41, fs=fs), filters.bandpass(x, (80.0, 250.0), 41, fs=fs),
               filters.highpass(x, 300.0, 41, fs=fs)]
    assert all(isinstance(r, cp.vector) and len(r) == n for r in results)
    inner = slice(20, n - 20)  # without the edges, where the input is extended with zeros
    for out, wave in zip(evaluate(*results), waves):
        assert list(out.values)[inner] == pytest.approx(wave[inner], abs=0.02)

    # Other input types, equal to convolve with the coefficients of firwin
    ref = convolve_ref(data, filters.firwin(31, 0.2), 'same')
    low_t, low_a = filters.lowpass(cp.tensor([cp.value(v) for v in data]), 0.2, 31), filters.lowpass(cp.array(data), 0.2, 31)
    assert isinstance(low_t, cp.tensor) and isinstance(low_a, cp.array)
    out_t, out_a = evaluate(low_t, low_a)
    assert list(out_t.values) == pytest.approx(ref, abs=1e-4) and out_a == pytest.approx(ref, abs=1e-4)
    assert list(filters.lowpass(cp.vector(data), 0.2, 31).values) == pytest.approx(ref)  # constants

    with pytest.raises(ValueError):
        filters.lowpass(cp.vector(cp.value(v) for v in data[:10]), 0.2, 31)  # more taps than samples
    with pytest.raises(ValueError):
        filters.highpass(x, 0.2, 30)  # even number of taps


def test_estimated_numtaps() -> None:
    """Without numtaps the number of coefficients is estimated from the transition width"""
    estimate = filters._estimate_numtaps  # pyright: ignore[reportPrivateUsage]
    assert estimate(50.0, 'hamming', 1000.0, None) == 133  # width 25: 3.3 * 1000 / 25
    assert estimate(450.0, 'hamming', 1000.0, None) == 133  # nearest edge is the Nyquist frequency
    assert estimate((80.0, 250.0), 'hamming', 1000.0, None) == 83  # width 40
    assert estimate((200.0, 230.0), 'hann', 1000.0, None) == 207  # width 15, limited by the band
    assert estimate(0.2, 'hamming', 2.0, 0.2) == 33
    assert estimate(0.2, 'blackman', 2.0, 0.2) == 55

    data = sample(128, 'float', 7)
    x = cp.vector(cp.value(v) for v in data)
    out_default, out_width = evaluate(filters.lowpass(x, 0.2), filters.highpass(x, 0.5, width=0.2))
    assert list(out_default.values) == pytest.approx(convolve_ref(data, filters.firwin(67, 0.2), 'same'), abs=1e-4)
    assert list(out_width.values) == pytest.approx(convolve_ref(data, filters.firwin(33, 0.5, 'highpass'), 'same'), abs=1e-4)

    with pytest.raises(ValueError, match='numtaps .133.'):
        filters.lowpass(x, 50.0, fs=1000.0)  # more taps estimated than samples
    for kwargs in [{'width': 0.0}, {'window': 'kaiser'}, {'fs': 0.3}]:
        with pytest.raises(ValueError):
            filters.lowpass(x, 0.2, **kwargs)  # type: ignore[arg-type]
