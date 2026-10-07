"""Tests for the neural network functions: activations, pooling, linear and softmax
against pure Python reference implementations."""
import math
from typing import Any

import pytest

import copapy as cp
from copapy import nn
from copapy.backend import get_dag_stats


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample(shape: tuple[int, ...], offset: int = 0, dtype: str = 'float') -> Any:
    """Nested lists of the given shape with deterministic values"""
    vals: list[Any] = [(i * 7 + offset) % 23 - 11 for i in range(math.prod(shape))]
    if dtype == 'float':
        vals = [v * 0.37 + 0.5 for v in vals]
    for d in reversed(shape[1:]):
        vals = [vals[i:i + d] for i in range(0, len(vals), d)]
    return vals


def flat(data: Any) -> list[Any]:
    if isinstance(data, cp.tensor | cp.vector):
        return list(data.values)
    return [x for d in data for x in flat(d)] if isinstance(data, list) else [data]


def pair(x: Any) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


def ref_pool2d(x: Any, op: str, kernel: Any, stride: Any = None, padding: Any = 0) -> Any:
    """Reference for a single input [c][h][w]"""
    (kh, kw), (ph, pw) = pair(kernel), pair(padding)
    sh, sw = pair(kernel if stride is None else stride)
    h, w = len(x[0]), len(x[0][0])
    oh, ow = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1

    def window(ch: Any, oy: int, ox: int) -> list[float]:
        return [ch[iy][ix] for iy in range(oy * sh - ph, oy * sh - ph + kh) for ix in range(ox * sw - pw, ox * sw - pw + kw)
                if 0 <= iy < h and 0 <= ix < w]

    def reduce(vals: list[float]) -> float:
        return max(vals) if op == 'max' else sum(vals) / (kh * kw)

    return [[[reduce(window(ch, oy, ox)) for ox in range(ow)] for oy in range(oh)] for ch in x]


def ref_softmax(x: list[float]) -> list[float]:
    e = [math.exp(v - max(x)) for v in x]
    return [v / sum(e) for v in e]


def test_relu_and_elementwise_min_max() -> None:
    df, di = sample((3, 7)), sample((20,), 2, 'int')
    other = sample((3, 7), 9)
    af, ai, ao = cp.array(df), cp.array(di), cp.array(other)
    packed = cp.tensor([[cp.value(v) for v in row] for row in df], packed=True)
    unpacked = cp.tensor([[cp.value(v) for v in row] for row in df], packed=False)

    results = [nn.relu(af), nn.relu(ai), nn.relu(packed), nn.relu(unpacked),
               cp.maximum(af, ao), cp.minimum(af, ao), cp.maximum(af, 1.5), cp.minimum(-2, ai),
               cp.maximum(packed, 0.25), cp.minimum(2.0, af)]
    refs = [[max(v, 0.0) for v in flat(df)], [max(v, 0) for v in di], [max(v, 0.0) for v in flat(df)], [max(v, 0.0) for v in flat(df)],
            [max(a, b) for a, b in zip(flat(df), flat(other))], [min(a, b) for a, b in zip(flat(df), flat(other))],
            [max(v, 1.5) for v in flat(df)], [min(v, -2) for v in di],
            [max(v, 0.25) for v in flat(df)], [min(v, 2.0) for v in flat(df)]]
    assert [type(r) for r in results[:4]] == [cp.array, cp.array, cp.tensor, cp.tensor]
    assert results[1].dtype == 'int'
    for out, ref in zip(evaluate(*results), refs):
        assert flat(out) == pytest.approx(ref, abs=1e-5)

    # One array stencil for arrays and packed tensors
    assert get_dag_stats([(nn.relu(packed).sum() + 1.0).net]).get('max_floatarr_int') == 1
    assert nn.relu(-2.5) == 0.0 and nn.relu(3) == 3


# c, h, w, kernel, stride, padding
POOL2D_CASES: list[Any] = [
    (1, 1, 1, 1, None, 0),
    (2, 6, 8, 2, None, 0),
    (3, 7, 9, 2, None, 0),  # incomplete windows at the edges are dropped
    (2, 8, 20, 3, 1, 1),  # 'same' size, rows wider than the SIMD width
    (2, 9, 11, (3, 2), (2, 1), (1, 0)),
    (1, 10, 13, (2, 4), (3, 2), (1, 2)),
    (2, 5, 5, 5, None, 2),
    (1, 4, 6, (1, 3), (1, 2), (0, 1)),
]


@pytest.mark.parametrize('c,h,w,kernel,stride,padding', POOL2D_CASES)
def test_pool2d(c: int, h: int, w: int, kernel: Any, stride: Any, padding: Any) -> None:
    dx = sample((c, h, w))
    x = cp.array(dx)
    res_max, res_avg = nn.max_pool2d(x, kernel, stride, padding), nn.avg_pool2d(x, kernel, stride, padding)
    ref_max, ref_avg = ref_pool2d(dx, 'max', kernel, stride, padding), ref_pool2d(dx, 'avg', kernel, stride, padding)
    assert res_max.shape == res_avg.shape == (c, len(ref_max[0]), len(ref_max[0][0]))
    out_max, out_avg = evaluate(res_max, res_avg)
    assert flat(out_max) == pytest.approx(flat(ref_max), abs=1e-5)
    assert flat(out_avg) == pytest.approx(flat(ref_avg), abs=1e-5)


def test_pool_batch_1d_and_types() -> None:
    dx = sample((2, 3, 6, 8))
    ref = [ref_pool2d(b, 'max', 2) for b in dx]
    batch = nn.max_pool2d(cp.array(dx), 2)
    as_tensor = nn.max_pool2d(cp.tensor([[[[cp.value(v) for v in r] for r in ch] for ch in b] for b in dx]), 2)
    assert batch.shape == (2, 3, 3, 4) and isinstance(as_tensor, cp.tensor) and as_tensor.shape == (2, 3, 3, 4)

    d1 = sample((3, 11), 4)
    max1, avg1 = nn.max_pool1d(cp.array(d1), 3, 2, 1), nn.avg_pool1d(cp.array([d1, d1]), 4)
    ref_max1 = [r[0] for r in ref_pool2d([[row] for row in d1], 'max', (1, 3), (1, 2), (0, 1))]
    ref_avg1 = [r[0] for r in ref_pool2d([[row] for row in d1], 'avg', (1, 4))]
    assert max1.shape == (3, 6) and avg1.shape == (2, 3, 2)

    glob = nn.global_avg_pool2d(cp.array(dx))
    ref_glob = [sum(flat(ch)) / 48 for b in dx for ch in b]
    assert glob.shape == (2, 3)

    out_batch, out_tensor, out_max1, out_avg1, out_glob = evaluate(batch, as_tensor, max1, avg1, glob)
    assert flat(out_batch) == flat(out_tensor) == pytest.approx(flat(ref), abs=1e-5)
    assert flat(out_max1) == pytest.approx(flat(ref_max1), abs=1e-5)
    assert flat(out_avg1) == pytest.approx(flat(ref_avg1) * 2, abs=1e-5)
    assert flat(out_glob) == pytest.approx(ref_glob, abs=1e-5)


def test_pool_errors() -> None:
    x = cp.array(sample((2, 6, 6)))
    for args in [(4, None, 3), (7,), (0,), (2, 0), ((1, 2, 3),)]:
        with pytest.raises(ValueError):
            nn.max_pool2d(x, *args)
    with pytest.raises(ValueError):
        nn.avg_pool2d(cp.array(sample((6, 6))), 2)  # missing channel dimension
    with pytest.raises(ValueError):
        nn.max_pool1d(cp.array(sample((6,))), 2)


def test_linear() -> None:
    dw, db = sample((4, 6), 3), sample((4,), 8)
    dx, dbatch = sample((6,), 1), sample((3, 6), 5)

    def ref(x: list[float], bias: bool = True) -> list[float]:
        return [sum(w * v for w, v in zip(row, x)) + (b if bias else 0.0) for row, b in zip(dw, db)]

    values = [cp.value(v) for v in dx]
    results = [nn.linear(cp.array(dx), cp.array(dw), cp.array(db)), nn.linear(cp.array(dx), dw),
               nn.linear(cp.array(dbatch), dw, db), nn.linear(cp.vector(values), dw, db),
               nn.linear(cp.tensor(values), cp.tensor(dw), cp.tensor(db))]
    assert [type(r) for r in results] == [cp.array, cp.array, cp.array, cp.vector, cp.tensor]
    assert [r.shape for r in results] == [(4,), (4,), (3, 4), (4,), (4,)]
    refs = [ref(dx), ref(dx, False), [v for row in dbatch for v in ref(row)], ref(dx), ref(dx)]
    for out, expected in zip(evaluate(*results), refs):
        assert flat(out) == pytest.approx(expected, abs=1e-4)

    with pytest.raises(ValueError):
        nn.linear(cp.array(sample((5,))), dw)  # feature mismatch
    with pytest.raises(ValueError):
        nn.linear(cp.array(dx), dw, [1.0, 2.0])  # bias size
    with pytest.raises(ValueError):
        nn.linear(cp.array(dx), dw[0])


def test_softmax() -> None:
    data = [1.5, -2.0, 0.25, 3.0, 3.0, -0.5]
    large = [1000.0, 999.0, -1000.0]  # exp(1000) overflows without the shift by the maximum
    results = [nn.softmax(cp.array(data)), nn.softmax(cp.vector(cp.value(v) for v in data)),
               nn.softmax(cp.tensor([cp.value(v) for v in data], packed=True)), nn.softmax(cp.array(large)),
               nn.softmax(cp.array([2, 0, -1]))]
    assert [type(r) for r in results] == [cp.array, cp.vector, cp.tensor, cp.array, cp.array]
    refs = [ref_softmax(data)] * 3 + [ref_softmax(large), ref_softmax([2.0, 0.0, -1.0])]
    for out, ref in zip(evaluate(*results), refs):
        assert flat(out) == pytest.approx(ref, abs=1e-6)
        assert sum(flat(out)) == pytest.approx(1.0, abs=1e-5)

    assert list(nn.softmax(cp.vector(data)).values) == pytest.approx(ref_softmax(data))  # constants
    with pytest.raises(ValueError):
        nn.softmax(cp.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(TypeError):
        nn.softmax(1.0)  # type: ignore[type-var]


def test_small_cnn() -> None:
    """Convolution, relu, max pooling, flatten, linear and softmax as one network"""
    dx = sample((1, 8, 8), 2)
    dw, db = [[[[v * 0.3 for v in r] for r in k] for k in f] for f in sample((3, 1, 3, 3), 6)], sample((3,), 1)
    dl, dlb = [[v * 0.05 for v in r] for r in sample((4, 27), 4)], sample((4,), 7)

    y = nn.conv2d(cp.array(dx), cp.array(dw), cp.array(db))            # 3 x 6 x 6
    y = nn.max_pool2d(nn.relu(y), 2)                                   # 3 x 3 x 3
    out = nn.softmax(nn.linear(y.reshape(-1), cp.array(dl), cp.array(dlb)))
    assert out.shape == (4,)

    conv = [[[sum(dw[c][0][ky][kx] * dx[0][oy + ky][ox + kx] for ky in range(3) for kx in range(3)) + db[c]
              for ox in range(6)] for oy in range(6)] for c in range(3)]
    pooled = flat(ref_pool2d([[[max(v, 0.0) for v in r] for r in ch] for ch in conv], 'max', 2))
    ref = ref_softmax([sum(w * v for w, v in zip(row, pooled)) + b for row, b in zip(dl, dlb)])

    result, = evaluate(out)
    assert result == pytest.approx(ref, abs=1e-4)
    # One stencil for each layer, independent of the size of the layers
    stats = get_dag_stats([(out.sum() + 1.0).net])
    assert all(stats.get(op) == 1 for op in ('conv2d_floatarr_floatarr', 'max_floatarr_int', 'maxpool2d_floatarr', 'matvec_floatarr_floatarr'))
