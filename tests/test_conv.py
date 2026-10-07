"""Tests for the convolution array stencil (conv1d and conv2d) against a
pure Python reference implementation."""
from typing import Any

import pytest

import copapy as cp


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample(shape: tuple[int, ...], offset: int = 0, dtype: str = 'float') -> Any:
    """Nested lists of the given shape with deterministic values"""
    size = 1
    for d in shape:
        size *= d
    vals: list[Any] = [(i * 7 + offset) % 23 - 11 for i in range(size)]
    if dtype == 'float':
        vals = [v * 0.37 + 0.5 for v in vals]
    for d in reversed(shape[1:]):
        vals = [vals[i:i + d] for i in range(0, len(vals), d)]
    return vals


def flat(data: Any) -> list[Any]:
    return [x for d in data for x in flat(d)] if isinstance(data, list) else [data]


def pair(x: Any) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


def ref_conv2d(x: Any, w: Any, b: Any = None, stride: Any = 1, padding: Any = 0, dilation: Any = 1) -> Any:
    """Reference for a single input [ci][h][w] and weights [co][ci][kh][kw]"""
    (sh, sw), (ph, pw), (dh, dw) = pair(stride), pair(padding), pair(dilation)
    h, wd, kh, kw = len(x[0]), len(x[0][0]), len(w[0][0]), len(w[0][0][0])
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    def at(q: int, iy: int, ix: int) -> float:
        return x[q][iy][ix] if 0 <= iy < h and 0 <= ix < wd else 0.0

    return [[[sum(w[c][q][ky][kx] * at(q, oy * sh - ph + ky * dh, ox * sw - pw + kx * dw)
                  for q in range(len(x)) for ky in range(kh) for kx in range(kw)) + (b[c] if b else 0.0)
              for ox in range(ow)] for oy in range(oh)] for c in range(len(w))]


def ref_conv1d(x: Any, w: Any, b: Any = None, stride: int = 1, padding: int = 0, dilation: int = 1) -> Any:
    ret = ref_conv2d([[row] for row in x], [[[k] for k in wc] for wc in w], b, (1, stride), (0, padding), (1, dilation))
    return [rows[0] for rows in ret]


# ci, h, w, co, kh, kw, stride, padding, dilation, bias
CONV2D_CASES: list[Any] = [
    (1, 1, 1, 1, 1, 1, 1, 0, 1, False),
    (1, 5, 5, 1, 3, 3, 1, 0, 1, False),
    (3, 8, 20, 4, 3, 3, 1, 1, 1, True),  # 'same' padding, rows wider than the SIMD width
    (2, 9, 11, 3, 3, 5, 2, (1, 2), 1, True),
    (2, 10, 13, 2, 2, 4, (3, 2), (0, 3), 1, True),  # even kernels, different strides
    (2, 12, 17, 2, 3, 3, 1, 2, (2, 3), True),
    (3, 7, 9, 5, 1, 1, 1, 0, 1, True),  # 1x1 convolution
    (1, 2, 3, 2, 5, 5, 1, 2, 1, True),  # kernel larger than the unpadded input
    (2, 6, 6, 1, 3, 3, 2, 3, 2, False),  # output rows and columns only from padding
]


@pytest.mark.parametrize('ci,h,w,co,kh,kw,stride,padding,dilation,bias', CONV2D_CASES)
def test_conv2d(ci: int, h: int, w: int, co: int, kh: int, kw: int,
                stride: Any, padding: Any, dilation: Any, bias: bool) -> None:
    dx, dw = sample((ci, h, w)), sample((co, ci, kh, kw), 5)
    db = sample((co,), 3) if bias else None
    ref = ref_conv2d(dx, dw, db, stride, padding, dilation)

    res = cp.nn.conv2d(cp.array(dx), cp.array(dw), cp.array(db) if db else None, stride, padding, dilation)
    out, = evaluate(res)

    assert res.shape == (co, len(ref[0]), len(ref[0][0]))
    assert flat(out) == pytest.approx(flat(ref), rel=1e-4, abs=1e-4)


# ci, length, co, k, stride, padding, dilation, bias
CONV1D_CASES: list[Any] = [
    (1, 1, 1, 1, 1, 0, 1, False),
    (1, 10, 1, 3, 1, 0, 1, True),
    (3, 37, 4, 5, 1, 2, 1, True),
    (2, 30, 3, 4, 3, 1, 1, True),
    (2, 25, 2, 3, 2, 4, 4, False),
]


@pytest.mark.parametrize('ci,length,co,k,stride,padding,dilation,bias', CONV1D_CASES)
def test_conv1d(ci: int, length: int, co: int, k: int, stride: int, padding: int, dilation: int, bias: bool) -> None:
    dx, dw = sample((ci, length)), sample((co, ci, k), 5)
    db = sample((co,), 3) if bias else None
    ref = ref_conv1d(dx, dw, db, stride, padding, dilation)

    res = cp.nn.conv1d(cp.array(dx), cp.array(dw), cp.array(db) if db else None, stride, padding, dilation)
    out, = evaluate(res)

    assert res.shape == (co, len(ref[0]))
    assert flat(out) == pytest.approx(flat(ref), rel=1e-4, abs=1e-4)


def test_batch() -> None:
    dx, dw, db = sample((3, 2, 6, 7)), sample((4, 2, 3, 3), 5), sample((4,), 3)
    d1, w1 = sample((3, 2, 12)), sample((4, 2, 3), 5)

    res2 = cp.nn.conv2d(cp.array(dx), cp.array(dw), cp.array(db), stride=2, padding=1)
    res1 = cp.nn.conv1d(cp.array(d1), cp.array(w1), cp.array(db), padding=1)
    out2, out1 = evaluate(res2, res1)

    assert res2.shape == (3, 4, 3, 4)
    assert res1.shape == (3, 4, 12)
    assert flat(out2) == pytest.approx(flat([ref_conv2d(x, dw, db, 2, 1) for x in dx]), rel=1e-4, abs=1e-4)
    assert flat(out1) == pytest.approx(flat([ref_conv1d(x, w1, db, 1, 1) for x in d1]), rel=1e-4, abs=1e-4)


def test_tensor_int_and_nested_inputs() -> None:
    """Tensors return tensors, int data is converted and weights can be nested lists"""
    dx, dw = sample((2, 5, 6), dtype='int'), sample((3, 2, 2, 2), 5, 'int')
    ref = ref_conv2d(dx, dw)

    res_t = cp.nn.conv2d(cp.tensor(dx), cp.tensor(dw))
    res_a = cp.nn.conv2d(cp.array(dx), dw, [0, 0, 0])
    d1 = dx[0][:2]  # 2 channels of length 6
    res_c = cp.nn.conv1d(cp.tensor([[cp.value(float(v)) for v in row] for row in d1]), dw[0]).sum()
    out_t, out_a, out_c = evaluate(res_t, res_a, res_c)

    assert isinstance(res_t, cp.tensor) and isinstance(res_a, cp.array)
    assert res_t.shape == res_a.shape == (3, 4, 5)
    assert list(out_t.values) == pytest.approx(flat(ref))
    assert flat(out_a) == pytest.approx(flat(ref))
    assert out_c == pytest.approx(sum(flat(ref_conv1d(d1, dw[0]))))


def test_chained_layers_and_rerun() -> None:
    """Two layers with operations in between, weights and inputs rewritten on the target"""
    dx, dw1, db1, dw2 = sample((1, 8, 8)), sample((2, 1, 3, 3), 5), sample((2,), 3), sample((1, 2, 3, 3), 7)
    x, w1, b1 = cp.array(dx), cp.array(dw1), cp.array(db1)
    y = cp.nn.conv2d(cp.nn.conv2d(x, w1, b1, padding=1) * 0.5, cp.array(dw2), stride=2)

    def ref(vx: Any, vw1: Any) -> Any:
        hidden = [[[v * 0.5 for v in row] for row in ch] for ch in ref_conv2d(vx, vw1, db1, padding=1)]
        return ref_conv2d(hidden, dw2, stride=2)

    tg = cp.Target()
    tg.compile(y)
    tg.run()
    assert flat(tg.read_value(y)) == pytest.approx(flat(ref(dx, dw1)), rel=1e-4, abs=1e-4)

    dx2, dw1b = sample((1, 8, 8), 9), sample((2, 1, 3, 3), 11)
    tg.write_value(x, dx2)
    tg.write_value(w1, dw1b)
    tg.run()
    assert flat(tg.read_value(y)) == pytest.approx(flat(ref(dx2, dw1b)), rel=1e-4, abs=1e-4)


def test_code_size_independent_of_shape() -> None:
    from copapy.backend import compile_to_dag

    def program_size(n: int) -> int:
        y = cp.nn.conv2d(cp.array(sample((2, n, n))), cp.array(sample((3, 2, 3, 3))))
        dw, _ = compile_to_dag([y.net.source], cp.generic_sdb)
        return len(dw.get_data()) - 4 * 2 * n * n  # without the data of the input

    assert program_size(40) == program_size(8)


def test_errors() -> None:
    x = cp.array(sample((2, 5, 5)))
    w = cp.array(sample((3, 2, 3, 3)))
    with pytest.raises(ValueError):
        cp.nn.conv2d(x, cp.array(sample((3, 1, 3, 3))))  # channel mismatch
    with pytest.raises(ValueError):
        cp.nn.conv2d(x, w, cp.array([1.0, 2.0]))  # bias size
    with pytest.raises(ValueError):
        cp.nn.conv2d(x, cp.array(sample((3, 2, 6, 6))))  # kernel larger than input
    with pytest.raises(ValueError):
        cp.nn.conv2d(x, w, stride=0)
    with pytest.raises(ValueError):
        cp.nn.conv2d(x, w, padding=(1, 2, 3))
    with pytest.raises(ValueError):
        cp.nn.conv2d(cp.array(sample((5, 5))), w)  # missing channel dimension
    with pytest.raises(ValueError):
        cp.nn.conv1d(x, w)
    with pytest.raises(TypeError):
        cp.nn.conv2d(x, 1.0)
