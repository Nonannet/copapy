"""Tests for the array class: array stencils for element-wise operations,
reductions, matrix-vector products and the interaction with scalar values."""
import math
import operator
from typing import Any, Callable

import pytest

import copapy as cp


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample_values(n: int, dtype: str, offset: int = 0) -> list[Any]:
    if dtype == 'int':
        return [(i * 7 + offset) % 23 - 11 for i in range(n)]
    return [((i * 7 + offset) % 23 - 11) * 0.37 + 0.5 for i in range(n)]


OPS: dict[str, Callable[[Any, Any], Any]] = {
    'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': operator.truediv}

# Lengths below, equal and above the SIMD width and the unrolling of the reductions
SIZES = [1, 3, 8, 17, 100]


@pytest.mark.parametrize('n', SIZES)
@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('t1,t2', [('int', 'int'), ('int', 'float'), ('float', 'int'), ('float', 'float')])
def test_elementwise(n: int, op: str, t1: str, t2: str) -> None:
    va = sample_values(n, t1)
    vb = [v if v != 0 else 3 for v in sample_values(n, t2, 5)]  # no division by zero
    a, b = cp.array(va, t1), cp.array(vb, t2)
    s = cp.value(vb[0])
    f = OPS[op]

    results = evaluate(f(a, b), f(a, s), f(s, a if op != 'div' else b), f(a, 3), f(2.5, b))
    expected = [
        [f(x, y) for x, y in zip(va, vb)],
        [f(x, vb[0]) for x in va],
        [f(vb[0], y) for y in (va if op != 'div' else vb)],
        [f(x, 3) for x in va],
        [f(2.5, y) for y in vb]]

    for res, exp in zip(results, expected):
        assert res == pytest.approx(exp, rel=1e-5)


@pytest.mark.parametrize('n', SIZES)
@pytest.mark.parametrize('dtype', ['int', 'float'])
def test_reductions(n: int, dtype: str) -> None:
    va = sample_values(n, dtype)
    vb = sample_values(n, dtype, 3)
    a, b = cp.array(va), cp.array(vb)

    res_sum, res_dot, res_matmul = evaluate(a.sum(), a.dot(b), a @ b)

    assert res_sum == pytest.approx(sum(va), rel=1e-5)
    assert res_dot == pytest.approx(sum(x * y for x, y in zip(va, vb)), rel=1e-5)
    assert res_matmul == pytest.approx(res_dot)
    if dtype == 'int':
        assert isinstance(res_sum, int) and isinstance(res_dot, int)


@pytest.mark.parametrize('m,n', [(1, 1), (3, 5), (8, 17), (20, 64)])
def test_matvec(m: int, n: int) -> None:
    rows = [sample_values(n, 'float', r) for r in range(m)]
    vx = sample_values(n, 'float', 11)
    mat, x = cp.array(rows), cp.array(vx)

    res, = evaluate(mat @ x)

    assert mat.shape == (m, n)
    assert res == pytest.approx([sum(a * b for a, b in zip(r, vx)) for r in rows], rel=1e-5)


ARG_VALS = [-3.0, -1.0, -0.5, -0.01, 0.0, 0.3, 0.75, 1.0, 2.5, 9.0, 40.0]
UNIT_VALS = [-1.0, -0.7, -0.01, 0.0, 0.2, 0.5, 0.95, 1.0, 0.1, -0.3, 0.6]
POS_VALS = [0.001, 0.5, 1.0, 2.5, 7.0, 100.0, 12345.0, 0.25, 3.0, 9.0, 16.0]

UNARY_FUNCS: dict[str, tuple[Callable[[Any], Any], list[float]]] = {
    'sqrt': (math.sqrt, POS_VALS), 'exp': (math.exp, ARG_VALS), 'log': (math.log, POS_VALS),
    'sin': (math.sin, ARG_VALS), 'cos': (math.cos, ARG_VALS), 'tan': (math.tan, ARG_VALS),
    'asin': (math.asin, UNIT_VALS), 'acos': (math.acos, UNIT_VALS), 'atan': (math.atan, ARG_VALS),
    'tanh': (math.tanh, ARG_VALS + [1e-4, -0.2, 0.25]), 'abs': (abs, ARG_VALS)}


@pytest.mark.parametrize('name', UNARY_FUNCS)
def test_unary_functions(name: str) -> None:
    ref_func, vals = UNARY_FUNCS[name]
    ints = [1, 2, 5, 30] if name in ('sqrt', 'log') else [-1, 0, 1] if name in ('asin', 'acos') else [-7, -1, 0, 1, 2, 5]
    func = getattr(cp, name)

    res_f, res_i = func(cp.array(vals)), func(cp.array(ints))
    out_f, out_i = evaluate(res_f, res_i)

    assert isinstance(res_f, cp.array) and res_f.shape == (len(vals),)
    assert res_f.dtype == 'float' and res_i.dtype == ('int' if name == 'abs' else 'float')
    assert out_f == pytest.approx([ref_func(v) for v in vals], rel=1e-5, abs=1e-6)
    assert out_i == pytest.approx([ref_func(v) for v in ints], rel=1e-5, abs=1e-6)


@pytest.mark.parametrize('t1,t2', [('int', 'int'), ('int', 'float'), ('float', 'int'), ('float', 'float')])
def test_binary_functions(t1: str, t2: str) -> None:
    va = [abs(v) + 1 for v in sample_values(11, t1)]  # positive bases
    vb = [v % 4 - 1 if t2 == 'int' else v * 0.2 for v in sample_values(11, t2, 5)]
    a, b = cp.array(va, t1), cp.array(vb, t2)
    sa, sb = cp.value(va[3]), cp.value(vb[4])

    res = [a ** b, a ** sb, sa ** b, cp.pow(a, 0.5), cp.pow(2, b), 2.5 ** b,
           cp.atan2(a, b), cp.atan2(a, sb), cp.atan2(sa, b), cp.atan2(a, -1.5), cp.atan2(-2, b)]
    ref = [[x ** y for x, y in zip(va, vb)], [x ** vb[4] for x in va], [va[3] ** y for y in vb],
           [x ** 0.5 for x in va], [2 ** y for y in vb], [2.5 ** y for y in vb],
           [math.atan2(x, y) for x, y in zip(va, vb)], [math.atan2(x, vb[4]) for x in va],
           [math.atan2(va[3], y) for y in vb], [math.atan2(x, -1.5) for x in va], [math.atan2(-2, y) for y in vb]]

    for r, o, e in zip(res, evaluate(*res), ref):
        assert r.dtype == 'float'
        assert o == pytest.approx(e, rel=1e-5, abs=1e-6)


def test_small_integer_powers() -> None:
    """Powers 1 to 7 are multiplications and keep the element type"""
    va = [-3, -1, 0, 2, 5]
    a = cp.array(va)
    m = cp.array([[1.5, -2.0], [0.5, 3.0]])

    res = [a ** 1, a ** 2, a ** 3, cp.pow(a, 7), m ** 2, a ** 0, cp.exp(m).sum(), cp.nn.sigmoid(m)]
    out = evaluate(*res)

    assert a ** 1 is a
    assert [r.dtype for r in res[:4]] == ['int'] * 4
    assert out[:4] == [va, [v ** 2 for v in va], [v ** 3 for v in va], [v ** 7 for v in va]]
    assert out[4] == [[2.25, 4.0], [0.25, 9.0]]
    assert out[5] == [1.0] * 5
    assert out[6] == pytest.approx(sum(math.exp(v) for v in (1.5, -2.0, 0.5, 3.0)), rel=1e-5)
    assert flat(out[7]) == pytest.approx([1 / (1 + math.exp(-v)) for v in (1.5, -2.0, 0.5, 3.0)], rel=1e-5)


def test_chained_ops_and_elements() -> None:
    va = [1.0, 2.0, 3.0, 4.0, 5.0]
    vb = [10, 20, 30, 40, 50]
    a, b = cp.array(va), cp.array(vb)
    s = cp.value(2.5)

    d = (a + b) * s - 1
    m = cp.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
    h = m @ a
    x = d[2] * 2 + h[1]  # scalar ops on array elements
    k = -(a * x)  # array op with a computed scalar
    e = cp.sqrt(d.sum()) + a[-1]  # scalar op on reduction result

    ref_d = [(p + q) * 2.5 - 1 for p, q in zip(va, vb)]
    ref_x = ref_d[2] * 2 + 6.0

    res_d, res_x, res_k, res_e, res_h = evaluate(d, x, k, e, h)

    assert res_d == pytest.approx(ref_d)
    assert res_x == pytest.approx(ref_x)
    assert res_k == pytest.approx([-p * ref_x for p in va])
    assert res_e == pytest.approx(sum(ref_d) ** 0.5 + 5.0)
    assert res_h == pytest.approx([55.0, 6.0])


def test_write_and_rerun() -> None:
    a = cp.array([1.0, 2.0, 3.0])
    w = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    s = cp.value(2.0)
    y = w @ (a * s)

    tg = cp.Target()
    tg.compile(y)
    tg.run()
    assert tg.read_value(y) == pytest.approx([2.0, 12.0])

    tg.write_value(a, [1, 1, 1])
    tg.write_value(w, [[0, 0, 1], [2, 0, 0]])
    tg.write_value(s, 3.0)
    tg.run()
    assert tg.read_value(y) == pytest.approx([3.0, 6.0])
    assert tg.read_value(w) == [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]]


def test_shared_subexpression() -> None:
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])

    c1 = a + b
    c2 = a + b  # identical operation is computed once

    r1, r2, r3 = evaluate(c1, c2, c1 * c2)

    assert r1 == r2 == [5, 7, 9]
    assert r3 == [25, 49, 81]


def test_code_size_independent_of_length() -> None:
    from copapy.backend import compile_to_dag

    def program_size(n: int) -> int:
        a = cp.array([1.0] * n)
        b = cp.array([2.0] * n)
        c = ((a + b) * 3.0 - a) / b
        dw, _ = compile_to_dag([c.net.source], cp.generic_sdb)
        return len(dw.get_data()) - 2 * 4 * n  # without the data of a, b

    assert program_size(1000) == program_size(10)


def test_errors() -> None:
    a = cp.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        a + cp.array([1.0, 2.0])
    with pytest.raises(IndexError):
        a[3]
    with pytest.raises(ValueError):
        cp.array([[1, 2], [3]])
    with pytest.raises(ValueError):
        cp.array([1, 2.5], 'int')  # float element in an int array
    with pytest.raises(ValueError):
        cp.array(['a', 'b'])
    with pytest.raises(ValueError):
        cp.array([[1.0, 2.0], [3.0, 4.0]]) @ cp.array([[1.0, 2.0, 3.0]])
    with pytest.raises(IndexError):
        a[1, 2]


def flat(data: Any) -> list[Any]:
    return [x for d in data for x in flat(d)] if isinstance(data, list) else [data]


def nested(n0: int, n1: int, n2: int, dtype: str = 'float') -> list[Any]:
    vals = sample_values(n0 * n1 * n2, dtype)
    return [[vals[(i * n1 + j) * n2:(i * n1 + j + 1) * n2] for j in range(n1)] for i in range(n0)]


VIEWS: dict[str, Callable[[Any], Any]] = {
    'row': lambda x: x[1],
    'column': lambda x: x[:, 2],
    'slice_step': lambda x: x[1:, ::2, 3],
    'reverse': lambda x: x[::-1, 1:3],
    'neg_index': lambda x: x[-1, -2],
    'transpose': lambda x: x.T,
    'transpose_axes': lambda x: x.transpose(1, 0, 2),
    'transpose_slice': lambda x: x.transpose(2, 0, 1)[1:4, :, ::3],
    'reshape': lambda x: x.reshape(4, -1),
    'reshape_transpose': lambda x: x.reshape(6, 10).T,
}


@pytest.mark.parametrize('view', VIEWS)
@pytest.mark.parametrize('dtype', ['int', 'float'])
def test_views(view: str, dtype: str) -> None:
    data = nested(3, 4, 5, dtype)
    res = VIEWS[view](cp.array(data))
    ref = VIEWS[view](cp.tensor(data, packed=False))  # computed at trace time

    out, = evaluate(res)

    assert res.shape == ref.shape
    assert flat(out) == pytest.approx(list(ref.values))


@pytest.mark.parametrize('shape1,shape2', [((3, 1), (4,)), ((4,), (3, 4)), ((2, 1, 5), (3, 1)),
                                           ((1,), (6,)), ((3, 4), (3, 4))])
@pytest.mark.parametrize('op', OPS)
def test_broadcasting(shape1: tuple[int, ...], shape2: tuple[int, ...], op: str) -> None:
    def make(shape: tuple[int, ...], offset: int) -> list[Any]:
        size = 1
        for d in shape:
            size *= d
        vals: list[Any] = [v if v != 0 else 1.5 for v in sample_values(size, 'float', offset)]
        for d in reversed(shape[1:]):
            vals = [vals[i:i + d] for i in range(0, len(vals), d)]
        return vals

    d1, d2 = make(shape1, 0), make(shape2, 5)
    res = OPS[op](cp.array(d1), cp.array(d2))
    ref = OPS[op](cp.tensor(d1, packed=False), cp.tensor(d2, packed=False))

    out, = evaluate(res)

    assert res.shape == ref.shape
    assert flat(out) == pytest.approx(list(ref.values))


@pytest.mark.parametrize('m,k,n', [(1, 1, 1), (2, 3, 4), (5, 8, 3), (16, 17, 9)])
@pytest.mark.parametrize('t1,t2', [('int', 'int'), ('int', 'float'), ('float', 'int'), ('float', 'float')])
def test_matmul(m: int, k: int, n: int, t1: str, t2: str) -> None:
    da = nested(1, m, k, t1)[0]
    db = nested(1, k, n, t2)[0]
    va = da[0]  # length k

    res = [cp.array(da) @ cp.array(db), cp.array(va) @ cp.array(db)]
    ref = [cp.tensor(da, packed=False) @ cp.tensor(db, packed=False),
           cp.tensor(va, packed=False) @ cp.tensor(db, packed=False)]

    out = evaluate(*res)

    for o, r, x in zip(out, ref, res):
        assert x.shape == r.shape
        assert x.dtype == ('int' if t1 == t2 == 'int' else 'float')
        assert flat(o) == pytest.approx(list(r.values), rel=1e-5)


def test_pack_values() -> None:
    import math
    x = cp.value(2.0)
    y = cp.value(3.0)
    p = cp.array([x, x * y, 1.5, cp.sin(y) + x, y])  # computed values and constants
    q = cp.array([[cp.value(1), 2], [3, cp.value(4)]])
    r = (p * 2.0).sum() + p[1]

    tg = cp.Target()
    qq = q @ q
    tg.compile(p, r, qq)
    tg.run()
    assert tg.read_value(p) == pytest.approx([2.0, 6.0, 1.5, math.sin(3) + 2, 3.0])
    assert tg.read_value(r) == pytest.approx(2 * (2 + 6 + 1.5 + math.sin(3) + 2 + 3) + 6)
    assert tg.read_value(qq) == [[7, 10], [15, 22]]

    tg.write_value(x, 1.0)
    tg.run()
    assert tg.read_value(p) == pytest.approx([1.0, 3.0, 1.5, math.sin(3) + 1, 3.0])


def hybrid_program(t1: Any, t2: Any, w: Any, s: Any) -> list[Any]:
    """Tensor operations with array implementation and scalar fallbacks"""
    return [t1 + t2, t1 * s - 1.5, 2.0 / (t2 + 10.0), -t1, w @ t1[:, 0], w @ t1, t2.T @ w.T, t1.sum(),
            t1.T[1:3], t1.reshape(-1)[::3], (t1 + t2).mean(),
            t1 > 0.0, t1 ** 2, t1.map(lambda v: v * v + 1), t1.sum(axis=0),
            cp.exp(t1 * 0.1), cp.tanh(t2), cp.sqrt(t1 * t1 + 1.0), cp.sin(t1), cp.abs(t2), cp.atan2(t1, t2),
            cp.atan2(t1, s), (t1 * t1 + 1.0) ** 0.3, 1.5 ** (t2 * 0.1), cp.pow(t1 ** 2 + 0.5, t2 * 0.1), t1 ** 3]


def test_hybrid_tensor() -> None:
    d1 = nested(1, 8, 12)[0]
    d2 = nested(1, 8, 12, 'float')[0][::-1]
    dw = [row[:8] for row in nested(1, 5, 12)[0]]

    def build(packed: bool | None) -> list[Any]:
        t1 = cp.tensor([[cp.value(v) for v in row] for row in d1], packed=packed)
        t2 = cp.tensor([[cp.value(v) for v in row] for row in d2], packed=packed)
        w = cp.tensor(dw, packed=packed)
        return hybrid_program(t1, t2, w, cp.value(2.5))  # w: 5 x 8

    res_packed = build(None)  # 96 elements: above the threshold
    res_scalar = build(False)

    out_packed = evaluate(*res_packed)
    out_scalar = evaluate(*res_scalar)

    assert res_packed[0]._packed_array() is not None  # computed by array stencils
    assert res_scalar[0]._packed_array() is None
    assert all(r._packed_array() is not None for r in res_packed[-11:])  # functions by array stencils
    for p, s, rp, rs in zip(out_packed, out_scalar, res_packed, res_scalar):
        assert getattr(rp, 'shape', ()) == getattr(rs, 'shape', ())
        p_vals = list(p.values) if isinstance(p, cp.tensor) else [p]
        s_vals = list(s.values) if isinstance(s, cp.tensor) else [s]
        assert p_vals == pytest.approx(s_vals, rel=1e-5)


def test_hybrid_threshold_and_constants() -> None:
    small = cp.tensor([cp.value(float(i)) for i in range(10)])
    large = cp.tensor([cp.value(float(i)) for i in range(100)])
    assert (small * 2.0)._packed_array() is None
    assert (large * 2.0)._packed_array() is not None

    # Only copapy values and constants other than zero count for the threshold (64)
    def mixed(n_values: int, n_nonzero: int) -> Any:
        return cp.tensor([cp.value(1.0)] * n_values + [2.5] * n_nonzero + [0.0] * (200 - n_values - n_nonzero))
    assert mixed(64, 0)._get_array() is None
    assert mixed(65, 0)._get_array() is not None
    assert mixed(30, 34)._get_array() is None
    assert mixed(30, 35)._get_array() is not None
    assert cp.eye(100)._get_array() is not None  # 100 ones

    # Constants are still evaluated at trace time and sparse constants stay unpacked,
    # also if the other operand is packed
    const = cp.tensor([float(i) for i in range(100)])
    assert (const * 2.0).values[3] == 6.0
    sparse = cp.tensor([[1.0 if i == j and i < 50 else 0.0 for j in range(100)] for i in range(100)])
    assert sparse._get_array() is None
    res = sparse @ large
    assert res.values[5] is large.values[5]  # multiplications by 0 and 1 eliminated

    threshold = cp.tensor.pack_threshold
    try:
        cp.tensor.pack_threshold = None
        assert (cp.tensor([cp.value(float(i)) for i in range(100)]) * 2.0)._packed_array() is None
    finally:
        cp.tensor.pack_threshold = threshold


def test_hybrid_code_size() -> None:
    from copapy.backend import compile_to_dag

    def program_size(n: int) -> int:
        w = cp.tensor([[((i * 7 + j) % 13 - 6) * 0.01 for j in range(n)] for i in range(n)])
        x = cp.tensor([cp.value(float(i)) for i in range(n)])
        y = cp.tensor([cp.value(float(i)) for i in range(n)], packed=False)
        h = (w @ x + 0.5) * 2.0
        res = (w @ h - x).sum()
        dw, _ = compile_to_dag([cp.backend.Store(res)], cp.generic_sdb)
        dw_scalar, _ = compile_to_dag([cp.backend.Store(((w @ y + 0.5) * 2.0).sum())], cp.generic_sdb)
        return len(dw.get_data()) - 4 * n * n, len(dw_scalar.get_data())  # without the data of w

    size_64, scalar_64 = program_size(64)
    size_128, _ = program_size(128)
    assert size_64 < scalar_64 / 10
    assert size_128 < size_64 * 2.5  # grows only with the packing of x (linear)


def test_autograd_array_error() -> None:
    x = cp.tensor([cp.value(float(i)) for i in range(100)])
    y = (x * 2.0).sum()
    with pytest.raises(NotImplementedError):
        cp.grad(y, list(x.values))


def test_hybrid_values_access_keeps_packed() -> None:
    """Reading the element values of an array-only tensor must not change its source"""
    x = cp.tensor([cp.value(float(i)) for i in range(100)])
    y = x * 2.0
    assert y._packed_array() is not None
    refs = y.values  # element references into the array
    assert y.values is refs  # cached
    assert y._packed_array() is not None
    assert y.reshape(10, 10)._packed_array() is not None

    tg = cp.Target()
    tg.compile(y)
    tg.run()
    assert list(tg.read_value(y).values) == [i * 2.0 for i in range(100)]


INDEX_CASES_3D: list[Any] = [(0, 1, 0), (1, 1, 1), (-1, 0, -1), 0, 1, -1, (1, 0), (0, slice(None), 1),
                             slice(1, None), (slice(None), -1)]
INDEX_CASES_2D: list[Any] = [1, -1, (slice(None), 2), (slice(0, 2), slice(1, 3)), (-1, slice(None)),
                             (slice(None, None, 2), slice(None, None, 2)), slice(1, None),
                             (slice(None), slice(-1, None)), (2, 1)]


@pytest.mark.parametrize('data,cases', [([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], INDEX_CASES_3D),
                                        ([[10, 20, 30], [40, 50, 60], [70, 80, 90]], INDEX_CASES_2D)])
def test_hybrid_indexing(data: Any, cases: list[Any]) -> None:
    """Indexing and slicing of an array-backed tensor (strided copies for sub-tensors,
    single elements from the array memory) compared with the scalar path"""
    def build(packed: bool) -> Any:
        t = cp.tensor([[[cp.value(v) for v in r] if isinstance(r, list) else cp.value(r) for r in row] for row in data],
                      packed=packed)
        return t + 1  # array-backed result for packed=True

    t_packed, t_scalar = build(True), build(False)
    assert t_packed._packed_array() is not None

    res_packed = [t_packed[k] for k in cases]
    res_scalar = [t_scalar[k] for k in cases]
    out_packed = evaluate(*res_packed)
    out_scalar = evaluate(*res_scalar)

    for k, rp, rs, op, os in zip(cases, res_packed, res_scalar, out_packed, out_scalar):
        assert rp.shape == rs.shape, k
        if rp.ndim > 0:
            assert rp._packed_array() is not None, k  # sub-tensor by strided copy
        assert list(op.values) == list(os.values), k

    with pytest.raises(IndexError):
        t_packed[len(data)]
    with pytest.raises(IndexError):
        t_packed[(0,) * (t_packed.ndim - 1) + (len(data[0][0]) if t_packed.ndim == 3 else 3,)]
    with pytest.raises(IndexError):
        t_packed[(0,) * (t_packed.ndim + 1)]


def test_array_dtype_promotion() -> None:
    """Mixed elements are promoted: float if any element is float, bool as int"""
    i = cp.value(3)
    a = cp.array([i, 2.5, 1])  # int variable converted by a float_int stencil
    b = cp.array([True, 2])
    c = cp.array([cp.value(True), i])
    assert (a.dtype, b.dtype, c.dtype) == ('float', 'int', 'int')
    assert cp.array([1, 2], 'float').dtype == 'float'

    out = evaluate(a, b, c * 2)
    assert out == [[3.0, 2.5, 1.0], [1, 2], [2, 6]]
    assert all(type(v) is float for v in out[0])


def test_bool_array() -> None:
    """Bool arrays are labelled bool and stored and computed as int"""
    x = cp.value(2.0)
    flags = [True, False, True, True]
    b = cp.array(flags)
    bv = cp.array([x > 1.0, x > 3.0])  # bool values from comparisons
    assert (b.dtype, b.net.dtype, bv.dtype) == ('bool', 'int', 'bool')
    assert cp.array([1, 0], 'int').dtype == 'int'

    # Views and elements keep the label, operations compute as int
    view = b.reshape(2, 2).T[0]
    assert view.dtype == 'bool' and b[2].dtype == 'bool'
    s = b + b
    assert s.dtype == 'int' and (b + 0).dtype == 'int' and b.sum().dtype == 'int'
    w = b * 2.5
    assert w.dtype == 'float'

    tg = cp.Target()
    tg.compile(b, bv, view, s, w, b.sum())
    tg.run()
    assert tg.read_value(b) == flags and all(type(v) is bool for v in tg.read_value(b))
    assert tg.read_value(bv) == [True, False]
    assert tg.read_value(view) == [True, True]
    assert tg.read_value(s) == [2, 0, 2, 2] and all(type(v) is int for v in tg.read_value(s))
    assert tg.read_value(w) == [2.5, 0.0, 2.5, 2.5]
    assert tg.read_value(b[1]) is False

    tg.write_value(b, [False, True, False, False])
    tg.run()
    assert tg.read_value(b) == [False, True, False, False]
    assert tg.read_value(s) == [0, 2, 0, 0]

    with pytest.raises(ValueError):
        cp.array([0, 2], 'bool')  # only bool elements in a bool array


def test_bool_tensor_packed() -> None:
    x = cp.tensor([cp.value(float(i)) for i in range(100)])
    flags = x > 50.0  # bool tensor
    assert flags.dtype == 'bool'
    packed = cp.tensor(flags, packed=True)
    assert packed._get_array() is not None and packed._get_array().dtype == 'bool'  # type: ignore[union-attr]
    count = packed.sum()
    out, = evaluate(count)
    assert out == 49


def test_hybrid_vector() -> None:
    """Large vectors use array stencils, results equal the scalar path"""
    n = 100
    def build(packed: bool | None) -> list[Any]:
        v1 = cp.vector((cp.value(float(i)) for i in range(n)), packed=packed)
        v2 = cp.vector((cp.value(float(i % 7) + 0.5) for i in range(n)), packed=packed)
        s = cp.value(1.5)
        return [v1 + v2, v1 * s - 1.0, 2.0 / v2, -v1, v1 ** 2, v1 @ v2, v1.sum(), v2.magnitude(),
                v1.normalize(), v1[10:20], cp.sqrt(v2), v1 > 50.0]

    res_packed = build(None)
    res_scalar = build(False)
    assert isinstance(res_packed[0], cp.vector) and res_packed[0]._packed_array() is not None
    assert res_scalar[0]._packed_array() is None

    out_packed = evaluate(*res_packed)
    out_scalar = evaluate(*res_scalar)
    for p, s in zip(out_packed, out_scalar):
        p_vals = list(p.values) if isinstance(p, cp.vector) else [p]
        s_vals = list(s.values) if isinstance(s, cp.vector) else [s]
        assert p_vals == pytest.approx(s_vals, rel=1e-5)

    # Small vectors stay unpacked, the threshold is set per class
    assert (cp.vector([cp.value(1.0), cp.value(2.0)]) * 2.0)._packed_array() is None
    assert cp.vector.pack_threshold == cp.tensor.pack_threshold == 64


def test_vector_dtype_promotion() -> None:
    i = cp.value(3)
    v = cp.vector([i, 2.5, 1])
    assert v.dtype == 'float' and [type(x).__name__ for x in v.values] == ['value', 'float', 'float']
    assert cp.vector([1, 2]).dtype == 'int'
    assert cp.vector(cp.tensor([1, 2])).values == (1, 2)  # 0-d tensors are unwrapped


def test_quaternion_not_packed_or_promoted() -> None:
    """Quaternions keep their elements as given and never use array stencils"""
    q = cp.quaternion(1, 2, 3.0, cp.value(4))
    assert q.values[0] == 1 and type(q.values[0]) is int  # no type promotion
    assert cp.quaternion.pack_threshold is None
    assert q._get_array() is None and q._get_array(force=True) is None
