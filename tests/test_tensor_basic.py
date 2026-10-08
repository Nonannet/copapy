"""Basic tests for the tensor class: construction, indexing, element-wise
operations, broadcasting, reshaping and reductions."""
import itertools
import math
import operator
from typing import Any, Callable

import pytest

import copapy as cp
from runner_helpers import trig_tol


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def nested_shape(data: Any) -> tuple[int, ...]:
    return (len(data),) + nested_shape(data[0]) if isinstance(data, list) else ()


def flat(data: Any) -> list[Any]:
    return [x for d in data for x in flat(d)] if isinstance(data, list) else [data]


def broadcast_ref(op: Callable[[Any, Any], Any], a: Any, b: Any) -> tuple[tuple[int, ...], list[Any]]:
    """Numpy style broadcasting of nested lists as reference"""
    sa, sb = nested_shape(a), nested_shape(b)
    ndim = max(len(sa), len(sb))
    sa = (1,) * (ndim - len(sa)) + sa
    sb = (1,) * (ndim - len(sb)) + sb
    shape = tuple(max(x, y) for x, y in zip(sa, sb))
    fa, fb = flat(a), flat(b)

    def element(f: list[Any], s: tuple[int, ...], idx: tuple[int, ...]) -> Any:
        pos = 0
        for i, n in zip(idx, s):
            pos = pos * n + (i if n > 1 else 0)
        return f[pos]

    return shape, [op(element(fa, sa, idx), element(fb, sb, idx)) for idx in itertools.product(*(range(n) for n in shape))]


T3D = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


@pytest.mark.parametrize("data", [42, 2.5, [1, 2, 3, 4, 5], [[1, 2, 3], [4, 5, 6]], T3D, [[[1, 2, 3, 4]] * 3] * 2],
                         ids=['scalar_int', 'scalar_float', '1d', '2d', '3d', '3d_nonsquare'])
def test_construction(data: Any):
    t = cp.tensor(data)
    shape = nested_shape(data)
    assert t.shape == shape
    assert t.ndim == len(shape)
    assert t.size() == math.prod(shape)
    assert t.values == tuple(flat(data))


def test_construction_from_vector_and_variables():
    t = cp.tensor(cp.vector([1.0, 2.0]))
    assert t.shape == (2,)
    assert t.values == (1.0, 2.0)

    t = cp.tensor([[cp.value(1), 2], [3, cp.value(4)]])
    assert t.shape == (2, 2)
    assert isinstance(t.values[0], cp.value)
    assert t.values[1:3] == (2, 3)


def test_ragged_input():
    with pytest.raises(ValueError):
        cp.tensor([[1, 2], [3]])


def test_shape_does_not_match_values():
    with pytest.raises(ValueError):
        cp.tensor([1, 2, 3], (2, 2))
    with pytest.raises(ValueError):
        cp.tensor([[1, 2], [3, 4]], (2, 2))

    # 0-d tensors are accepted as scalars
    t = cp.tensor([cp.tensor(1), cp.tensor(2)], (2,))
    assert t.values == (1, 2)


def test_truth_value_is_ambiguous():
    t = cp.tensor([1, 2])
    with pytest.raises(TypeError):
        bool(t == cp.tensor([1, 3]))
    with pytest.raises(TypeError):
        bool(cp.vector([1, 2]) == cp.vector([1, 2]))
    with pytest.raises(TypeError):
        if t:
            pass


def test_indexing():
    t = cp.tensor(T3D)

    assert t[0, 1, 0].get_scalar() == 3
    assert t[1, 1, 1].get_scalar() == 8
    assert t[-1, 0, -1].get_scalar() == 6
    assert t[1][0][1].get_scalar() == 6
    assert t.get_scalar(0, 1, 1) == 4

    assert t[0].shape == (2, 2)
    assert t[0].values == (1, 2, 3, 4)
    assert t[1, 0].values == (5, 6)

    with pytest.raises(IndexError):
        t[2]
    with pytest.raises(IndexError):
        t[0, 0, 2]


def test_slicing():
    t = cp.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    cases = [
        (t[1], (3,), (40, 50, 60)),
        (t[:, 2], (3,), (30, 60, 90)),
        (t[0:2, 1:3], (2, 2), (20, 30, 50, 60)),
        (t[-1, :], (3,), (70, 80, 90)),
        (t[::2, ::2], (2, 2), (10, 30, 70, 90)),
        (t[1:], (2, 3), (40, 50, 60, 70, 80, 90)),
        (t[:, -1:], (3, 1), (30, 60, 90)),
    ]
    for s, shape, values in cases:
        assert s.shape == shape
        assert s.values == values

    assert cp.tensor([0, 1, 2, 3, 4, 5])[::2].values == (0, 2, 4)


OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]


@pytest.mark.parametrize("op", OPERATORS, ids=[op.__name__ for op in OPERATORS])
def test_elementwise_operators(op: Callable[[Any, Any], Any]):
    a = [[1.0, 2.0], [3.0, 4.5]]
    b = [[2.0, 0.5], [4.0, 1.5]]
    s = 1.5

    cases = [
        (op(cp.tensor(a), cp.tensor(b)), [op(x, y) for x, y in zip(flat(a), flat(b))]),
        (op(cp.tensor(a), s), [op(x, s) for x in flat(a)]),
    ]
    for res, ref in cases:
        assert isinstance(res, cp.tensor)
        assert res.shape == (2, 2)
        assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]

    va = cp.tensor([[cp.value(x) for x in row] for row in a])
    compiled = evaluate(op(va, cp.tensor(b)), op(va, s))
    for res, (_, ref) in zip(compiled, cases):
        assert res.shape == (2, 2)
        assert res.values == pytest.approx(ref, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("op", OPERATORS, ids=[op.__name__ for op in OPERATORS])
def test_reflected_operators(op: Callable[[Any, Any], Any]):
    """Scalar on the left side: s - t must be s - t and not t - s"""
    a = [[1.0, 2.0], [3.0, 4.5]]
    s = 1.5
    ref = [op(s, x) for x in flat(a)]

    res = op(s, cp.tensor(a))
    assert isinstance(res, cp.tensor)
    assert res.shape == (2, 2)
    assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]

    res, = evaluate(op(s, cp.tensor([[cp.value(x) for x in row] for row in a])))
    assert res.shape == (2, 2)
    assert res.values == pytest.approx(ref, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_rsub():
    t = cp.tensor([[1, 2], [3, 4]])

    assert (10 - t).values == (9, 8, 7, 6)
    assert ([[10, 20], [30, 40]] - t).values == (9, 18, 27, 36)

    res, = evaluate(cp.value(10) - t)
    assert res.values == (9, 8, 7, 6)


def test_negation_and_comparison():
    t = cp.tensor([[1, -2], [3, 0]])
    assert (-t).values == (-1, 2, -3, 0)
    assert (t > 0).values == (True, False, True, False)
    assert (t == cp.tensor([[1, 2], [3, 0]])).values == (True, False, True, True)


def test_integer_division_result():
    assert (cp.tensor([1, 2, 3]) / 2).values == (0.5, 1.0, 1.5)


def test_mixed_constant_types():
    """Equal constants of different type (1, 1.0, True) are promoted to the
    type of the tensor: float if any element is float, bool counts as int"""
    vt = cp.tensor([cp.value(1), cp.value(1), cp.value(1)])
    assert vt.dtype == 'int'

    ct = cp.tensor([1, 1.0, 1])
    assert ct.dtype == 'float'
    assert ct.values == (1.0, 1.0, 1.0) and all(type(v) is float for v in ct.values)
    res = vt + ct
    assert res.dtype == 'float'
    assert [v.dtype for v in res.values] == ['float', 'float', 'float']

    res_int = vt + cp.tensor([1, True, 1])
    assert res_int.dtype == 'int'
    assert [v.dtype for v in res_int.values] == ['int', 'int', 'int']

    compiled, compiled_int = evaluate(res, res_int)
    assert compiled.values == (2.0, 2.0, 2.0)
    assert [type(v) for v in compiled.values] == [float, float, float]
    assert compiled_int.values == (2, 2, 2)
    assert [type(v) for v in compiled_int.values] == [int, int, int]


def test_dtype_promotion():
    """The tensor type is float if any element is float, int variables are
    converted when the tensor is created, the same for small and large tensors"""
    i, x = cp.value(3), cp.value(2.5)
    assert cp.tensor([i, x]).dtype == 'float'
    assert cp.tensor([i, 2]).dtype == 'int'
    assert cp.tensor([True, False]).dtype == 'bool'
    assert cp.tensor([True, 2]).values == (1, 2)
    assert cp.tensor([[1, 2], [3, 4.5]]).dtype == 'float'

    mixed = cp.tensor([i, x])
    assert [v.dtype for v in mixed.values] == ['float', 'float']  # converted in the constructor
    assert mixed.values[1] is x  # elements of the tensor type are not converted

    # Small (scalar path) and large (packed) tensors give the same types and results
    small = cp.tensor([i, x]) * 2
    large = cp.tensor([i, x] * 50) * 2
    out_small, out_large = evaluate(small, large)
    assert out_small.values == (6.0, 5.0)
    assert out_large.values == (6.0, 5.0) * 50
    assert all(type(v) is float for v in out_small.values + out_large.values)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [10.0, 20.0, 30.0]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[10.0, 20.0, 30.0]]),
        ([[1.0], [2.0]], [10.0, 20.0, 30.0]),
        (T3D, [[100.0, 200.0], [300.0, 400.0]]),
        (T3D, [1.0, 2.0]),
        (T3D, [[[10.0]], [[20.0]]]),
        ([1.0, 2.0], T3D),
        (T3D, 5.0),
    ],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul], ids=['add', 'sub', 'mul'])
def test_broadcasting(op: Callable[[Any, Any], Any], a: Any, b: Any):
    shape, ref = broadcast_ref(op, a, b)
    res = op(cp.tensor(a), cp.tensor(b))
    assert res.shape == shape
    assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]


def test_broadcasting_with_vector():
    res = cp.tensor(T3D) + cp.vector([1.0, 2.0])
    shape, ref = broadcast_ref(operator.add, T3D, [1.0, 2.0])
    assert res.shape == shape
    assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(("a", "b"), [([[1, 2, 3]], [1, 2]), ([[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]), (T3D, [1, 2, 3])])
def test_broadcasting_incompatible(a: Any, b: Any):
    with pytest.raises(ValueError):
        cp.tensor(a) + cp.tensor(b)


def test_elementwise_math_functions():
    data = [[0.0, 0.5], [1.0, 2.5]]
    t = cp.tensor(data)
    vt = cp.tensor([[cp.value(x) for x in row] for row in data])

    funcs: list[tuple[Callable[[Any], Any], Callable[[float], float]]] = [
        (cp.exp, math.exp), (cp.sqrt, math.sqrt), (cp.sin, math.sin), (cp.abs, abs),
        (lambda x: cp.minimum(x, 1.0), lambda x: min(x, 1.0)),
    ]
    compiled = evaluate(*(f(vt) for f, _ in funcs))
    for (f, ref_f), res in zip(funcs, compiled):
        ref = [ref_f(x) for x in flat(data)]
        assert f(t).shape == (2, 2)
        assert f(t).values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]
        assert res.values == pytest.approx(ref, rel=trig_tol())  # pyright: ignore[reportUnknownMemberType]


def test_map():
    t = cp.tensor(T3D)
    m = t.map(lambda x: x * 10 + 1)
    assert m.shape == (2, 2, 2)
    assert m.values == tuple(x * 10 + 1 for x in flat(T3D))


@pytest.mark.parametrize(("new_shape", "expected"), [((2, 3), (2, 3)), ((3, 2), (3, 2)), ((6,), (6,)), ((3, -1), (3, 2)),
                                                      ((-1, 2), (3, 2)), ((1, 2, 3), (1, 2, 3)), ((6, 1), (6, 1))])
def test_reshape(new_shape: tuple[int, ...], expected: tuple[int, ...]):
    t = cp.tensor([1, 2, 3, 4, 5, 6])

    r = t.reshape(*new_shape)
    assert r.shape == expected
    assert r.values == (1, 2, 3, 4, 5, 6)
    assert t.reshape(new_shape).shape == expected


def test_reshape_invalid():
    with pytest.raises(ValueError):
        cp.tensor([1, 2, 3, 4, 5, 6]).reshape(4, 2)


def test_flatten():
    for data in ([[1, 2, 3], [4, 5, 6]], T3D, [1, 2]):
        f = cp.tensor(data).flatten()
        assert f.shape == (len(flat(data)),)
        assert f.values == tuple(flat(data))

    assert cp.tensor([[1, 2], [3, 4]]).reshape(4).values == cp.tensor([[1, 2], [3, 4]]).flatten().values


def test_transpose():
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.transpose().shape == (3, 2)
    assert t.transpose().values == (1, 4, 2, 5, 3, 6)
    assert t.T.values == t.transpose().values
    assert t.T.T.values == t.values

    # 3D: axes are reversed
    data = [[[i * 100 + j * 10 + k for k in range(4)] for j in range(3)] for i in range(2)]
    t3 = cp.tensor(data)
    tt = t3.transpose()
    assert tt.shape == (4, 3, 2)
    for i, j, k in itertools.product(range(2), range(3), range(4)):
        assert tt.get_scalar(k, j, i) == data[i][j][k]


def test_sum():
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.sum() == 21

    cases = [
        (t.sum(axis=0), (3,), (5, 7, 9)),
        (t.sum(axis=1), (2,), (6, 15)),
        (t.sum(axis=0, keepdims=True), (1, 3), (5, 7, 9)),
        (t.sum(axis=1, keepdims=True), (2, 1), (6, 15)),
    ]

    t3 = cp.tensor(T3D)
    cases += [
        (t3.sum(axis=0), (2, 2), (6, 8, 10, 12)),
        (t3.sum(axis=2), (2, 2), (3, 7, 11, 15)),
        (t3.sum(axis=(0, 2)), (2,), (1 + 2 + 5 + 6, 3 + 4 + 7 + 8)),
        (t3.sum(axis=(1, 2)), (2,), (10, 26)),
        (t3.sum(axis=1, keepdims=True), (2, 1, 2), (4, 6, 12, 14)),
        (t3.sum(axis=(0, 2), keepdims=True), (1, 2, 1), (14, 22)),
        (t3.sum(keepdims=True), (1, 1, 1), (36,)),
        (t3.sum(axis=(0, 1, 2), keepdims=True), (1, 1, 1), (36,)),
        (t.sum(axis=-1), (2,), (6, 15)),
        (t.sum(axis=-2), (3,), (5, 7, 9)),
        (t3.sum(axis=(0, -1)), (2,), (14, 22)),
    ]
    assert t3.sum(axis=(0, 1, 2)) == 36

    for s, shape, values in cases:
        assert s.shape == shape
        assert s.values == values


def test_sum_invalid_axis():
    with pytest.raises(ValueError):
        cp.tensor([[1, 2], [3, 4]]).sum(axis=2)
    with pytest.raises(ValueError):
        cp.tensor([[1, 2], [3, 4]]).sum(axis=-3)


def test_mean():
    t = cp.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.mean() == pytest.approx(3.5)  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(axis=0).values == pytest.approx((2.5, 3.5, 4.5))  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(axis=1).values == pytest.approx((2.0, 5.0))  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(axis=-1).values == pytest.approx((2.0, 5.0))  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(axis=(0, 1)) == pytest.approx(3.5)  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(keepdims=True).shape == (1, 1)
    assert t.mean(keepdims=True).values == pytest.approx((3.5,))  # pyright: ignore[reportUnknownMemberType]
    assert t.mean(axis=0, keepdims=True).shape == (1, 3)
    assert t.mean(axis=0, keepdims=True).values == pytest.approx((2.5, 3.5, 4.5))  # pyright: ignore[reportUnknownMemberType]

    t3 = cp.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t3.mean(axis=(0, -1)).values == pytest.approx((3.5, 5.5))  # pyright: ignore[reportUnknownMemberType]


def test_compiled_reductions():
    data = [[1.5, 2.0, 3.0], [4.0, 5.0, -6.5]]
    vt = cp.tensor([[cp.value(x) for x in row] for row in data])

    total, sum0, sum1, mean, mean0 = evaluate(vt.sum(), vt.sum(axis=0), vt.sum(axis=1), vt.mean(), vt.mean(axis=0))
    assert total == pytest.approx(sum(flat(data)))  # pyright: ignore[reportUnknownMemberType]
    assert sum0.values == pytest.approx([a + b for a, b in zip(*data)])  # pyright: ignore[reportUnknownMemberType]
    assert sum1.values == pytest.approx([sum(r) for r in data])  # pyright: ignore[reportUnknownMemberType]
    assert mean == pytest.approx(sum(flat(data)) / 6)  # pyright: ignore[reportUnknownMemberType]
    assert mean0.values == pytest.approx([(a + b) / 2 for a, b in zip(*data)])  # pyright: ignore[reportUnknownMemberType]


def test_compiled_reshape_and_transpose():
    vt = cp.tensor([cp.value(float(x)) for x in range(6)])
    r = vt.reshape(2, 3)
    res_r, res_t, res_f = evaluate(r, r.T, r.T.flatten())
    assert res_r.shape == (2, 3)
    assert res_r.values == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert res_t.shape == (3, 2)
    assert res_t.values == (0.0, 3.0, 1.0, 4.0, 2.0, 5.0)
    assert res_f.values == (0.0, 3.0, 1.0, 4.0, 2.0, 5.0)


@pytest.mark.parametrize(("factory", "shape", "fill"), [(cp.zeros, (2, 3), 0), (cp.zeros, (3,), 0), (cp.zeros, [2, 2, 2], 0),
                                                         (cp.ones, (3, 2), 1), (cp.ones, (4,), 1), (cp.ones, [1, 2], 1)])
def test_zeros_ones(factory: Callable[[Any], cp.tensor[Any]], shape: Any, fill: int):
    t = factory(shape)
    assert t.shape == tuple(shape)
    assert t.values == (fill,) * math.prod(shape)


@pytest.mark.parametrize(("args", "expected"), [((0, 10, 2), [0, 2, 4, 6, 8]), ((5,), [0, 1, 2, 3, 4]),
                                                ((2, 5), [2, 3, 4]), ((0.0, 1.0, 0.25), [0.0, 0.25, 0.5, 0.75]),
                                                ((5, 0, -2), [5, 3, 1])])
def test_arange(args: tuple[float, ...], expected: list[float]):
    ar = cp.arange(*args)
    assert ar.shape == (len(expected),)
    assert ar.values == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_compile_constants_only(capfd: pytest.CaptureFixture[str]):
    """A program without variables has an empty data section"""
    t = cp.arange(4) * 2
    res, = evaluate(t)
    assert res.values == (0, 2, 4, 6)
    assert 'failed' not in capfd.readouterr().err


def test_concat():
    t1 = cp.tensor([[1, 2], [3, 4]])
    t2 = cp.tensor([[5, 6], [7, 8]])
    t3 = cp.tensor([[9, 10], [11, 12]])

    concat_0 = cp.concat([t1, t2, t3], axis=0)
    assert concat_0.shape == (6, 2)
    assert concat_0.values == tuple(range(1, 13))

    concat_1 = cp.concat([t1, t2, t3], axis=1)
    assert concat_1.shape == (2, 6)
    assert concat_1.values == (1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12)

    mixed = cp.concat([cp.vector([1, 2]), cp.tensor([3, 4])])
    assert mixed.values == (1, 2, 3, 4)


def test_concat_shape_mismatch():
    with pytest.raises((AssertionError, ValueError)):
        cp.concat([cp.tensor([[1, 2]]), cp.tensor([[1, 2, 3]])], axis=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
