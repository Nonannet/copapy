"""Tests for the vector class and the vector helper functions."""
import math
import operator
from typing import Any, Callable

import pytest

import copapy as cp
from copapy import filters
from runner_helpers import trig_tol


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def variables(values: list[float]) -> cp.vector[float]:
    return cp.vector(cp.value(v) for v in values)


A = [1.0, 2.0, 3.0]
B = [4.0, -5.0, 6.5]


def test_construction():
    assert cp.vector([1.0, 2.0, 3.0]).values == (1.0, 2.0, 3.0)
    assert cp.vector(range(3)).values == (0, 1, 2)
    assert cp.vector(float(v) for v in range(3)).values == (0.0, 1.0, 2.0)

    v = cp.vector([1.1, 2, cp.value(5)])
    assert isinstance(v.values[2], cp.value)
    assert v.values[:2] == (1.1, 2)

    assert cp.vector(cp.tensor([1, 2])).shape == (2,)


def test_sequence_protocol():
    v = cp.vector([10, 20, 30, 40])

    assert len(v) == 4
    assert v.shape == (4,)
    assert list(v) == [10, 20, 30, 40]
    assert v[0] == 10
    assert v[-1] == 40

    s = v[1:3]
    assert isinstance(s, cp.vector)
    assert s.values == (20, 30)
    assert v[::2].values == (10, 30)


OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]


@pytest.mark.parametrize("op", OPERATORS, ids=[op.__name__ for op in OPERATORS])
def test_elementwise_operators(op: Callable[[Any, Any], Any]):
    a = [1.5, 2.0, 3.0]
    b = [4.0, 0.5, 2.0]
    s = 2.5

    cases = [
        (op(cp.vector(a), cp.vector(b)), [op(x, y) for x, y in zip(a, b)]),
        (op(cp.vector(a), s), [op(x, s) for x in a]),
        (op(s, cp.vector(a)), [op(s, x) for x in a]),
    ]

    for res, ref in cases:
        assert isinstance(res, cp.vector)
        assert res.values == pytest.approx(ref)  # pyright: ignore[reportUnknownMemberType]

    compiled = [op(variables(a), variables(b)), op(variables(a), cp.vector(b)), op(variables(a), s), op(s, variables(a))]
    refs = [cases[0][1], cases[0][1], cases[1][1], cases[2][1]]
    for res, ref in zip(evaluate(*compiled), refs):
        assert res.values == pytest.approx(ref, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("op", OPERATORS, ids=[op.__name__ for op in OPERATORS])
def test_value_on_left_side(op: Callable[[Any, Any], Any]):
    """cp.value op vector must dispatch to the reflected vector operator"""
    a = [1.5, 2.0, 3.0]
    res = op(cp.value(2.5), cp.vector(a))
    assert isinstance(res, cp.vector)

    compiled, = evaluate(res)
    assert compiled.values == pytest.approx([op(2.5, x) for x in a], rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


def test_negation_and_comparison():
    v = cp.vector([-1.0, 0.0, 2.0])
    assert (-v).values == (1.0, -0.0, -2.0)
    assert (v > 0.0).values == (False, False, True)
    assert (v < 0.0).values == (True, False, False)
    assert (v == cp.vector([-1.0, 1.0, 2.0])).values == (True, False, True)

    vv = variables([-1.0, 0.0, 2.0])
    neg, gt, eq = evaluate(-vv, vv > 0.0, vv == cp.vector([-1.0, 1.0, 2.0]))
    assert neg.values == (1.0, 0.0, -2.0)
    assert gt.values == (False, False, True)
    assert eq.values == (True, False, True)


def test_length_mismatch():
    with pytest.raises((AssertionError, ValueError)):
        cp.vector([1.0, 2.0, 3.0]) + cp.vector([1.0, 2.0])
    with pytest.raises((AssertionError, ValueError)):
        cp.vector([1.0, 2.0, 3.0]).dot(cp.vector([1.0, 2.0]))


def test_cross_requires_3d():
    with pytest.raises((AssertionError, ValueError)):
        cp.vector([1.0, 0.0]).cross(cp.vector([0.0, 1.0]))


def vector_ops(a: cp.vector[Any], b: cp.vector[Any]) -> dict[str, Any]:
    """Operations evaluated once with constant and once with variable vectors"""
    return {
        'dot': a.dot(b),
        'matmul': a @ b,
        'cross': a.cross(b),
        'cross_anti': b.cross(a),
        'sum': a.sum(),
        'magnitude': a.magnitude(),
        'normalize': a.normalize(),
        'map': a.map(lambda x: x * x + 1),
        'distance': cp.distance(a, b),
        'scalar_projection': cp.scalar_projection(a, b),
        'vector_projection': cp.vector_projection(a, b),
        'angle_between': cp.angle_between(a, b),
        'rotate_vector': cp.rotate_vector(a, cp.vector([0.0, 0.0, 1.0]), math.pi / 2),
        'sqrt': cp.sqrt(a),
        'sin': cp.sin(b),
        'exp': cp.exp(a),
        'minimum': cp.minimum(a, b),
        'maximum': cp.maximum(a, 2.0),
        'clamp': cp.clamp(b, -1.0, 2.0),
        'median': filters.median(cp.vector([*a, *b, 0.5])),
    }


def expected_ops(a: list[float], b: list[float]) -> dict[str, Any]:
    """Reference values calculated without copapy"""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return {
        'dot': dot,
        'matmul': dot,
        'cross': cross,
        'cross_anti': [-c for c in cross],
        'sum': sum(a),
        'magnitude': mag_a,
        'normalize': [x / mag_a for x in a],
        'map': [x * x + 1 for x in a],
        'distance': math.dist(a, b),
        'scalar_projection': dot / mag_b,
        'vector_projection': [dot / mag_b ** 2 * y for y in b],
        'angle_between': math.acos(dot / (mag_a * mag_b)),
        'rotate_vector': [-a[1], a[0], a[2]],
        'sqrt': [math.sqrt(x) for x in a],
        'sin': [math.sin(x) for x in b],
        'exp': [math.exp(x) for x in a],
        'minimum': [min(x, y) for x, y in zip(a, b)],
        'maximum': [max(x, 2.0) for x in a],
        'clamp': [min(max(x, -1.0), 2.0) for x in b],
        'median': sorted([*a, *b, 0.5])[3],
    }


def values_of(x: Any) -> Any:
    return x.values if isinstance(x, cp.vector) else x


@pytest.mark.parametrize("name", list(expected_ops(A, B)))
def test_vector_operation(name: str):
    ref = expected_ops(A, B)[name]

    res = vector_ops(cp.vector(A), cp.vector(B))[name]
    assert not isinstance(res, cp.value)
    assert values_of(res) == pytest.approx(ref, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]

    res, = evaluate(vector_ops(variables(A), variables(B))[name])
    assert values_of(res) == pytest.approx(ref, rel=1e-5, abs=trig_tol())  # pyright: ignore[reportUnknownMemberType]


def test_normalize_properties():
    v = cp.vector([3.0, -4.0, 12.0])
    n = v.normalize()
    assert n.magnitude() == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
    assert n.values == pytest.approx([3 / 13, -4 / 13, 12 / 13])  # pyright: ignore[reportUnknownMemberType]


def test_cross_product_properties():
    a = cp.vector(A)
    b = cp.vector(B)
    c = a.cross(b)
    assert c.dot(a) == pytest.approx(0.0, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
    assert c.dot(b) == pytest.approx(0.0, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
    assert a.cross(a).values == pytest.approx([0.0, 0.0, 0.0])  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("v1", "v2", "expected"),
    [
        ([1.0], [2.0], 0.0),
        ([1.0, 0.0], [0.0, 1.0], math.pi / 2),
        ([1.0, 0.0], [-2.0, 0.0], math.pi),
        ([5.0, 0.0, 0.0], [5.0, 5.0, 0.0], math.pi / 4),
        ([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], math.pi / 4),
    ],
)
def test_angle_between(v1: list[float], v2: list[float], expected: float):
    assert cp.angle_between(cp.vector(v1), cp.vector(v2)) == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]

    res, = evaluate(cp.angle_between(variables(v1), cp.vector(v2)))
    assert res == pytest.approx(expected, abs=1e-3)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize(
    ("axis", "angle", "expected"),
    [
        ([0.0, 0.0, 1.0], math.pi / 2, [-2.0, 1.0, 3.0]),
        ([0.0, 0.0, 5.0], math.pi / 2, [-2.0, 1.0, 3.0]),  # axis does not need to be normalized
        ([1.0, 0.0, 0.0], math.pi, [1.0, -2.0, -3.0]),
        ([0.0, 1.0, 0.0], -math.pi / 2, [-3.0, 2.0, 1.0]),
        ([1.0, 2.0, 3.0], 1.234, [1.0, 2.0, 3.0]),  # rotation around the vector itself
        ([1.0, 1.0, 0.0], 0.0, [1.0, 2.0, 3.0]),
    ],
)
def test_rotate_vector(axis: list[float], angle: float, expected: list[float]):
    v = cp.vector([1.0, 2.0, 3.0])
    rotated = cp.rotate_vector(v, cp.vector(axis), angle)
    assert isinstance(rotated, cp.vector)
    assert rotated.values == pytest.approx(expected, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]
    assert rotated.magnitude() == pytest.approx(v.magnitude())  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("vlist", [[50, 21, 20, 10, 22, 1, 80, 70, 90], [3], [2.5, -1.0, 7.0], [5, 5, 1, 5, 9]])
def test_median(vlist: list[float]):
    ref = sorted(vlist)[len(vlist) // 2]

    assert filters.median(cp.vector(vlist)) == ref

    res, = evaluate(filters.median(cp.vector(cp.value(v) for v in vlist)))
    assert res == ref


def test_vector_concat():
    concat_vec = cp.concat([cp.vector([1, 2, 3]), cp.vector([4, 5]), cp.vector([6])])
    assert concat_vec.shape == (6,)
    assert concat_vec.values == (1, 2, 3, 4, 5, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
