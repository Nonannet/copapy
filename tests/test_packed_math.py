"""Element-wise functions, comparisons, iif and the sum along axes for arrays and
packed vectors and tensors: array stencils instead of one operation per element."""
import math
from typing import Any, Callable

import pytest

import copapy as cp

A = [-1.5, -0.25, 0.0, 0.5, 0.75, 2.0, 3.5, -4.0]
B = [0.5, -0.25, 1.0, -2.0, 0.75, 4.0, -3.5, 0.25]


def make(kind: str, data: list[Any] = A) -> Any:
    """Container of copapy values, vectors and tensors are packed"""
    values = [cp.value(v) for v in data]
    if kind == 'array':
        return cp.array(values)
    if kind == 'vector':
        return cp.vector(values, packed=True)
    return cp.tensor(values, (2, len(data) // 2), packed=True)


def flat(data: Any) -> list[Any]:
    """Flat list of the elements read from the target"""
    if isinstance(data, cp.vector | cp.tensor):
        return list(data.values)
    if isinstance(data, list):
        return [x for row in data for x in flat(row)]
    return [data]


def run(*results: Any) -> list[list[Any]]:
    tg = cp.Target()
    tg.compile(*results)
    tg.run()
    return [flat(tg.read_value(r)) for r in results]


def scalar_ops(*results: Any) -> list[str]:
    """Names of the scalar operations the results depend on, without the
    constants and the access to array elements"""
    nets: list[Any] = []
    for r in results:
        if isinstance(r, cp.array):
            nets.append(r.net)
        elif isinstance(r, cp.value):
            nets.append(r.net)
        else:
            nets += [v.net for v in r.values if isinstance(v, cp.value)]
    seen: set[int] = set()
    names: list[str] = []
    while nets:
        net = nets.pop()
        if id(net) in seen:
            continue
        seen.add(id(net))
        source = net.source
        name = getattr(source, 'name', '')
        if name and 'arr' not in name and not name.startswith(('const', 'element')):
            names.append(name)
        nets += list(getattr(source, 'args', ()))
    return names


KINDS = ['vector', 'tensor', 'array']

FUNCTIONS: dict[str, tuple[Callable[[Any, Any], Any], Callable[[float, float], float]]] = {
    'exp': (lambda x, y: cp.exp(x), lambda a, b: math.exp(a)),
    'sin': (lambda x, y: cp.sin(x), lambda a, b: math.sin(a)),
    'atan': (lambda x, y: cp.atan(x), lambda a, b: math.atan(a)),
    'tanh': (lambda x, y: cp.tanh(x), lambda a, b: math.tanh(a)),
    'sqrt abs': (lambda x, y: cp.sqrt(cp.abs(x)), lambda a, b: math.sqrt(abs(a))),
    'sign': (lambda x, y: cp.sign(x), lambda a, b: (a > 0) - (a < 0)),
    'clamp': (lambda x, y: cp.clamp(x, -0.5, 1.0), lambda a, b: min(max(a, -0.5), 1.0)),
    'clamp value': (lambda x, y: cp.clamp(x, cp.value(-0.5), 1), lambda a, b: min(max(a, -0.5), 1.0)),
    'minimum': (lambda x, y: cp.minimum(x, y), lambda a, b: min(a, b)),
    'minimum scalar': (lambda x, y: cp.minimum(x, 0.5), lambda a, b: min(a, 0.5)),
    'maximum': (lambda x, y: cp.maximum(x, y), lambda a, b: max(a, b)),
    'maximum scalar': (lambda x, y: cp.maximum(0.5, x), lambda a, b: max(a, 0.5)),
    'atan2': (lambda x, y: cp.atan2(x, y), lambda a, b: math.atan2(a, b)),
    'atan2 scalar': (lambda x, y: cp.atan2(0.5, x), lambda a, b: math.atan2(0.5, a)),
    'pow': (lambda x, y: cp.pow(cp.abs(x), y), lambda a, b: abs(a) ** b),
    'pow scalar': (lambda x, y: cp.pow(2.5, x), lambda a, b: 2.5 ** a),
    'gt': (lambda x, y: x > y, lambda a, b: a > b),
    'lt': (lambda x, y: x < y, lambda a, b: a < b),
    'ge': (lambda x, y: x >= y, lambda a, b: a >= b),
    'le': (lambda x, y: x <= y, lambda a, b: a <= b),
    'eq': (lambda x, y: x == y, lambda a, b: a == b),
    'ne': (lambda x, y: x != y, lambda a, b: a != b),
    'gt scalar': (lambda x, y: x > 0.5, lambda a, b: a > 0.5),
    'le scalar': (lambda x, y: x <= 0.5, lambda a, b: a <= 0.5),
    'eq scalar': (lambda x, y: x == 0.75, lambda a, b: a == 0.75),
    'ne value': (lambda x, y: x != cp.value(0.75), lambda a, b: a != 0.75),
    'sigmoid': (lambda x, y: cp.nn.sigmoid(x), lambda a, b: 1 / (1 + math.exp(-a))),
}


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_elementwise(kind: str, name: str) -> None:
    func, ref = FUNCTIONS[name]
    x = make(kind)
    result = func(x, make(kind, B))
    assert type(result) is type(x) and result.shape == x.shape
    # Only single scalar operations (e.g. the comparison of a float condition)
    assert len(scalar_ops(result)) < 2, scalar_ops(result)
    assert run(result)[0] == pytest.approx([float(ref(a, b)) for a, b in zip(A, B)], rel=1e-5, abs=1e-6)


@pytest.mark.parametrize('name', ['gt', 'le', 'eq', 'ne', 'gt scalar', 'eq scalar'])
def test_comparison_types(name: str) -> None:
    """Comparisons of vectors and tensors are bool like for unpacked ones, of arrays int"""
    func = FUNCTIONS[name][0]
    for kind in ('vector', 'tensor'):
        packed, unpacked = func(make(kind), make(kind, B)), func(cp.vector(A), cp.vector(cp.value(v) for v in B))
        assert packed.dtype == unpacked.dtype == 'bool'
    assert func(make('array'), make('array', B)).dtype == 'int'


def test_array_comparison_with_other_types() -> None:
    x = make('array')
    assert (x == None) is False and (x != 'a') is True  # noqa: E711
    assert x in [x] and len({x, x}) == 1
    ints = cp.array([1, 2, 3, 4]) == cp.array([cp.value(1), cp.value(0), cp.value(3), cp.value(5)])
    mixed = cp.array([1, 2, 3, 4]) != cp.array([cp.value(1.0), cp.value(2.5), cp.value(3.0), cp.value(4.0)])
    assert run(ints, mixed) == [[1, 0, 1, 0], [0, 1, 0, 0]]


def test_small_vectors_are_not_packed() -> None:
    """Below the pack threshold one scalar operation is used for each element"""
    x = cp.vector(cp.value(v) for v in A)
    y = cp.vector(cp.value(v) for v in B)
    results = [cp.sin(x), cp.minimum(x, y), x > y, x == y, cp.sign(x), cp.clamp(x, -0.5, 1.0), cp.iif(x[0] < 0, x, y)]
    assert all(type(r) is cp.vector for r in results)
    assert [len(scalar_ops(r)) >= len(A) for r in results] == [True] * len(results)
    refs = [[math.sin(a) for a in A], [min(a, b) for a, b in zip(A, B)], [a > b for a, b in zip(A, B)],
            [a == b for a, b in zip(A, B)], [(a > 0) - (a < 0) for a in A], [min(max(a, -0.5), 1.0) for a in A],
            A]
    for res, ref in zip(run(*results), refs):
        assert res == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_clamp_numbers_and_values() -> None:
    assert cp.clamp(5, 0, 3) == 3 and cp.clamp(-2.5, -1.0, 1.0) == -1.0 and cp.clamp(0.25, -1.0, 1.0) == 0.25
    results = [cp.clamp(cp.value(x), -1.0, 2.0) for x in (-3.0, 0.5, 7.0)] + [cp.clamp(cp.value(9), 1, 4)]
    assert results[3].dtype == 'int'
    assert [r for (r,) in run(*results)] == [-1.0, 0.5, 2.0, 4]


def test_iif_containers() -> None:
    """Vectors, tensors and arrays are selected as a whole by a scalar condition"""
    x, y = make('array'), make('array', B)
    flag = cp.value(1)
    ints = cp.iif(flag, cp.array([cp.value(i) for i in range(4)]), 7)
    assert ints.dtype == 'int'
    assert run(ints) == [[0, 1, 2, 3]]
    for kind in KINDS:
        a, b = make(kind), make(kind, B)
        results = [cp.iif(cp.value(0.5) > 0.25, a, b * 2.0), cp.iif(cp.value(0), a, b * 2.0), cp.iif(cp.value(0.0), 1.5, a)]
        assert all(type(r) is type(a) and r.shape == a.shape for r in results)
        # Only scalar operations for the condition and a scalar result
        assert all(len(scalar_ops(r)) <= 4 for r in results), [scalar_ops(r) for r in results]
        assert run(*results) == [A, [2.0 * v for v in B], A]
    # Condition known at trace time: no code
    assert cp.iif(1, x, y) is x and cp.iif(0.0, x, y) is y
    assert cp.iif(0, cp.vector([1.0, 2.0]), cp.vector([3.0, 4.0])).values == (3.0, 4.0)
    # Selected at runtime
    tg = cp.Target()
    result = cp.iif(flag, x, y)
    tg.compile(result)
    for cond, ref in ((1, A), (0, B)):
        tg.write_value(flag, cond)
        tg.run()
        assert tg.read_value(result) == ref
    with pytest.raises(AssertionError):
        cp.iif(x > 0, x, y)


@pytest.mark.parametrize('kind', ['tensor', 'array'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1, (0, 1), (1, 2), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_sum_axis(kind: str, axis: Any, keepdims: bool) -> None:
    shape = (2, 3, 4)
    data = [0.5 * i * i - 3.0 * i + 1.0 for i in range(24)]
    values = [cp.value(v) for v in data]
    x: Any = cp.tensor(values, shape, packed=True) if kind == 'tensor' else cp.array(values).reshape(*shape)
    result = x.sum(axis, keepdims)

    axes = sorted(a % 3 for a in ((axis,) if isinstance(axis, int) else axis))
    ref_shape = [1 if i in axes else d for i, d in enumerate(shape)]
    ref = [0.0] * (ref_shape[0] * ref_shape[1] * ref_shape[2])
    for i in range(24):
        index = (i // 12, i // 4 % 3, i % 4)
        reduced = [0 if k in axes else index[k] for k in range(3)]
        ref[(reduced[0] * ref_shape[1] + reduced[1]) * ref_shape[2] + reduced[2]] += data[i]

    if len(axes) == 3 and not keepdims:
        assert isinstance(result, cp.value)
    else:
        assert type(result) is type(x)
        assert result.shape == (tuple(ref_shape) if keepdims else tuple(d for i, d in enumerate(shape) if i not in axes))
    assert not scalar_ops(result)
    assert run(result)[0] == pytest.approx(ref, rel=1e-5, abs=1e-5)


def test_sum_axis_int_and_mean() -> None:
    ints: Any = cp.array([[cp.value(1), cp.value(2), cp.value(3)], [cp.value(4), cp.value(5), cp.value(6)]])
    columns, rows = ints.sum(0), ints.sum(axis=1, keepdims=True)
    assert columns.dtype == 'int' and rows.shape == (2, 1)
    mean = cp.tensor([cp.value(float(i)) for i in range(6)], (2, 3), packed=True).mean(axis=0)
    assert not scalar_ops(mean)
    assert run(columns, rows, mean) == [[5, 7, 9], [6, 15], [1.5, 2.5, 3.5]]
    with pytest.raises(ValueError):
        ints.sum(2)


@pytest.mark.parametrize('kind', KINDS)
def test_grad(kind: str) -> None:
    """Gradients through the element-wise functions of packed vectors and tensors"""
    x, y = make(kind), make(kind, B)
    terms = {
        'sin': (cp.sin(x).sum(), [math.cos(a) for a in A]),
        'clamp': ((cp.clamp(x, -0.5, 1.0) * y).sum(), [b if -0.5 < a < 1.0 else 0.0 for a, b in zip(A, B)]),
        'minimum': ((cp.minimum(x, y) * 2.0).sum(), [2.0 if a < b else 0.0 for a, b in zip(A, B)]),
        'iif': ((cp.iif(x.sum() > 100.0, x * x, x * 3.0)).sum(), [3.0] * len(A)),
        'comparison': ((x * (x > 0.1)).sum(), [1.0 if a > 0.1 else 0.0 for a in A]),
    }
    grads = [cp.grad(term, x) for term, _ in terms.values()]
    for res, (_, ref) in zip(run(*grads), terms.values()):
        assert res == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_grad_sum_axis() -> None:
    data = [float(i) - 2.5 for i in range(24)]
    weights = [[1.0, -2.0, 0.5, 3.0], [2.0, 0.0, -1.0, 4.0]]
    x = cp.array([cp.value(v) for v in data]).reshape(2, 3, 4)
    result = (x.sum(1) * cp.array(weights)).sum() + (x.sum((0, 2)) * cp.array([1.0, 10.0, 100.0])).sum()
    ref = [weights[i // 12][i % 4] + 10.0 ** (i // 4 % 3) for i in range(24)]
    assert run(cp.grad(result, x))[0] == pytest.approx(ref)
    packed = cp.tensor([cp.value(v) for v in data[:6]], (2, 3), packed=True)
    grad = cp.grad((packed.sum(axis=0) * cp.tensor([1.0, 2.0, 3.0])).sum(), packed)
    assert run(grad)[0] == pytest.approx([1.0, 2.0, 3.0] * 2)
