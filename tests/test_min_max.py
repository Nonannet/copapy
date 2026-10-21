"""Tests for the reductions min and max of vectors, tensors and arrays"""
import random
from typing import Any

import pytest

import copapy as cp
from copapy.backend import get_dag_stats


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def sample(n: int, dtype: str, seed: int) -> list[Any]:
    rnd = random.Random(seed)
    if dtype == 'int':
        return [rnd.randint(-50, 50) for _ in range(n)]
    return [round(rnd.uniform(-10, 10), 2) for _ in range(n)]


@pytest.mark.parametrize('n', [1, 2, 7, 8, 9, 16, 31, 70])
@pytest.mark.parametrize('dtype', ['int', 'float'])
def test_min_max(n: int, dtype: str) -> None:
    data = sample(n, dtype, n)
    arr = cp.array(data)
    vec = cp.vector(cp.value(x) for x in data)
    ten = cp.tensor([cp.value(x) for x in data])
    unpacked = cp.vector((cp.value(x) for x in data), packed=False)
    packed = cp.tensor([cp.value(x) for x in data], packed=True)

    inputs = [arr, vec, ten, unpacked, packed]
    results = evaluate(*[cp.min(x) for x in inputs], *[cp.max(x) for x in inputs], arr.min(), arr.max())
    assert results[:5] == pytest.approx([min(data)] * 5)
    assert results[5:10] == pytest.approx([max(data)] * 5)
    assert results[10:] == pytest.approx([min(data), max(data)])
    assert all(isinstance(r, int if dtype == 'int' else float) for r in results)


def test_min_max_constants() -> None:
    """Constants are reduced at trace time"""
    assert cp.min(cp.vector([3, -1, 2])) == -1
    assert cp.max(cp.tensor([[1.5, 4.0], [-2.0, 0.5]])) == 4.0

    # Constant elements of a mixed vector do not generate operations
    v = cp.vector([5.0, cp.value(2.0), -3.0, cp.value(7.0), 9.0])
    low, high = cp.min(v), cp.max(v)
    stats = get_dag_stats([(low + high).net])  # operations in front of the addition
    assert (stats.get('min_float_float'), stats.get('max_float_float')) == (2, 2)
    assert evaluate(low, high) == [-3.0, 9.0]


def test_min_max_tensor_shape() -> None:
    data = [[1.0, -4.5, 2.0], [8.0, 0.0, -1.0]]
    t = cp.tensor([[cp.value(x) for x in row] for row in data])
    assert evaluate(cp.min(t), cp.max(t), cp.min(cp.array(data)), cp.max(cp.array(data).T)) == [-4.5, 8.0, -4.5, 8.0]


def test_min_max_code_size() -> None:
    """One array stencil instead of one operation per element"""
    v = cp.vector(cp.value(float(i % 11)) for i in range(100))
    stats = get_dag_stats([(cp.max(v) + 1.0).net])  # operations in front of the addition
    assert stats.get('max_floatarr') == 1
    assert 'max_float_float' not in stats


def test_min_max_errors() -> None:
    with pytest.raises(ValueError):
        cp.min(cp.vector([]))
    with pytest.raises(TypeError):
        cp.max(3.0)  # type: ignore[call-overload]
