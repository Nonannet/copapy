"""Constant folding over additions and multiplications with constants (fast_math)"""
from typing import Any

import pytest

import copapy as cp
from copapy.backend import Node, Store, canonicalize, fold_linear, get_dag_stats


def count_ops(*values: Any, fast_math: bool = True) -> dict[str, int]:
    """Number of operations by name without the types, constants and stores"""
    stats = get_dag_stats([Store(v.net) for v in values], fast_math)
    ret: dict[str, int] = {}
    for name, n in stats.items():
        if not name.startswith(('const', 'store')):
            ret[name.split('_')[0]] = ret.get(name.split('_')[0], 0) + n
    return ret


def run(*values: Any) -> Any:
    tg = cp.Target(fast_math=True)
    tg.compile(*values)
    tg.run()
    return tg, tg.read_value(values[0] if len(values) == 1 else list(values))


def test_distributed_factor() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    y = (3 * a + 6 * b) * 5

    assert count_ops(y, fast_math=False) == {'mul': 3, 'add': 1}
    assert count_ops(y) == {'mul': 2, 'add': 1}  # 15 * a + 30 * b

    tg, result = run(y)
    assert result == pytest.approx(45.0)
    tg.write_value(a, -1.0)
    tg.write_value(b, 4.0)
    tg.run()
    assert tg.read_value(y) == pytest.approx(105.0)


def test_default_is_off() -> None:
    a = cp.value(2.0)
    y = a * 3.0 * 5.0
    assert get_dag_stats([y.net]) == get_dag_stats([y.net], fast_math=False)
    assert count_ops(y, fast_math=False) == {'mul': 2}
    assert count_ops(y) == {'mul': 1}
    assert not cp.Target().fast_math


def test_shared_value_is_computed_once() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    s = 3 * a + 6 * b
    y = s * 5
    z = cp.sin(s)

    # Distributing the factor would compute 3 * a + 6 * b and 15 * a + 30 * b
    assert count_ops(y, z) == count_ops(y, z, fast_math=False) == {'mul': 3, 'add': 1, 'sin': 1}

    _, (ry, rz) = run(y, z)
    assert ry == pytest.approx(45.0)
    assert rz == pytest.approx(0.412118, abs=1e-4)


def test_not_more_operations() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    assert count_ops((a + b) * 5) == {'add': 1, 'mul': 1}  # not 5 * a + 5 * b
    assert count_ops(a * 2 + b * 2) == {'add': 1, 'mul': 1}  # (a + b) * 2
    assert count_ops((a + b) * 5 * 2) == {'add': 1, 'mul': 1}
    assert count_ops(a * 2 + a * 3) == {'mul': 1}
    assert count_ops(-(a * 2 - b) * 4) == {'mul': 2, 'add': 1}  # a * -8 + b * 4
    assert count_ops(2 * (a + 1) + 3) == {'mul': 1, 'add': 1}  # a * 2 + 5


def test_cancelled_terms() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    y = (a + b) - a
    z = (a * 3 + 1) - 3 * a

    assert count_ops(y) == {}
    assert count_ops(z) == {}

    tg, result = run(y, z)
    assert result == [0.5, 1.0]
    tg.write_value(b, 7.0)
    tg.run()
    assert tg.read_value(y) == 7.0  # y is b


def test_inputs_are_not_folded() -> None:
    a = cp.value(2.0)
    k = cp.value(3.0)  # can be changed by write_value
    y = a * k * 5.0
    assert count_ops(y) == {'mul': 2}

    tg, result = run(y)
    assert result == pytest.approx(30.0)
    tg.write_value(k, 4.0)
    tg.run()
    assert tg.read_value(y) == pytest.approx(40.0)


def test_non_linear_operations_are_kept() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    y = (cp.sin(a * 2 * 3) * 4 + a * b * 2) * 0.5
    # sin(a * 6) * 2 + (a * b)
    assert count_ops(y) == {'mul': 3, 'sin': 1, 'add': 1}
    assert count_ops(y, fast_math=False) == {'mul': 6, 'sin': 1, 'add': 1}
    _, result = run(y)
    assert result == pytest.approx(-0.0731458, abs=1e-4)


def test_int() -> None:
    i = cp.value(7)
    y = (3 * i + 4) * 5 - i
    assert count_ops(y) == {'mul': 1, 'add': 1}  # i * 14 + 20
    q = (i * 6) // 4 * 2  # floor division is not linear
    assert count_ops(q) == {'mul': 2, 'floordiv': 1}
    wrapped = (i * 65536 + 1) * 65536  # the coefficient 2**32 is 0 in 32 bit
    assert count_ops(wrapped) == {}

    tg, result = run(y, q, wrapped)
    assert result == [118, 20, 65536]
    assert all(isinstance(r, int) for r in result)
    tg.write_value(i, -3)
    tg.run()
    assert tg.read_value([y, q, wrapped]) == [-22, -10, 65536]


def test_long_chain() -> None:
    values = [cp.value(float(i)) for i in range(3000)]
    y = values[0] * 2.0
    for v in values[1:]:
        y = y + v * 2.0
    assert count_ops(y) == {'add': 2999, 'mul': 1}


def test_given_dag_is_not_changed() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    p = cp.array([(3 * a + 6 * b) * 5, a, 1.0, b * 2 * 3])
    y = (p.sum() * 2 + a) * 3

    def snapshot(node: Node) -> dict[int, tuple[int, ...]]:
        ret: dict[int, tuple[int, ...]] = {}
        stack = [node]
        while stack:
            node = stack.pop()
            if id(node) not in ret:
                ret[id(node)] = tuple(id(net) for net in node.args)
                stack += [net.source for net in node.args]
        return ret

    roots, _ = canonicalize([Store(y.net)])
    before = [snapshot(r) for r in roots]
    folded, _ = fold_linear(roots)
    assert [snapshot(r) for r in roots] == before
    assert folded[0] is not roots[0]


def test_values_in_arrays() -> None:
    a = cp.value(2.0)
    b = cp.value(0.5)
    p = cp.array([(3 * a + 6 * b) * 5, a, 1.0, b * 2 * 3])
    y = (p.sum() * 2 + a) * 3

    tg, result = run(y, p)
    assert result[0] == pytest.approx(312.0)  # (51 * 2 + 2) * 3
    assert result[1] == pytest.approx([45.0, 2.0, 1.0, 3.0])
    assert tg.read_value(p[0]) == pytest.approx(45.0)
    assert tg.read_value(p[3]) == pytest.approx(3.0)

    tg.write_value(a, 1.0)
    tg.run()
    assert tg.read_value(y) == pytest.approx(213.0)  # (35 * 2 + 1) * 3
    assert tg.read_value(p) == pytest.approx([30.0, 1.0, 1.0, 3.0])


def test_same_results_as_without_folding() -> None:
    def build(x: Any, k: Any) -> list[Any]:
        v = cp.vector([x * 2, x + 1, 3.0, k * x]) * 4
        w = (v + v * 0.5 - 1.5) * 2
        return [w.dot(w) * 0.25 - x * 3, (w[0] - w[1] * 2 + k * 5) * 3, (k * 3 + 2) * (k * 3 + 2) * 2]

    results = []
    for fast_math in (False, True):
        x = cp.value(1.25)
        k = cp.value(3)
        out = build(x, k)
        tg = cp.Target(fast_math=fast_math)
        tg.compile(out)
        tg.run()
        results.append(tg.read_value(out))
    assert results[1] == pytest.approx(results[0], rel=1e-5)
