import copapy as cp
from copapy import value
from copapy.backend import get_dag_stats, Store
import copapy.backend as cpb
import math
import pytest
from typing import Any
from runner_helpers import trig_tol


def show_dag(val: value[Any]):
    out = [Store(val.net)]

    print(out)
    print('-- get_edges:')

    edges = list(cpb.get_all_dag_edges(cpb.canonicalize(out)[0]))
    for p in edges:
        print('#', p)

    print('-- get_ordered_ops:')
    ordered_ops = cpb.stable_toposort(edges)
    for p in ordered_ops:
        print('#', p)

    print('-- get_consts:')
    const_list = cpb.get_const_nets(ordered_ops)
    for p in const_list:
        print('#', p)

    print('-- add_read_ops:')
    output_ops = list(cpb.add_load_ops(ordered_ops))
    for p in output_ops:
        print('#', p)

    print('-- add_write_ops:')
    extended_output_ops = list(cpb.add_store_ops(output_ops, const_list))
    for p in extended_output_ops:
        print('#', p)
    print('--')


def test_get_dag_stats(monkeypatch: pytest.MonkeyPatch):
    # Scalar DAG optimization: vectors are not packed into arrays
    monkeypatch.setattr(cp.vector, 'pack_threshold', None)

    sum_size = 10
    v_size = 200

    v1 = cp.vector(cp.value(float(v)) for v in range(v_size))
    v2 = cp.vector(cp.value(float(v)) for v in [5]*v_size)

    v3 = sum((v1 + i + 7) @ v2 for i in range(sum_size))

    assert isinstance(v3, value)
    stat = get_dag_stats([v3.net])
    print(stat)

    # The int constants i and 7 are converted to float at trace time (7.0 is
    # merged with i = 7), both additions per element and the sum are float additions.
    # Adding i = 0 generates no code: v1 + 0 + 7 is merged with v1 + 7 for i = 7
    assert stat['const_float'] == 2 * v_size + sum_size - 1
    assert 'float_int' not in stat
    assert stat['add_float_float'] == 3 * sum_size * v_size - 2 - 2 * v_size


def test_dag_reduction():

    a = value(8)

    v3 = (a * 3 + 7 + 2) + (a * 3 + 7 + 2)

    show_dag(v3)

    assert isinstance(v3, value)
    stat = get_dag_stats([v3.net])
    print(stat)


def dag_nodes(roots: list[cpb.Node]) -> list[cpb.Node]:
    return cpb.stable_toposort(cpb.get_all_dag_edges(roots))


def test_canonicalize_merges_equivalent_nodes():
    a = value(8)
    b = value(3)
    v1 = (a * 3 + b) * 2.5
    v2 = (b + a * 3) * 2.5  # commutative add with swapped arguments
    roots = [Store(v1.net), Store(v2.net)]

    # Nodes have no identity beyond the object identity before the pass
    assert v1.net.source is not v2.net.source
    assert len([n for n in dag_nodes(roots) if n.name == 'mul_int_int']) == 2

    new_roots, net_map = cpb.canonicalize(roots)
    names = [n.name for n in dag_nodes(new_roots)]
    assert len(new_roots) == 1
    assert names.count('mul_int_int') == 1
    assert names.count('add_int_int') == 1
    assert names.count('const_int') == 3  # a, b and 3
    assert names.count('const_float') == 1
    assert net_map[v1.net] is net_map[v2.net]

    # The given DAG is not changed
    assert roots[1].args[0] is v2.net
    assert v2.net.source.args[0].source is not v1.net.source.args[0].source

    # Not commutative: the order of the arguments is part of the identity
    new_roots, _ = cpb.canonicalize([Store((a - b).net), Store((b - a).net)])
    assert len(new_roots) == 2


def test_canonicalize_recalculates_identities():
    """Identities are derived from the DAG on each call, so they follow changes of the DAG"""
    a = value(8)
    b = value(3)
    v1 = (a + 1) * 2
    v2 = (b + 1) * 2
    roots = [Store(v1.net), Store(v2.net)]
    assert len(cpb.canonicalize(roots)[0]) == 2

    # Replace b by a in the second expression
    add = v2.net.source.args[0].source
    add.args = (a.net, add.args[1])
    new_roots, net_map = cpb.canonicalize(roots)
    assert len(new_roots) == 1
    assert net_map[v1.net] is net_map[v2.net]

    # ... and back
    add.args = (b.net, add.args[1])
    assert len(cpb.canonicalize(roots)[0]) == 2


def test_canonicalize_constants():
    a = value(2.0)
    roots = [Store(cp.atan2(a, 0.0).net), Store(cp.atan2(a, -0.0).net),
             Store(cp.atan2(a, 1.0).net), Store(cp.atan2(a, 1.0).net)]
    new_roots, _ = cpb.canonicalize(roots)
    consts = [n.value for n in dag_nodes(new_roots) if isinstance(n, cpb.Constant)]
    assert len(new_roots) == 3
    assert sorted(str(c) for c in consts) == ['-0.0', '0.0', '1.0']  # signed zeros are not merged


def test_merged_values_are_readable():
    a = value(8.0)
    v1 = a * 3.0 + 1.0
    v2 = a * 3.0 + 1.0
    arr = cp.array([1.0, a, 3.0])

    tg = cp.Target()
    tg.compile(v1, v2, arr)
    tg.run()
    assert tg.read_value(v1) == tg.read_value(v2) == 25.0
    assert tg.read_value(arr[1]) == tg.read_value(arr[1]) == 8.0  # a new element on each access

    tg.write_value(a, 2.0)
    tg.run()
    assert tg.read_value(v2) == 7.0
    assert tg.read_value(arr[1]) == 2.0


def emitted_ops(val: value[Any]) -> list[str]:
    end_nodes = [Store(val.net)]
    ordered_ops = cpb.stable_toposort(cpb.get_all_dag_edges(cpb.canonicalize(end_nodes)[0]))
    return [node.name for _, node in cpb.add_load_ops(ordered_ops)]


def test_square_stencil():
    # x**2 and x*x use the single argument square stencil instead of a multiply
    # that would load the same value into both argument registers.
    x = value(2.5)

    for name in (emitted_ops(x ** 2), emitted_ops(x * x)):
        assert 'square_float' in name
        assert 'mul_float_float' not in name
        assert len([o for o in name if o.startswith('load_')]) == 1

    # Integer squaring uses the integer square stencil.
    assert 'square_int' in emitted_ops(value(3) ** 2)

    # x**3 = (x*x)*x: the squaring is a square, the second multiply stays a mul.
    ops_cubed = emitted_ops(x ** 3)
    assert 'square_float' in ops_cubed
    assert 'mul_float_float' in ops_cubed
    assert ops_cubed.index('square_float') < ops_cubed.index('mul_float_float')


def test_square_stencil_result():
    xs = [0.0, 1.5, -2.0, 3.7, 100.0]
    vals = [value(x) for x in xs]
    ret_test = [v for x in vals for v in (x ** 3, x ** 2, x * x)]
    ret_ref = [x ** e for x in xs for e in (3, 2, 2)]

    tg = cp.Target()
    tg.compile(ret_test)
    tg.run()

    for test, ref in zip(ret_test, ret_ref):
        assert isinstance(test, cp.value)
        assert tg.read_value(test) == pytest.approx(ref, rel=1e-4, abs=1e-4)


def run(*vals: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*vals)
    tg.run()
    return [tg.read_value(v) for v in vals]


def test_int_to_float_conversion():
    """Float-only operations and mixed types use float stencils, int arguments
    are converted by the float_int stencil, int constants at trace time"""
    x = value(2.5)
    i = value(3)
    j = value(-2)

    # Int constants are converted at trace time
    assert 'float_int' not in emitted_ops(x + 1)
    assert 'add_float_float' in emitted_ops(x + 1)
    assert 'float_int' not in emitted_ops(cp.sin(value(1.0)) * 2)

    # Int variables are converted at runtime, int-int operations stay int
    assert emitted_ops(cp.sin(i)).count('float_int') == 1
    assert 'sin_float' in emitted_ops(cp.sin(i))
    assert 'add_int_int' in emitted_ops(i + j)
    assert 'float_int' not in emitted_ops(i + j)
    assert emitted_ops(i / j).count('float_int') == 2

    # Commutative ops: the converted argument comes first (it is still in register 0)
    s = x * i
    assert isinstance(s.net.source, cpb.Op) and s.net.source.args[0].source.name == 'float_int'

    results = run(cp.sin(i), i + x, x + i, x * i, i / j, i // 2.0, x > i, i == 3.0,
                  cp.pow(i, 2), cp.atan2(i, x), cp.minimum(i, x), cp.sqrt(value(9)), i + j)
    expected = [math.sin(3), 5.5, 5.5, 7.5, -1.5, 1.0, 0, 1, 9.0, math.atan2(3, 2.5), 2.5, 3.0, 1]
    for r, e in zip(results, expected):
        assert r == pytest.approx(e, rel=trig_tol(1e-6))
    assert isinstance(results[-1], int)
    assert isinstance(results[4], float)


def test_int_to_float_conversion_errors():
    with pytest.raises(NotImplementedError, match='mod not implemented for float and int'):
        value(2.5) % value(2)


if __name__ == "__main__":
    test_get_dag_stats(pytest.MonkeyPatch())
    test_dag_reduction()
    test_square_stencil()
    test_square_stencil_result()
    test_int_to_float_conversion()
    test_int_to_float_conversion_errors()
