import copapy as cp
from copapy import value
from copapy.backend import get_dag_stats, Store
import copapy.backend as cpb
import pytest
from typing import Any


def show_dag(val: value[Any]):
    out = [Store(val.net)]

    print(out)
    print('-- get_edges:')

    edges = list(cpb.get_all_dag_edges(out))
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


def test_get_dag_stats():

    sum_size = 10
    v_size = 200

    v1 = cp.vector(cp.value(float(v)) for v in range(v_size))
    v2 = cp.vector(cp.value(float(v)) for v in [5]*v_size)

    v3 = sum((v1 + i + 7) @ v2 for i in range(sum_size))

    assert isinstance(v3, value)
    stat = get_dag_stats([v3.net])
    print(stat)

    assert stat['const_float'] == 2 * v_size
    assert stat['add_float_float'] == sum_size * v_size - 2


def test_dag_reduction():

    a = value(8)

    v3 = (a * 3 + 7 + 2) + (a * 3 + 7 + 2)

    show_dag(v3)

    assert isinstance(v3, value)
    stat = get_dag_stats([v3.net])
    print(stat)


def emitted_ops(val: value[Any]) -> list[str]:
    end_nodes = [Store(val.net)]
    ordered_ops = cpb.stable_toposort(cpb.get_all_dag_edges(end_nodes))
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


if __name__ == "__main__":
    test_get_dag_stats()
    test_dag_reduction()
    test_square_stencil()
    test_square_stencil_result()
