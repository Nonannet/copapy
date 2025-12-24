import copapy as cp
from copapy import value
from copapy.backend import get_dag_stats, Write
import copapy.backend as cpb
from typing import Any


def show_dag(val: value[Any]):
    out = [Write(val.net)]

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
    output_ops = list(cpb.add_read_ops(ordered_ops))
    for p in output_ops:
        print('#', p)

    print('-- add_write_ops:')
    extended_output_ops = list(cpb.add_write_ops(output_ops, const_list))
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


if __name__ == "__main__":
    test_get_dag_stats()
    test_dag_reduction()