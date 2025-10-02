from copapy import Write, const, Node
import copapy as rc
from typing import Iterable, Generator


def test_ast_generation():
    #c1 = const(1.11)
    #c2 = const(2.22)
    #c3 = const(3.33)

    #i1 = c1 + c2
    #i2 = c2 * i1
    #i3 = i2 + 4

    #r1 = i1 + i3
    #r2 = i3 * i2
    c1 = const(4)
    i1 = c1 * 2
    r1 = i1 + 7
    r2 = i1 + 9
    out = [Write(r1), Write(r2)]

    print(out)
    print('-- get_edges:')

    edges = list(rc.get_all_dag_edges(out))
    for p in edges:
        print('#', p)

    print('-- get_ordered_ops:')
    ordered_ops = list(rc.stable_toposort(edges))
    for p in ordered_ops:
        print('#', p)

    print('-- get_consts:')
    const_list = rc.get_consts(ordered_ops)
    for p in const_list:
        print('#', p)

    print('-- add_read_ops:')
    output_ops = list(rc.add_read_ops(ordered_ops))
    for p in output_ops:
        print('#', p)

    print('-- add_write_ops:')
    extended_output_ops = list(rc.add_write_ops(output_ops, const_list))
    for p in extended_output_ops:
        print('#', p)
    print('--')


if __name__ == "__main__":
    test_ast_generation()
