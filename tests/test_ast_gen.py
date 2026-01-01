from copapy import value
from copapy.backend import Store
import copapy.backend as cpb


def test_ast_generation():
    #c1 = const(1.11)
    #c2 = const(2.22)
    #c3 = const(3.33)

    #i1 = c1 + c2
    #i2 = c2 * i1
    #i3 = i2 + 4

    #r1 = i1 + i3
    #r2 = i3 * i2

    #c1 = const(4)
    #i1 = c1 * 2
    #r1 = i1 + 7
    #r2 = i1 + 9
    #out = [Store(r1), Store(r2)]

    c1 = value(4)
    c2 = value(2)
    #i1 = c1 * 2
    #r1 = i1 + 7 + (c2 + 7 * 9)
    #r2 = i1 + 9
    #out = [Store(r1), Store(r2)]
    r1 = c1 * 5 + 8 + c2 * 3
    out = [Store(r1)]

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


if __name__ == "__main__":
    test_ast_generation()
