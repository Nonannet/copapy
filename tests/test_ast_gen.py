from copapy import variable
from copapy.backend import Write
import copapy.backend as cpbe


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
    #out = [Write(r1), Write(r2)]

    c1 = variable(4)
    c2 = variable(2)
    #i1 = c1 * 2
    #r1 = i1 + 7 + (c2 + 7 * 9)
    #r2 = i1 + 9
    #out = [Write(r1), Write(r2)]
    r1 = c1 * 5 + 8 + c2 * 3
    out = [Write(r1)]

    print(out)
    print('-- get_edges:')

    edges = list(cpbe.get_all_dag_edges(out))
    for p in edges:
        print('#', p)

    print('-- get_ordered_ops:')
    ordered_ops = list(cpbe.stable_toposort(edges))
    for p in ordered_ops:
        print('#', p)

    print('-- get_consts:')
    const_list = cpbe.get_const_nets(ordered_ops)
    for p in const_list:
        print('#', p)

    print('-- add_read_ops:')
    output_ops = list(cpbe.add_read_ops(ordered_ops))
    for p in output_ops:
        print('#', p)

    print('-- add_write_ops:')
    extended_output_ops = list(cpbe.add_write_ops(output_ops, const_list))
    for p in extended_output_ops:
        print('#', p)
    print('--')


if __name__ == "__main__":
    test_ast_generation()
