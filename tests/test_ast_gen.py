from copapy import Write, const
import copapy as rc

def test_ast_generation():
    c1 = const(1.11)
    c2 = const(2.22)
    #c3 = const(3.33)

    #i1 = c1 + c2
    #i2 = c2 * i1
    #i3 = i2 + 4

    #r1 = i1 + i3
    #r2 = i3 * i2

    i1 = c1 * 2
    i2 = i1 + 3

    r1 = i1 + i2
    r2 = c2 + 4 + c1

    out = [Write(r1), Write(r2)]
    print(out)
    print('--')

    path_segments = list(rc.get_path_segments(out))
    for p in path_segments:
        print(p)
    print('--')

    ordered_ops = list(rc.get_ordered_ops(path_segments))
    for p in path_segments:
        print(p)
    print('--')

    const_list = rc.get_consts(ordered_ops)
    for p in const_list:
        print(p)
    print('--')

    output_ops = list(rc.add_read_ops(ordered_ops))
    for p in output_ops:
        print(p)
    print('--')

    extended_output_ops = list(rc.add_write_ops(output_ops, const_list))
    for p in extended_output_ops:
        print(p)
    print('--')


if __name__ == "__main__":
    test_ast_generation()