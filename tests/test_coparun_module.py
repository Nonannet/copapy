from coparun_module import coparun
from copapy import Write, const
import copapy
from copapy import binwrite


def test_compile():

    c1 = const(4)
    c2 = const(2)

    i1 = c1 * 2
    r1 = i1 + 7 + (c2 + 7 * 9)
    r2 = i1 + 9
    out = [Write(r1), Write(r2)]

    il = copapy.compile_to_instruction_list(out)

    copapy.read_variable(il, r1)
    copapy.read_variable(il, r2)

    il.write_com(binwrite.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    # run program command
    il.write_com(binwrite.Command.END_PROG)

    #print('* Data to runner:')
    #il.print()

    print('+ run coparun')
    result = coparun(il.get_data())

    assert result == 1


if __name__ == "__main__":
    test_compile()
