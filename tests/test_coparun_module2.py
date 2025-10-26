from coparun_module import coparun
from copapy import variable
from copapy.backend import Write, compile_to_dag, add_read_command
import copapy
from copapy import _binwrite


def test_compile():

    c1 = variable(4)
    c2 = variable(2) * 4

    i1 = c2 * 2
    r1 = i1 + 7 + (c1 + 7 * 9)
    r2 = i1 + 9
    out = [Write(r1), Write(r2), Write(c2)]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in (c1, c2, i1, r1, r2):
        add_read_command(il, variables, net)

    # run program command
    il.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #il.print()

    print('+ run coparun')
    result = coparun(il.get_data())

    assert result == 1


if __name__ == "__main__":
    test_compile()
