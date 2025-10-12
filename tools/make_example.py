from copapy import CPVariable, Target, Write, binwrite
import copapy
from pytest import approx


def test_compile() -> None:

    c1 = CPVariable(9)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    ret = [c1 / 4]

    out = [Write(r) for r in ret]

    il, _ = copapy.compile_to_instruction_list(out, copapy.generic_sdb)

    # run program command
    il.write_com(binwrite.Command.RUN_PROG)
    il.write_int(0)

    il.write_com(binwrite.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    il.write_com(binwrite.Command.END_PROG)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')


if __name__ == "__main__":
    test_compile()
