from copapy import _binwrite, variable
from copapy.backend import Write, compile_to_instruction_list
import copapy as cp


def test_compile() -> None:

    c1 = variable(9.0)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    #ret = [c1 // 3.3 + 5]
    ret = [cp.sqrt(c1)]

    out = [Write(r) for r in ret]

    il, _ = compile_to_instruction_list(out, cp.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    il.write_com(_binwrite.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')


if __name__ == "__main__":
    test_compile()
