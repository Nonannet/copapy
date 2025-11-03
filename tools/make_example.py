from copapy import variable
from copapy.backend import Write, compile_to_dag, stencil_db_from_package
import copapy as cp
from copapy._binwrite import Command


def compile_to_x86_64() -> None:
    """Test compilation of a simple program for x86_64."""
    c1 = variable(9.0)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    ret = [c1 // 3.3 + 5]
    #ret = [cp.sqrt(c1)]
    #c2 = cp._math.get_42()
    #ret = [c2]

    out = [Write(r) for r in ret]

    sdb = stencil_db_from_package('x86_64')
    dw, _ = compile_to_dag(out, sdb)

    dw.write_com(Command.DUMP_CODE)

    print('* Data to runner:')
    dw.print()

    dw.to_file('bin/test.copapy')


def compile_to_aarch64() -> None:
    """Test compilation of a simple program for aarch64."""
    c1 = variable(9.0)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    #ret = [cp.sin(c1), cp.sqrt(c1) + 5]
    ret = [c1 // 3.3 + 5]
    #c2 = cp._math.get_42()
    #ret = [c2]

    out = [Write(r) for r in ret]

    sdb = stencil_db_from_package('aarch64')
    dw, _ = compile_to_dag(out, sdb)

    dw.write_com(Command.DUMP_CODE)

    print('* Data to runner:')
    dw.print()

    dw.to_file('bin/test-aarch64.copapy')


if __name__ == "__main__":
    compile_to_x86_64()
    compile_to_aarch64()
