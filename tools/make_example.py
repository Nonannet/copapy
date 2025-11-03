from copapy import variable
from copapy.backend import Write, compile_to_dag, stencil_db_from_package
import subprocess
import copapy as cp
from copapy._binwrite import Command
import os


if os.name == 'nt':
    # On Windows wsl and qemu-user is required:
    # sudo apt install qemu-user
    qemu_command = ['wsl', 'qemu-aarch64',
                    'bin/coparun-aarch64',
                    'bin/test-aarch64.copapy',
                    'bin/test-aarch64.copapy.bin']
else:
    qemu_command = ['qemu-aarch64',
                    'bin/coparun-aarch64',
                    'bin/test-aarch64.copapy',
                    'bin/test-aarch64.copapy.bin']


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


def test_compile() -> None:
    """Test compilation of a simple program."""
    c1 = variable(9.0)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    ret = [c1 // 3.3 + 5]
    #ret = [cp.sqrt(c1)]
    #c2 = cp._math.get_42()
    #ret = [c2]

    out = [Write(r) for r in ret]

    dw, vars = compile_to_dag(out, cp.generic_sdb)

    # run program command
    dw.write_com(Command.RUN_PROG)

    # read first 32 byte
    dw.write_com(Command.READ_DATA)
    dw.write_int(0)
    dw.write_int(32)

    # read variables
    for addr, lengths, _ in vars.values():
        dw.write_com(Command.READ_DATA)
        dw.write_int(addr)
        dw.write_int(lengths)

    dw.write_com(Command.END_COM)

    print('* Data to runner:')
    dw.print()

    dw.to_file('bin/test.copapy')


def test_compile_aarch64() -> None:
    """Test compilation of a simple program."""
    c1 = variable(9.0)

    #ret = [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4]
    #ret = [cp.sin(c1), cp.sqrt(c1) + 5]
    ret = [cp.sqrt(c1)]
    #c2 = cp._math.get_42()
    #ret = [c2]

    out = [Write(r) for r in ret]

    sdb = stencil_db_from_package('aarch64')
    dw, vars = compile_to_dag(out, sdb)

    # run program command
    #dw.write_com(Command.RUN_PROG)

    # read first 32 byte
    #dw.write_com(Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(32)

    # read variables
    #for addr, lengths, _ in vars.values():
    #    dw.write_com(Command.READ_DATA)
    #    dw.write_int(addr)
    #    dw.write_int(lengths)

    #dw.write_com(Command.END_COM)
    dw.write_com(Command.DUMP_CODE)

    print('* Data to runner:')
    dw.print()

    dw.to_file('bin/test-aarch64.copapy')

    result = run_command(qemu_command)
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'DUMP_CODE' in result


if __name__ == "__main__":
    #test_compile()
    test_compile_aarch64()
