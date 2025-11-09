from copapy import NumLike
from copapy.backend import Write, compile_to_dag, add_read_command
import subprocess
from copapy import _binwrite
import copapy.backend as backend
import copapy as cp
import os
import warnings
import pytest

if os.name == 'nt':
    # On Windows wsl and qemu-user is required:
    # sudo apt install qemu-user
    qemu_command = ['wsl', 'qemu-aarch64']
else:
    qemu_command = ['qemu-aarch64']


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


def check_for_qemu() -> bool:
    command = qemu_command + ['--version']
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    except Exception:
        return False
    return result.returncode == 0


def function(c1: NumLike, c2: NumLike) -> tuple[NumLike, ...]:
    i1 = c1 // 3.3 + 5
    i2 = c2 * 5 + c1
    r1 = i1 + i2 * 55 / 4
    r2 = 4 * i2 + 5

    return i1, i2, r1, r2


@pytest.mark.runner
def test_compile():
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.variable(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.variable(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    ret = (t2, t4, t5)

    out = [Write(r) for r in ret]

    sdb = backend.stencil_db_from_package('aarch64')
    il, variables = compile_to_dag(out, sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in ret:
        assert isinstance(net, backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test-aarch64.copapy')

    if not check_for_qemu():
        warnings.warn("qemu-aarch64 not found, aarch64 test skipped!", UserWarning)
    else:
        command = ['bin/coparun-aarch64', 'bin/test-aarch64.copapy']
        result = run_command(qemu_command + command)
        print('* Output from runner:\n--')
        print(result)
        print('--')

        assert 'Return value: 1' in result

        # Compare to x86_64 reference results
        assert " size=4 data=24 00 00 00" in result
        assert " size=4 data=56 55 25 42" in result
        assert " size=4 data=B4 F9 C8 41" in result


if __name__ == "__main__":
    #test_example()
    test_compile()
