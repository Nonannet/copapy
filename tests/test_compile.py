from copapy import NumLike
from copapy.backend import Store, compile_to_dag, add_read_command
import copapy as cp
import subprocess
import struct
from copapy import _binwrite
import copapy.backend
import pytest


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


def test_example():
    c1 = 4
    c2 = 2

    i1 = c1 * 2
    r1 = i1 + 7 + (c2 + 7 * 9)
    r2 = i1 + 9

    en = {'little': '<', 'big': '>'}['little']
    data = struct.pack(en + 'i', r1)
    print("example r1 " + ' '.join(f'{b:02X}' for b in data))

    data = struct.pack(en + 'i', r2)
    print("example r2 " + ' '.join(f'{b:02X}' for b in data))


def function(c1: NumLike, c2: NumLike) -> tuple[NumLike, ...]:
    i1 = c1 // 3.3 + 5
    i2 = c2 * 5 + c1
    r1 = i1 + i2 * 55 / 4
    r2 = 4 * i2 + 5

    return i1, i2, r1, r2


@pytest.mark.runner
def test_compile():

    #c1 = variable(4)
    #c2 = variable(2)

    #ret = function(c1, c2)
    #ret = [c1 // 3.3 + 5]

    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.value(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.value(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    ret = (t2, t4, t5)

    out = [Store(r) for r in ret]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for v in ret:
        assert isinstance(v, cp.value)
        add_read_command(il, variables, v.net)

    il.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #il.print()

    il.to_file('build/runner/test.copapy')

    result = run_command(['build/runner/coparun', 'build/runner/test.copapy'])
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result
    #assert 'END_COM' in result


if __name__ == "__main__":
    #test_example()
    test_compile()
