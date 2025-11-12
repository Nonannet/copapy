from copapy import variable
from copapy.backend import Write, compile_to_dag, add_read_command
import copapy as cp
import subprocess
from copapy import _binwrite
import copapy.backend
import pytest


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


@pytest.mark.runner
def test_compile_sqrt():
    test_vals = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0]

    test_vals = [100000.0]

    ret = [r for v in test_vals for r in (cp.sqrt(variable(v)),)]


    out = [Write(r) for r in ret]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in ret:
        assert isinstance(net, copapy.backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')

    result = run_command(['bin/coparun', 'bin/test.copapy'])
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result
    #assert 'END_COM' in result


@pytest.mark.runner
def test_compile_log():
    test_vals = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0]

    test_vals = [100000.0]

    ret = [r for v in test_vals for r in (cp.log(variable(v)),)]


    out = [Write(r) for r in ret]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in ret:
        assert isinstance(net, copapy.backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')

    result = run_command(['bin/coparun', 'bin/test.copapy'])
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result
    #assert 'END_COM' in result


@pytest.mark.runner
def test_compile_sin():
    test_vals = [0.0, 0.0001, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.28318530718, 100.0, 1000.0, 100000.0]

    test_vals = [100000.0]

    ret = [r for v in test_vals for r in (cp.sin(variable(v)),)]


    out = [Write(r) for r in ret]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in ret:
        assert isinstance(net, copapy.backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')

    result = run_command(['bin/coparun', 'bin/test.copapy'])
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result
    #assert 'END_COM' in result


if __name__ == "__main__":
    #test_compile_sqrt()
    #test_compile_log()
    test_compile_sin()
