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
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}\n -Return code: {result.returncode}"
    return result.stdout


@pytest.mark.runner
def test_compile():

    test_vals = [0.0, -1.5, -2.0, -2.5, -3.0]

    # Function with no passing-on-jump as last instruction:
    ret_test = [r for v in test_vals for r in (cp.tan(variable(v)),)]

    out = [Write(r) for r in ret_test]

    il, variables = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)
    #il.write_com(_binwrite.Command.DUMP_CODE)

    for net in ret_test:
        assert isinstance(net, copapy.backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    #il.print()

    il.to_file('build/runner/test.copapy')

    result = run_command(['build/runner/coparun', 'build/runner/test.copapy', 'build/runner/test.copapy.bin'])
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result, 'No Return value: 1'
    #assert 'END_COM' in result


if __name__ == "__main__":
    #test_example()
    test_compile()
