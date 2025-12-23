from copapy import value, NumLike
from copapy.backend import Write, compile_to_dag, add_read_command, Net
import copapy
import subprocess
from copapy import _binwrite
import pytest


def run_command(command: list[str], encoding: str = 'utf8') -> str:
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    assert error is None, f"Error occurred: {error.decode(encoding)}"
    return output.decode(encoding)


def function(c1: NumLike) -> list[NumLike]:
    r1 = c1 / 2
    return [r1]


@pytest.mark.runner
def test_compile():

    c1 = value(16)

    ret = function(c1)

    out = [Write(r) for r in ret]

    il, vars = compile_to_dag(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for v in ret:
        assert isinstance(v, value)
        add_read_command(il, vars, v.net)

    il.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #il.print()

    il.to_file('build/runner/test.copapy')

    result = run_command(['build/runner/coparun', 'build/runner/test.copapy'])
    print('* Output from runner:')
    print(result)

    assert 'Return value: 1' in result


if __name__ == "__main__":
    #test_example()
    test_compile()
