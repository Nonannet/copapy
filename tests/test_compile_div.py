from copapy import cpvalue, NumLike
from copapy.backend import Write, compile_to_instruction_list
import copapy
import subprocess
from copapy import _binwrite


def run_command(command: list[str], encoding: str = 'utf8') -> str:
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    assert error is None, f"Error occurred: {error.decode(encoding)}"
    return output.decode(encoding)


def function(c1: NumLike) -> list[NumLike]:
    r1 = c1 / 2
    return [r1]


def test_compile():

    c1 = cpvalue(16)

    ret = function(c1)

    out = [Write(r) for r in ret]

    il, _ = compile_to_instruction_list(out, copapy.generic_sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    il.write_com(_binwrite.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')

    result = run_command(['bin/coparun', 'bin/test.copapy'])
    print('* Output from runner:')
    print(result)

    assert 'Return value: 1' in result


if __name__ == "__main__":
    #test_example()
    test_compile()
