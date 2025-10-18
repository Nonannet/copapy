from copapy import NumLike, Write, cpvalue, Net
import copapy
import subprocess
from copapy import binwrite


def run_command(command: list[str], encoding: str = 'utf8') -> str:
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    assert error is None, f"Error occurred: {error.decode(encoding)}"
    return output.decode(encoding)


def function(c1: NumLike, c2: NumLike) -> tuple[NumLike, ...]:
    i1 = c1 * 3.3 + 5
    i2 = c2 * 5 + c1
    r1 = i1 + i2 * 55 / 4
    r2 = 4 * i2 + 5

    return i1, i2, r1, r2


def test_compile():

    c1 = cpvalue(4)
    c2 = cpvalue(2)

    ret = function(c1, c2)

    dw, variable_list = copapy.compile_to_instruction_list([Write(net) for net in ret], copapy.generic_sdb)

    # run program command
    dw.write_com(binwrite.Command.RUN_PROG)

    dw.write_com(binwrite.Command.READ_DATA)
    dw.write_int(0)
    dw.write_int(36)

    for net, name in zip(ret, ['i1', 'i2', 'r1', 'r2']):
        print('+', name)
        assert isinstance(net, Net)
        copapy.add_read_command(dw, variable_list, net)

    dw.write_com(binwrite.Command.END_COM)

    dw.to_file('bin/test.copapy')
    result = run_command(['bin/coparun', 'bin/test.copapy'])
    print('* Output from runner:')
    print(result)
    assert 'Return value: 1' in result


if __name__ == "__main__":
    #test_example()
    test_compile()
