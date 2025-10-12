from copapy import Write, CPVariable
import copapy
import subprocess
import struct
from copapy import binwrite


def run_command(command: list[str], encoding: str = 'utf8') -> str:
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    assert error is None, f"Error occurred: {error.decode(encoding)}"
    return output.decode(encoding)


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

    # assert False
    #example r1 42 A0 00 00
    #example r2 41 88 00 00



def function(c1, c2):
    i1 = c1 * 3.3 + 5
    i2 = c2 * 5 + c1
    #r1 = i1 + i2 * 55 / 4
    r1 = i1 + i2 * 55 / 4
    r2 = 4 * i2 + 5

    return i1, i2, r1, r2


def test_compile():

    c1 = CPVariable(4)
    c2 = CPVariable(2)

    ret = function(c1, c2)

    out = [Write(r) for r in ret]

    il, _ = copapy.compile_to_instruction_list(out, copapy.generic_sdb)

    # run program command
    il.write_com(binwrite.Command.RUN_PROG)
    il.write_int(0)

    il.write_com(binwrite.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    il.write_com(binwrite.Command.END_COM)

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
