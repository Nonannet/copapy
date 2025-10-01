from copapy import Write, const
import copapy
import subprocess
import struct
from copapy import binwrite as binw


def run_command(command: list[str], encoding: str = 'utf8') -> str:
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    assert error is None, f"Error occurred: {error.decode(encoding)}"
    return output.decode(encoding)


def test_example():
    c1 = 1.11
    c2 = 2.22

    i1 = c1 * 2
    i2 = i1 + 3

    r1 = i1 + i2
    r2 = c2 + 4 + c1

    en = {'little': '<', 'big': '>'}['little']
    data = struct.pack(en + 'f', r1)
    print("example r1 " + ' '.join(f'{b:02X}' for b in data))

    data = struct.pack(en + 'f', r2)
    print("example r2 " + ' '.join(f'{b:02X}' for b in data))

    # assert False
    # example r1 7B 14 EE 40
    # example r2 5C 8F EA 40


def test_compile():

    print(run_command(['bash', 'build.sh']))

    c1 = const(4)
    #c2 = const(2)

    #i1 = c1 * 2
    #i2 = i1 + 3

    #r1 = i1 + i2
    #r2 = c2 + 4 + c1

    #out = [Write(r1), Write(r2)]

    i1 = c1 * 2
    r1 = i1 + 7
    out = Write(r1)

    il = copapy.compile_to_instruction_list(out)

    #copapy.read_variable(il, i1)
    copapy.read_variable(il, r1)

    il.write_com(binw.Command.READ_DATA)
    il.write_int(0)
    il.write_int(36)

    # run program command
    il.write_com(binw.Command.END_PROG)

    print('#', il.print())

    il.to_file('test.copapy')

    result = run_command(['./bin/runmem2', 'test.copapy'])
    print(result)

    assert 'Return value: 1' in result


if __name__ == "__main__":
    test_compile()
