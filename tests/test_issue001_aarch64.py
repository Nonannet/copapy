from copapy import NumLike, iif, variable
from copapy.backend import Write, compile_to_dag, add_read_command
import subprocess
from copapy import _binwrite
import copapy.backend as backend
import os
import warnings
import re
import struct

if os.name == 'nt':
    # On Windows wsl and qemu-user is required:
    # sudo apt install qemu-user
    qemu_command = ['wsl', 'qemu-aarch64']
else:
    qemu_command = ['qemu-aarch64']

def parse_results(log_text: str) -> dict[int, bytes]:
    regex = r"^READ_DATA offs=(\d*) size=(\d*) data=(.*)$"
    matches = re.finditer(regex, log_text, re.MULTILINE)
    var_dict: dict[int, bytes] = {}

    for match in matches:
        value_str: list[str] = match.group(3).strip().split(' ')
        print('--', value_str)
        value = bytes(int(v, base=16) for v in value_str)
        if len(value) <= 8:
            var_dict[int(match.group(1))] = value

    return var_dict

def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


def check_for_qemu() -> bool:
    command = qemu_command + ['--version']
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    except:
        return False
    return result.returncode == 0


def function1(c1: NumLike) -> list[NumLike]:
    return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
            c1 * 4, c1 * -4,
            c1 + 4, c1 - 4,
            c1 > 2, c1 > 100, c1 < 4, c1 < 100]


def function2(c1: NumLike) -> list[NumLike]:
    return [c1 * 4.44, c1 * -4.44]


def function3(c1: NumLike) -> list[NumLike]:
    return [c1 / 4]


def function4(c1: NumLike) -> list[NumLike]:
    return [c1 == 9, c1 == 4, c1 != 9, c1 != 4]


def function5(c1: NumLike) -> list[NumLike]:
    return [c1 == True, c1 == False, c1 != True, c1 != False, c1 / 2, c1 + 2]


def function6(c1: NumLike) -> list[NumLike]:
    return [c1 == True]


def iiftests(c1: NumLike) -> list[NumLike]:
    return [iif(c1 > 5, 8, 9),
            iif(c1 < 5, 8.5, 9.5),
            iif(1 > 5, 3.3, 8.8) + c1,
            iif(1 < 5, c1 * 3.3, 8.8),
            iif(c1 < 5, c1 * 3.3, 8.8)]


def test_compile():

    a1 = 0.0
    a2 = 3
    
    c1 = variable(a1)
    c2 = variable(a2)

    ret_test = [c1 + c2, c2 + c2]
    ret_ref: list[int | float] = [a1 + a2, a2 + a2]

    #out = [Write(r) for r in ret_test]
    out = [Write(r) for r in ret_test]

    #ret_test += [c_i, v2]
    #ret_ref += [9, 4.44, -4.44]

    sdb = backend.stencil_db_from_package('aarch64')
    dw, variables = compile_to_dag(out, sdb)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    # run program command
    dw.write_com(_binwrite.Command.RUN_PROG)
    #dw.write_com(_binwrite.Command.DUMP_CODE)

    for net in ret_test:
        assert isinstance(net, backend.Net)
        add_read_command(dw, variables, net)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    dw.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    dw.print()

    dw.to_file('bin/test-aarch64.copapy')

    if not check_for_qemu():
        warnings.warn("qemu-aarch64 not found, aarch64 test skipped!", UserWarning)
    else:
        command = ['bin/coparun-aarch64', 'bin/test-aarch64.copapy'] + ['bin/test-aarch64.copapy.bin']
        result = run_command(qemu_command + command)
        print('* Output from runner:\n--')
        print(result)
        print('--')

        assert 'Return value: 1' in result

        result_data = parse_results(result)

        for test, ref in zip(ret_test, ret_ref):
            assert isinstance(test, variable)
            address = variables[test][0]
            data = result_data[address]
            if test.dtype == 'int':
                val = int.from_bytes(data, sdb.byteorder, signed=True)
            elif test.dtype == 'bool':
                val = bool.from_bytes(data, sdb.byteorder)
            elif test.dtype == 'float':
                en = {'little': '<', 'big': '>'}[sdb.byteorder]
                val = struct.unpack(en + 'f', data)[0]
                assert isinstance(val, float)
            else:
                raise Exception(f"Unknown type: {test.dtype}")
            print('+', val, ref, test.dtype, f"  addr={address}")
            #for t in (int, float, bool):
            #    assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
            #assert val == pytest.approx(ref, 1e-5), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    #test_example()
    test_compile()
