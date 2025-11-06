from copapy import NumLike, iif, variable
from copapy.backend import Write, compile_to_dag, add_read_command
import subprocess
from copapy import _binwrite
import copapy.backend as backend
import copapy as cp
import os
import warnings
import re
import struct
import pytest

def parse_results(log_text: str) -> dict[int, bytes]:
    regex = r"^READ_DATA offs=(\d*) size=(\d*) data=(.*)$"
    matches = re.finditer(regex, log_text, re.MULTILINE)
    var_dict: dict[int, bytes] = {}

    for match in matches:
        value_str: list[str] = match.group(3).strip().split(' ')
        print('--', value_str)
        value = bytes(int(v, base=16) for v in value_str)
        var_dict[int(match.group(1))] = value

    return var_dict


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    return result.stdout


def function1(c1: NumLike) -> list[NumLike]:
    return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
            c1 * 4, c1 * -4,
            c1 + 4, c1 - 4,
            c1 > 2, c1 > 100, c1 < 4, c1 < 100]


def function2(c1: NumLike) -> list[NumLike]:
    return [c1 / 4.44, c1 / -4.44, c1 // 4.44, c1 // -4.44, (c1 * -1) // 4.44,
            c1 * 4.44, c1 * -4.44,
            c1 + 4.44, c1 - 4.44,
            c1 > 2, c1 > 100.11, c1 < 4.44, c1 < 100.11]


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
    c_i = variable(9)
    c_f = variable(1.111)
    c_b = variable(True)

    ret_test = function1(c_i) + function1(c_f) + function2(c_i) + function2(c_f) + function3(c_i) + function4(c_i) + function5(c_b) + [variable(9) % 2] + iiftests(c_i) + iiftests(c_f)
    ret_ref = function1(9) + function1(1.111) + function2(9) + function2(1.111) + function3(9) + function4(9) + function5(True) + [9 % 2] + iiftests(9) + iiftests(1.111)

    out = [Write(r) for r in ret_test]

    sdb = backend.stencil_db_from_package('native')
    il, variables = compile_to_dag(out, sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)

    for net in ret_test:
        assert isinstance(net, backend.Net)
        add_read_command(il, variables, net)

    il.write_com(_binwrite.Command.END_COM)

    print('* Data to runner:')
    il.print()

    il.to_file('bin/test.copapy')

    command = ['bin/coparun', 'bin/test.copapy']
    result = run_command(command)
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
        print('+', val, ref, test.dtype)
        for t in (int, float, bool):
            assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
        assert val == pytest.approx(ref, 1e-5), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    #test_example()
    test_compile()
