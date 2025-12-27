from copapy import NumLike, iif, value
from copapy.backend import Write, compile_to_dag, add_read_command
import subprocess
from copapy import _binwrite
import copapy.backend as backend
import warnings
import re
import struct
import platform
import copapy as cp
import pytest


def parse_results(log_text: str) -> dict[int, bytes]:
    regex = r"^READ_DATA offs=(\d*) size=(\d*) data=(.*)$"
    matches = re.finditer(regex, log_text, re.MULTILINE)
    var_dict: dict[int, bytes] = {}

    for match in matches:
        value_str: list[str] = match.group(3).strip().split(' ')
        #print('--', value_str)
        value = bytes(int(v, base=16) for v in value_str)
        if len(value) <= 8:
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


@pytest.mark.runner
def test_compile():
    #t1 = cp.vector([10, 11, 12]) + cp.vector(cp.variable(v) for v in range(3))
    #t2 = t1.sum()

    #t3 = cp.vector(cp.variable(1 / (v + 1)) for v in range(3))
    #t4 = ((t3 * t1) * 2).sum()
    #t5 = ((t3 * t1) * 2).magnitude()

    c_i = value(9)
    c_f = value(1.111)
    c_b = value(True)

    ret_test = function1(c_i) + function1(c_f) + function2(c_i) + function2(c_f) + function3(c_i) + function4(c_i) + function5(c_b) + [c_i % 2, cp.sin(c_f)] + iiftests(c_i) + iiftests(c_f)
    ret_ref = function1(9) + function1(1.111) + function2(9) + function2(1.111) + function3(9) + function4(9) + function5(True) + [9 % 2, cp.sin(1.111)] + iiftests(9) + iiftests(1.111)

    #ret_test = [cp.sin(c_i), cp.asin(variable(0.0))]
    #ret_ref = [cp.sin(9), cp.asin(0.0)]

    #ret_test: list[value[float]] = []
    #ret_ref: list[float] = []
    #sval = variable(8.0)
    #tval = 8.0
    #for i in range(20):
    #    tval = (10-i)  / 12
    #    sval = cp.asin(variable(tval))
    #    tval = cp.asin(tval)
    #    ret_test.append(sval)
    #    ret_ref.append(tval)

    #ret_test = [cp.sin(c_i)]
    #ret_ref = [cp.sin(9)]

    #ret_test = [cp.get_42(c_i)]
    #ret_ref = [cp.get_42(9)]

    out = [Write(r) for r in ret_test]

    #ret_test += [c_i, v2]
    #ret_ref += [9, 4.44, -4.44]

    sdb = backend.stencil_db_from_package('x86')
    dw, variables = compile_to_dag(out, sdb)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    # run program command
    dw.write_com(_binwrite.Command.RUN_PROG)
    #dw.write_com(_binwrite.Command.DUMP_CODE)

    for v in ret_test:
        assert isinstance(v, value)
        add_read_command(dw, variables, v.net)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    dw.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #dw.print()

    dw.to_file('build/runner/test-x86.copapy')

    if platform.machine() != 'AMD64' and platform.machine() != 'x86_64':
        warnings.warn(f"Test skipped, {platform.machine()} not supported for this test.", UserWarning)
    else:
        command = ['build/runner/coparun-x86', 'build/runner/test-x86.copapy', 'build/runner/test-x86.copapy.bin']

        try:
            result = run_command(command)
        except FileNotFoundError:
            warnings.warn("Test skipped, executable not found.", UserWarning)
            return

        print('* Output from runner:\n--')
        print(result)
        print('--')

        assert 'Return value: 1' in result

        result_data = parse_results(result)

        for test, ref in zip(ret_test, ret_ref):
            assert isinstance(test, value)
            address = variables[test.net][0]
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
            for t in (int, float, bool):
                assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
            assert val == pytest.approx(ref, 1e-5), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.runner
def test_vector_compile():
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.value(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.value(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    ret = (t2, t4, t5)

    out = [Write(r) for r in ret]

    sdb = backend.stencil_db_from_package('x86')
    il, variables = compile_to_dag(out, sdb)

    # run program command
    il.write_com(_binwrite.Command.RUN_PROG)
    #il.write_com(_binwrite.Command.DUMP_CODE)

    for v in ret:
        assert isinstance(v, cp.value)
        add_read_command(il, variables, v.net)

    il.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #il.print()

    il.to_file('build/runner/test-x86.copapy')

    if platform.machine() != 'AMD64' and platform.machine() != 'x86_64':
        warnings.warn(f"Test skipped, {platform.machine()} not supported for this test.", UserWarning)
    else:
        command = ['build/runner/coparun-x86', 'build/runner/test-x86.copapy', 'build/runner/test-x86.copapy.bin']
        try:
            result = run_command(command)
        except FileNotFoundError:
            warnings.warn("Test skipped, executable not found.", UserWarning)
            return

        print('* Output from runner:\n--')
        print(result)
        print('--')

        assert 'Return value: 1' in result

        # Compare to x86_64 reference results
        assert " size=4 data=24 00 00 00" in result
        assert " size=4 data=56 55 25 42" in result
        assert " size=4 data=B4 F9 C8 41" in result


@pytest.mark.runner
def test_sinus():
    a_val = 1.25  # TODO: Error on x86: a > 2 PI --> Sin result > 1

    a = cp.value(a_val)
    b = cp.value(0.87)

    # Define computations
    c = a + b * 2.0
    si = cp.sin(a)
    d = c ** 2 + si
    e = d + cp.sqrt(b)

    ret_test = [si, e]
    ret_ref = [cp.sin(a_val), (a_val + 0.87 * 2.0) ** 2 + cp.sin(a_val) + cp.sqrt(0.87)]

    out = [Write(r) for r in ret_test]

    sdb = backend.stencil_db_from_package('x86')
    dw, variables = compile_to_dag(out, sdb)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    # run program command
    dw.write_com(_binwrite.Command.RUN_PROG)
    #dw.write_com(_binwrite.Command.DUMP_CODE)

    for v in ret_test:
        assert isinstance(v, value)
        add_read_command(dw, variables, v.net)

    #dw.write_com(_binwrite.Command.READ_DATA)
    #dw.write_int(0)
    #dw.write_int(28)

    dw.write_com(_binwrite.Command.END_COM)

    #print('* Data to runner:')
    #dw.print()

    dw.to_file('build/runner/test-x86.copapy')

    if platform.machine() != 'AMD64' and platform.machine() != 'x86_64':
        warnings.warn(f"Test skipped, {platform.machine()} not supported for this test.", UserWarning)
    else:
        command = ['build/runner/coparun-x86', 'build/runner/test-x86.copapy', 'build/runner/test-x86.copapy.bin']

        try:
            result = run_command(command)
        except FileNotFoundError:
            warnings.warn("Test skipped, executable not found.", UserWarning)
            return

        print('* Output from runner:\n--')
        print(result)
        print('--')

        assert 'Return value: 1' in result

        result_data = parse_results(result)

        for test, ref in zip(ret_test, ret_ref):
            assert isinstance(test, value)
            address = variables[test.net][0]
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
            for t in (int, float, bool):
                assert isinstance(val, t) == isinstance(ref, t), f"Result type does not match for {val} and {ref}"
                assert val == pytest.approx(ref, 1e-7), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    #test_example()
    test_compile()
