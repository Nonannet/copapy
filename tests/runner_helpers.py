"""Shared helpers for tests running compiled copapy programs with a coparun runner"""
import os
import re
import struct
import subprocess
import warnings
from typing import Any, Sequence

import pytest

import copapy as cp
import copapy.backend as backend
from copapy import NumLike, iif, value, _binwrite
from copapy._stencils import stencil_database
from copapy.backend import Store, add_read_value_remote, compile_to_dag

# Relative path with native separators, on Windows .exe is appended automatically
NATIVE_RUNNER = os.path.normpath('build/runner/coparun')


def qemu_command(qemu: str, guest_base: bool = False) -> list[str]:
    """Command prefix for running a runner of a foreign architecture with qemu-user.

    On Windows qemu-user from WSL is used (sudo apt install qemu-user).

    Arguments:
        qemu: qemu-user executable, e.g. 'qemu-arm'
        guest_base: On WSL1 qemu can not reserve the low 4 GiB guest address space
            for 32 bit guests, the guest base must be placed above it with -B
    """
    if os.name == 'nt':
        return ['wsl', qemu] + (['-B', '0x100000000'] if guest_base else [])
    return [qemu]


def check_for_qemu(qemu: Sequence[str]) -> bool:
    try:
        result = subprocess.run(list(qemu) + ['--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception:
        return False
    return result.returncode == 0


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', check=False)
    assert result.returncode != 11, f"SIGSEGV (segmentation fault)\n -Error occurred: {result.stderr}\n -Output: {result.stdout}"
    assert result.returncode == 0, f"\n -Error occurred: {result.stderr}\n -Output: {result.stdout}\n -Return code: {result.returncode}"
    return result.stdout


def parse_results(log_text: str) -> dict[int, bytes]:
    """Returns the data of all READ_DATA outputs of a runner by address"""
    regex = r"^READ_DATA offs=(\d*) size=(\d*) data=(.*)$"
    var_dict: dict[int, bytes] = {}

    for match in re.finditer(regex, log_text, re.MULTILINE):
        data = bytes(int(v, base=16) for v in match.group(3).strip().split(' '))
        if len(data) <= 8:
            var_dict[int(match.group(1))] = data

    return var_dict


def write_program(ret: Sequence[Any], sdb: stencil_database, path: str) -> dict[Any, tuple[int, int, str]]:
    """Compile the values in ret to a runner command file that runs the program and
    reads back all values. Additionally a '-dump' file is written that dumps the
    patched code instead of running it (for debugging).

    Returns:
        Variable layout of the compiled program
    """
    dw, variables = compile_to_dag([Store(r) for r in ret], sdb)

    du = dw.copy()
    du.write_com(_binwrite.Command.DUMP_CODE)
    du.to_file(path.replace('.copapy', '-dump.copapy'))

    dw.write_com(_binwrite.Command.RUN_PROG)
    for v in ret:
        assert isinstance(v, value)
        add_read_value_remote(dw, variables, v.net)
    dw.write_com(_binwrite.Command.END_COM)
    dw.to_file(path)

    return variables


def run_program(runner: str, path: str, qemu: Sequence[str] = ()) -> str | None:
    """Run a command file written by write_program (the dump file first, which writes
    the patched code to <path>.bin) and check for a successful return.

    Arguments:
        runner: path of the runner executable
        path: command file
        qemu: command prefix for running a runner of a foreign architecture

    Returns:
        Runner output or None if runner or qemu are not available (test skipped with a warning)
    """
    if qemu and not check_for_qemu(qemu):
        warnings.warn(f"{qemu[-1] if qemu[0] != 'wsl' else qemu[1]} not found, test with {runner} skipped!", UserWarning)
        return None
    if not os.path.isfile(runner) and not os.path.isfile(runner + '.exe'):
        warnings.warn(f"{runner} not found, test skipped!", UserWarning)
        return None

    run_command(list(qemu) + [runner, path.replace('.copapy', '-dump.copapy'), path + '.bin'])
    result = run_command(list(qemu) + [runner, path])

    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'Return value: 1' in result
    return result


def check_results(result: str, ret_test: Sequence[Any], ret_ref: Sequence[Any],
                  variables: dict[Any, tuple[int, int, str]], sdb: stencil_database, rel: float = 1e-5) -> None:
    """Compare the values read back by the runner with reference values"""
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
        assert val == pytest.approx(ref, rel), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


def ops_program(c_i: NumLike, c_f: NumLike, c_b: NumLike) -> list[NumLike]:
    """Common test program for the operations on all architectures. Called with
    copapy values for the test and with Python numbers for the reference."""
    def arithmetic(c1: NumLike) -> list[NumLike]:
        return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
                c1 * 4, c1 * -4,
                c1 + 4, c1 - 4,
                c1 > 2, c1 > 100, c1 < 4, c1 < 100,
                c1 * 4.44, c1 * -4.44]

    def iiftests(c1: NumLike) -> list[NumLike]:
        return [iif(c1 > 5, 8, 9),
                iif(c1 < 5, 8.5, 9.5),
                iif(1 > 5, 3.3, 8.8) + c1,
                iif(1 < 5, c1 * 3.3, 8.8),
                iif(c1 < 5, c1 * 3.3, 8.8)]

    def mathtests(c1: NumLike) -> list[NumLike]:
        return [cp.sin(c1), cp.cos(c1), cp.tan(c1 / 10), cp.asin(c1 / 10), cp.atan(c1),
                cp.atan2(c1, 2.5), cp.exp(c1 / 10), cp.log(c1), cp.sqrt(c1)]

    return (arithmetic(c_i) + arithmetic(c_f) +
            [c_i / 4, c_i == 9, c_i == 4, c_i != 9, c_i != 4, c_i % 2] +
            [c_b == True, c_b == False, c_b != True, c_b != False, c_b / 2, c_b + 2] +  # noqa: E712
            iiftests(c_i) + iiftests(c_f) +
            mathtests(c_i) + mathtests(c_f))


def run_ops_test(arch: str, runner: str, qemu: Sequence[str] = ()) -> None:
    """Compile ops_program for arch, run it on the runner and check the results"""
    ret_test = ops_program(value(9), value(1.111), value(True))
    ret_ref = ops_program(9, 1.111, True)

    sdb = backend.stencil_db_from_package(arch)
    path = f'build/runner/test-{arch}.copapy'
    variables = write_program(ret_test, sdb, path)

    result = run_program(runner, path, qemu)
    if result is not None:
        check_results(result, ret_test, ret_ref, variables, sdb)


def run_vector_test(arch: str, runner: str, qemu: Sequence[str] = ()) -> None:
    """Compile a vector program for arch, run it on the runner and compare the
    results to the x86_64 reference"""
    t1 = cp.vector([10, 11, 12]) + cp.vector(cp.value(v) for v in range(3))
    t2 = t1.sum()

    t3 = cp.vector(cp.value(1 / (v + 1)) for v in range(3))
    t4 = ((t3 * t1) * 2).sum()
    t5 = ((t3 * t1) * 2).magnitude()

    path = f'build/runner/test-{arch}-vector.copapy'
    write_program([t2, t4, t5], backend.stencil_db_from_package(arch), path)

    result = run_program(runner, path, qemu)
    if result is not None:
        assert " size=4 data=24 00 00 00" in result
        assert " size=4 data=56 55 25 42" in result
        assert " size=4 data=B4 F9 C8 41" in result
