"""Shared helpers for tests running compiled copapy programs with a coparun runner"""
import operator
import os
import re
import struct
import subprocess
import warnings
from typing import Any, Sequence

import pytest

import copapy as cp
import copapy.backend as backend
from copapy import NumLike, iif, value, vector, _binwrite
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
                  variables: dict[Any, tuple[int, int, str]], sdb: stencil_database,
                  rel: float = 1e-5, abs_tol: float | None = None) -> None:
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
        assert val == pytest.approx(ref, rel, abs_tol), f"Result does not match: {val} and reference: {ref}"  # pyright: ignore[reportUnknownMemberType]


def ops_program(c_i: NumLike, c_f: NumLike, c_b: NumLike) -> list[NumLike]:
    """Common test program for the operations. Called with copapy values
    for the test and with Python numbers for the reference."""
    def arithmetic(c1: NumLike) -> list[NumLike]:
        return [c1 / 4, c1 / -4, c1 // 4, c1 // -4, (c1 * -1) // 4,
                c1 * 4, c1 * -4,
                c1 + 4, c1 - 4,
                c1 > 2, c1 > 100, c1 < 4, c1 < 100,
                c1 / 4.44, c1 / -4.44, c1 // 4.44, c1 // -4.44, (c1 * -1) // 4.44,
                c1 * 4.44, c1 * -4.44,
                c1 + 4.44, c1 - 4.44,
                c1 > 100.11, c1 < 4.44, c1 < 100.11]

    def iiftests(c1: NumLike) -> list[NumLike]:
        return [iif(c1 > 5, 8, 9),
                iif(c1 < 5, 8.5, 9.5),
                iif(1 > 5, 3.3, 8.8) + c1,
                iif(1 < 5, c1 * 3.3, 8.8),
                iif(c1 < 5, c1 * 3.3, 8.8)]

    return (arithmetic(c_i) + arithmetic(c_f) +
            [c_i / 4, c_i == 9, c_i == 4, c_i != 9, c_i != 4, c_i % 2] +
            [c_b == True, c_b == False, c_b != True, c_b != False, c_b / 2, c_b + 2] +  # noqa: E712
            iiftests(c_i) + iiftests(c_f))


def ops_test_values() -> tuple[list[NumLike], list[NumLike]]:
    """Returns the ops test program and its reference values"""
    return ops_program(value(9), value(1.111), value(True)), ops_program(9, 1.111, True)


TRIG_VALS = [0.0, 0.0001, 0.5, 1.5, 2.5, 3.5, 6.28318530718, 8.25, 100.0, 100000.0,
             -0.0001, -0.5, -1.5, -2.5, -3.5, -6.28318530718, -8.25, -100.0, -100000.0]
ARC_VALS = [-1.0, -0.95, -0.5, -0.01, 0.0, 0.01, 0.5, 0.95, 1.0]
SIGNED_VALS = [-1000.5, -2.5, 0.0, 2.5, 1000.5]
BINARY_VALS = [(1.0, 3.0), (-1.0, -3.0), (-1.0, 3.0), (1.0, 0.0), (0.0, 3.0), (0.5, -3.0), (2.5, 2.111)]


def math_program(val: Any) -> list[NumLike]:
    """Program calling all math functions. val converts the inputs: cp.value for
    the test and a no-op for the Python reference."""
    ret: list[NumLike] = []
    for func, args in [(cp.sqrt, [0.0, 0.0001, 0.5, 2.0, 6.25, 100000.0]),
                       (cp.exp, [-10.0, -1.0, 0.0, 0.5, 2.5, 10.0]),
                       (cp.log, [0.0001, 0.5, 0.999, 1.0, 2.5, 100000.0]),
                       (cp.sin, TRIG_VALS),
                       (cp.cos, TRIG_VALS),
                       (cp.tan, TRIG_VALS),
                       (cp.asin, ARC_VALS),
                       (cp.acos, ARC_VALS),
                       (cp.atan, ARC_VALS + [-1000.0, -2.0, 10.0]),
                       (cp.abs, SIGNED_VALS),
                       (cp.sign, SIGNED_VALS),
                       (cp.relu, SIGNED_VALS),
                       (cp.sigmoid, [-20.0, -1.0, 0.0, 2.5, 20.0]),
                       (cp.get_42, [1.0])]:
        ret += [func(val(a)) for a in args]

    # Integer arguments
    ret += [cp.abs(val(-9)), cp.sign(val(-9)), cp.sign(val(0)), cp.sign(val(7)), cp.sqrt(val(9)),
            cp.minimum(val(-9), 5), cp.maximum(val(-9), 5), cp.clamp(val(-9), -5, 5), cp.relu(val(-9))]

    # Both arguments variable, only the first or only the second one
    for func2 in (cp.atan2, cp.pow, cp.minimum, cp.maximum):
        for a, b in BINARY_VALS:
            ret += [func2(val(a), val(b)), func2(val(a), b), func2(a, val(b))]

    ret += [cp.clamp(val(x), 0.0, 1.0) for x in (-2.5, 0.3, 2.5)]
    ret += [cp.clamp(val(-7.0), val(-5.0), -1.0), cp.clamp(val(0.5), -1.0, val(0.25))]

    ret += [val(2.5) ** 2, val(9) ** 2, val(9) ** -1, val(9) ** 0.5, val(9) ** 2.111, val(2.5) ** -2, 2 ** val(2.5)]
    return ret


def math_test_values() -> tuple[list[NumLike], list[NumLike]]:
    """Returns the math test program and its reference values"""
    return math_program(value), math_program(lambda x: x)


def vector_program(v1: vector[Any], v2: vector[Any], vi: vector[Any]) -> list[NumLike]:
    """Program with vector operations, returns all resulting scalars"""
    def flat(*items: Any) -> list[NumLike]:
        return [x for item in items for x in (item.values if isinstance(item, vector) else [item])]

    t1 = cp.vector([10, 11, 12]) + vi
    t3 = cp.vector([1 / (i + 1) for i in range(3)]) * v1

    return flat(t1, t1.sum(), ((t3 * t1) * 2).sum(), ((t3 * t1) * 2).magnitude(),
                v1 + v2, v1 - v2, v1 * v2, v1 / v2, v1 ** 2, 2.5 - v1, 2.5 / v1, -v2,
                v1 > 1.5, v2 < 0.0, v1 == cp.vector([1.0, 5.0, 3.0]),
                v1.dot(v2), v1 @ v2, v1.cross(v2), v1.sum(), v1.magnitude(), v1.normalize(),
                v1.map(lambda x: x * x + 1),
                cp.distance(v1, v2), cp.scalar_projection(v1, v2), cp.vector_projection(v1, v2),
                cp.angle_between(v1, v2), cp.rotate_vector(v1, cp.vector([0.0, 0.0, 1.0]), 1.234),
                cp.sqrt(v1), cp.sin(v2), cp.exp(v1), cp.atan2(v1, v2),
                cp.minimum(v1, v2), cp.maximum(v1, 2.0), cp.clamp(v2, -1.0, 2.0), cp.abs(v2),
                cp.concat([v1, v2]).sum())


def vector_test_values() -> tuple[list[NumLike], list[NumLike]]:
    """Returns the vector test program and its reference values"""
    a = [1.0, 2.0, 3.0]
    b = [4.0, -5.0, 6.5]
    ret_test = vector_program(cp.vector(value(x) for x in a), cp.vector(value(x) for x in b), cp.vector(value(i) for i in range(3)))
    ret_ref = vector_program(cp.vector(a), cp.vector(b), cp.vector(range(3)))
    return ret_test, ret_ref


def array_test_values() -> tuple[list[NumLike], list[NumLike]]:
    """Returns a program with array stencils and its reference values. The array
    results are checked element-wise by reading the array memory as scalars."""
    n = 11  # not a multiple of the SIMD width or the unrolling of the reductions
    fa = [i * 0.75 - 3.2 for i in range(n)]
    fb = [(i * 5) % 7 + 0.5 for i in range(n)]
    ia = [i * 3 - 10 for i in range(n)]
    ib = [(i * 7) % 5 + 1 for i in range(n)]
    rows = [[(r * 3 + i) % 4 - 1.5 for i in range(n)] for r in range(3)]
    arrays = {id(fa): cp.array(fa), id(fb): cp.array(fb), id(ia): cp.array(ia), id(ib): cp.array(ib)}
    fs, i_s = value(1.5), value(3)

    ret_test: list[NumLike] = []
    ret_ref: list[NumLike] = []

    def add(result: Any, ref: list[NumLike]) -> None:
        if isinstance(result, cp.array):
            ret_test.extend(result.element(i) for i in range(result.size))
            ret_ref.extend(ref)
        else:
            ret_test.append(result)
            ret_ref.extend(ref)

    for f in (operator.add, operator.sub, operator.mul, operator.truediv):
        for x, y in [(fa, fb), (fa, ib), (ia, fb), (ia, ib)]:
            ax, ay = arrays[id(x)], arrays[id(y)]
            add(f(ax, ay), [f(p, q) for p, q in zip(x, y)])
            add(f(ax, fs), [f(p, 1.5) for p in x])
            add(f(i_s, ay), [f(3, q) for q in y])

    for x, y in [(fa, fb), (ia, ib), (ia, fb)]:
        ax, ay = arrays[id(x)], arrays[id(y)]
        add(ax.sum(), [sum(x)])
        add(ax @ ay, [sum(p * q for p, q in zip(x, y))])

    add(cp.array(rows) @ arrays[id(fb)], [sum(p * q for p, q in zip(r, fb)) for r in rows])

    # Scalar operations on array elements and reduction results
    af = arrays[id(fa)]
    add((af * fs)[3] * 2 + cp.sqrt(arrays[id(fb)].sum()), [fa[3] * 1.5 * 2 + sum(fb) ** 0.5])

    # Matrix products, strided copies (transpose, slices, broadcasting) and packed values
    ma = [[(r * 5 + c) % 7 - 3.5 for c in range(6)] for r in range(4)]
    mi = [[(r * 3 + c) % 5 - 2 for c in range(3)] for r in range(6)]
    am, ai = cp.array(ma), cp.array(mi)
    add(am @ ai, [sum(ma[r][k] * mi[k][c] for k in range(6)) for r in range(4) for c in range(3)])
    add(ai.T @ ai, [sum(mi[k][r] * mi[k][c] for k in range(6)) for r in range(3) for c in range(3)])
    add(am.T, [ma[r][c] for c in range(6) for r in range(4)])
    add(am[1:, ::2], [ma[r][c] for r in range(1, 4) for c in range(0, 6, 2)])
    add(am[:, 4] + cp.array([[1.0], [2.0]]), [ma[r][4] + b for b in (1.0, 2.0) for r in range(4)])
    packed = cp.array([fs, fs * fs, 0.5, fs + 1.0])
    add(packed * packed[1], [v * 2.25 for v in (1.5, 2.25, 0.5, 2.5)])

    # Convolutions with padding and stride: input 2 x 5 x 9, kernels 3 x 2 x 3 x 3
    cx = [[[(q * 11 + r * 5 + c * 3) % 7 - 3.0 for c in range(9)] for r in range(5)] for q in range(2)]
    cw = [[[[(o * 7 + q * 5 + r * 3 + c) % 5 * 0.25 - 0.5 for c in range(3)] for r in range(3)] for q in range(2)] for o in range(3)]
    cb = [0.5, -1.0, 2.0]

    def conv_ref(o: int, oy: int, ox: int) -> float:
        return cb[o] + sum(cw[o][q][ky][kx] * cx[q][oy * 2 - 1 + ky][ox - 1 + kx] for q in range(2) for ky in range(3)
                           for kx in range(3) if 0 <= oy * 2 - 1 + ky < 5 and 0 <= ox - 1 + kx < 9)

    add(cp.conv2d(cp.array(cx), cp.array(cw), cp.array(cb), stride=(2, 1), padding=1),
        [conv_ref(o, oy, ox) for o in range(3) for oy in range(3) for ox in range(9)])
    c1 = [[[(o * 3 + q * 2 + k) % 4 * 0.5 - 0.75 for k in range(3)] for q in range(5)] for o in range(2)]
    add(cp.conv1d(cp.array(cx[0]), cp.array(c1), padding=1),
        [sum(c1[o][q][k] * cx[0][q][i - 1 + k] for q in range(5) for k in range(3) if 0 <= i - 1 + k < 9)
         for o in range(2) for i in range(9)])

    return ret_test, ret_ref


TEST_PROGRAMS = {
    'ops': ops_test_values,
    'math': math_test_values,
    'vector': vector_test_values,
    'array': array_test_values,
}


def run_runner_test(name: str, arch: str, runner: str, qemu: Sequence[str] = ()) -> None:
    """Compile the test program name ('ops', 'math' or 'vector') for arch,
    run it on the runner and compare the results with the Python reference

    Arguments:
        name: test program
        arch: stencil architecture, 'native' for the stencils of this machine
        runner: path of the runner executable
        qemu: command prefix for running a runner of a foreign architecture
    """
    ret_test, ret_ref = TEST_PROGRAMS[name]()
    assert len(ret_test) == len(ret_ref)

    sdb = backend.stencil_db_from_package(arch)
    path = f'build/runner/test-{arch}-{name}.copapy'
    variables = write_program(ret_test, sdb, path)

    result = run_program(runner, path, qemu)
    if result is not None:
        check_results(result, ret_test, ret_ref, variables, sdb, rel=1e-5, abs_tol=1e-5)
