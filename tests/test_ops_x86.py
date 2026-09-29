import os
import platform

import pytest

import copapy as cp
import copapy.backend as backend
from runner_helpers import check_results, qemu_command, run_ops_test, run_program, run_vector_test, write_program

RUNNER = 'build/runner/coparun-x86'

if os.name == 'nt':
    qemu = qemu_command('qemu-i386', guest_base=True)
elif platform.machine().lower() in ('x86_64', 'amd64', 'i386', 'i686'):
    qemu = []  # 32 bit x86 code runs natively
else:
    qemu = qemu_command('qemu-i386')


@pytest.mark.runner
def test_compile() -> None:
    run_ops_test('x86', RUNNER, qemu)


@pytest.mark.runner
def test_vector_compile() -> None:
    run_vector_test('x86', RUNNER, qemu)


@pytest.mark.runner
def test_sinus() -> None:
    a_val = 8.25  # Error on x86 with a Windows ABI runner if a > 2 PI --> Sin result > 1

    a = cp.value(a_val)
    b = cp.value(0.87)

    # Define computations
    c = a + b * 2.0
    si = cp.sin(a)
    d = c ** 2 + si
    e = d + cp.sqrt(b)

    ret_test = [si, e]
    ret_ref = [cp.sin(a_val), (a_val + 0.87 * 2.0) ** 2 + cp.sin(a_val) + cp.sqrt(0.87)]

    sdb = backend.stencil_db_from_package('x86')
    path = 'build/runner/test-x86-sinus.copapy'
    variables = write_program(ret_test, sdb, path)

    result = run_program(RUNNER, path, qemu)
    if result is not None:
        check_results(result, ret_test, ret_ref, variables, sdb, rel=1e-6)


if __name__ == "__main__":
    test_compile()
