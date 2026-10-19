import os

import pytest

from runner_helpers import run_runner_test

ARCH = 'tricore'
RUNNER = 'build/runner/coparun-tricore.elf'
# No qemu-user exists for TriCore: The bare metal runner is run in
# qemu-system-tricore by this script (on Windows in WSL)
QEMU = (['wsl'] if os.name == 'nt' else []) + ['sh', 'tools/tricore/run_qemu.sh']


@pytest.mark.runner
def test_ops() -> None:
    run_runner_test('ops', ARCH, RUNNER, QEMU)


@pytest.mark.runner
def test_math() -> None:
    run_runner_test('math', ARCH, RUNNER, QEMU)


@pytest.mark.runner
def test_vector() -> None:
    run_runner_test('vector', ARCH, RUNNER, QEMU)


@pytest.mark.runner
def test_array() -> None:
    run_runner_test('array', ARCH, RUNNER, QEMU)


if __name__ == "__main__":
    test_ops()
    test_math()
    test_vector()
    test_array()
