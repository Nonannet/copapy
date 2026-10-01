import pytest

from runner_helpers import qemu_command, run_runner_test

ARCH = 'armv7'
RUNNER = 'build/runner/coparun-armv7'
QEMU = qemu_command('qemu-arm', guest_base=True)


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
