import pytest

from runner_helpers import qemu_command, run_vector_test


@pytest.mark.runner
def test_compile() -> None:
    run_vector_test('arm64', 'build/runner/coparun-aarch64', qemu_command('qemu-aarch64'))


if __name__ == "__main__":
    test_compile()
