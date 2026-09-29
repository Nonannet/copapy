import pytest

from runner_helpers import qemu_command, run_ops_test


@pytest.mark.runner
def test_compile() -> None:
    run_ops_test('riscv64', 'build/runner/coparun-riscv64', qemu_command('qemu-riscv64'))


if __name__ == "__main__":
    test_compile()
