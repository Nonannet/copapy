import pytest

from runner_helpers import qemu_command, run_ops_test


@pytest.mark.runner
def test_compile() -> None:
    run_ops_test('riscv32', 'build/runner/coparun-riscv32', qemu_command('qemu-riscv32', guest_base=True))


if __name__ == "__main__":
    test_compile()
