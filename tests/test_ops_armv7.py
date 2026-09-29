import pytest

from runner_helpers import qemu_command, run_ops_test


@pytest.mark.runner
def test_compile() -> None:
    run_ops_test('armv7', 'build/runner/coparun-armv7', qemu_command('qemu-arm', guest_base=True))


if __name__ == "__main__":
    test_compile()
