import pytest

from runner_helpers import qemu_command, run_ops_test


@pytest.mark.runner
def test_compile() -> None:
    # Cortex-M Thumb code runs on the ARMv7 runner
    run_ops_test('armv7mthumb', 'build/runner/coparun-armv7', qemu_command('qemu-arm', guest_base=True))


if __name__ == "__main__":
    test_compile()
