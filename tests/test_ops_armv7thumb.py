import pytest

from runner_helpers import qemu_command, run_ops_test


@pytest.mark.runner
def test_compile() -> None:
    # Thumb code runs on the ARMv7 runner
    run_ops_test('armv7thumb', 'build/runner/coparun-armv7', qemu_command('qemu-arm', guest_base=True))


if __name__ == "__main__":
    test_compile()


"""
qemu-arm -d in_asm,exec,cpu_reset -D qemu.log build/runner/coparun-armv7 build/runner/test-armv7thumb-dump.copapy build/runner/test-armv7thumb.copapy.bin

qemu-arm -d in_asm,exec -D qemu_trace.log \
  -global driver=pl011.audiomaddr,property=addr,value=0xff7ec000 \
  -global driver=pl011.audiomaddr,property=size,value=0x100000 \
  your_binary
"""
