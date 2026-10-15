"""Runs the test programs on the Zephyr runner (targets/zephyr). The
command stream is sent split into frames over the console UART (see
targets/zephyr/src/main.c).

By default the runner for the Cortex-M7 board mps2/an500 is run in
qemu-system-arm, build it with: tools/build_zephyr.sh mps2/an500

To run the tests on a board (e.g. Teensy 4.1) set the environment variable
COPARUN_SERIAL_PORT to the serial port of its console (requires pyserial).
"""
import os
import struct
import subprocess
import threading
import warnings
import zlib
from typing import IO, Sequence, cast

import pytest

from runner_helpers import check_for_qemu, qemu_command, run_runner_test

ARCH = 'armv7mthumb_fpv5'
RUNNER = 'build/runner/coparun-zephyr-mps2_an500.elf'
# Path of qemu-system-arm, e.g. from the host tools of the Zephyr SDK
QEMU = qemu_command(os.environ.get('QEMU_SYSTEM_ARM', 'qemu-system-arm')) + [
    '-M', 'mps2-an500', '-display', 'none', '-monitor', 'none', '-serial', 'stdio', '-no-reboot']
SERIAL_PORT = os.environ.get('COPARUN_SERIAL_PORT', '')
TOL = 5e-5  # sin and cos of CMSIS-DSP are based on a table with linear interpolation
TIMEOUT = 120  # s
FRAME_SIZE = 4096  # CONFIG_COPARUN_MAX_FRAME_SIZE
CONTROL_ABORT = 0
CONTROL_RESET = 1


def frame(data: bytes) -> bytes:
    return b'CPY1' + struct.pack('<I', len(data)) + data + struct.pack('<I', zlib.crc32(data))


def control_frame(code: int) -> bytes:
    return b'CPYC' + struct.pack('<I', code)


def read_until(rx: IO[bytes], lines: list[str], marker: str) -> None:
    """Appends the lines of the runner to lines up to a line starting with marker"""
    while True:
        raw = rx.readline()
        assert raw.endswith(b'\n'), f"No answer from the runner\n -Output: {lines}"
        lines.append(raw.decode('utf8', errors='replace').rstrip('\r\n'))
        if lines[-1].startswith(marker) or lines[-1].startswith('COPARUN_ERROR'):
            return


def send_stream(tx: IO[bytes], rx: IO[bytes], lines: list[str], data: bytes) -> None:
    """Sends a command stream in frames, each frame is answered by the runner"""
    for i in range(0, len(data), FRAME_SIZE):
        tx.write(frame(data[i:i + FRAME_SIZE]))
        tx.flush()
        read_until(rx, lines, 'COPARUN_DONE')
        if not lines[-1].startswith('COPARUN_DONE ret=0'):
            break


def run_zephyr_program(runner: str, path: str, qemu: Sequence[str] = ()) -> str | None:
    """Run a command file on the Zephyr runner in qemu or on a board

    Returns:
        Runner output or None if runner or qemu are not available (test skipped with a warning)
    """
    with open(path, 'rb') as f:
        data = f.read()

    lines: list[str] = []
    error = ''

    if SERIAL_PORT:
        import serial  # pyserial

        with serial.Serial(SERIAL_PORT, 115200, timeout=10) as ser:
            # The board may be in the middle of a command stream of an aborted run
            ser.write(control_frame(CONTROL_ABORT))
            read_until(cast(IO[bytes], ser), [], 'COPARUN_ABORT')
            send_stream(cast(IO[bytes], ser), cast(IO[bytes], ser), lines, data)
    else:
        if not check_for_qemu(qemu):
            warnings.warn(f"{qemu[1] if qemu[0] == 'wsl' else qemu[0]} not found, test with {runner} skipped!", UserWarning)
            return None
        if not os.path.isfile(runner):
            warnings.warn(f"{runner} not found, test skipped!", UserWarning)
            return None

        proc = subprocess.Popen(list(qemu) + ['-kernel', runner], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin and proc.stdout and proc.stderr
        watchdog = threading.Timer(TIMEOUT, proc.kill)
        watchdog.start()

        try:
            read_until(proc.stdout, lines, 'COPARUN_READY')
            send_stream(proc.stdin, proc.stdout, lines, data)
            proc.stdin.write(control_frame(CONTROL_RESET))  # terminates qemu (-no-reboot)
            proc.stdin.flush()
            proc.wait()
        finally:
            watchdog.cancel()
            proc.kill()
        error = proc.stderr.read().decode('utf8', errors='replace')

    result = '\n'.join(lines)
    print('* Output from runner:\n--')
    print(result)
    print('--')

    assert 'COPARUN_DONE ret=1' in result, f"\n -Error occurred: {error}\n -Output: {result}"
    assert 'Return value: 1' in result
    return result


@pytest.mark.runner
def test_ops() -> None:
    run_runner_test('ops', ARCH, RUNNER, QEMU, run=run_zephyr_program)


@pytest.mark.runner
def test_math() -> None:
    run_runner_test('math', ARCH, RUNNER, QEMU, TOL, run=run_zephyr_program)


@pytest.mark.runner
def test_vector() -> None:
    run_runner_test('vector', ARCH, RUNNER, QEMU, TOL, run=run_zephyr_program)


@pytest.mark.runner
def test_array() -> None:
    run_runner_test('array', ARCH, RUNNER, QEMU, TOL, run=run_zephyr_program)


if __name__ == "__main__":
    test_ops()
    test_math()
    test_vector()
    test_array()
