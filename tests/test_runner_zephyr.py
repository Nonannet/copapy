"""Runs the test programs on the Zephyr runner (targets/zephyr). The
command stream is sent split into frames over the console UART, the runner
answers with binary messages (see docs/source/runner_transfer.md).

By default the runner for the Cortex-M7 board mps2/an500 is run in
qemu-system-arm, build it with: tools/build_zephyr.sh mps2/an500

To run the tests on a board (e.g. Teensy 4.1) set the environment variable
COPARUN_SERIAL_PORT to the serial port of its console (requires pyserial).
"""
import os
import subprocess
import threading
import warnings
from typing import Sequence

import pytest

from copapy._serial import (CONTROL_ABORT, CONTROL_RESET, MSG_CONTROL, MSG_DONE, MSG_READY, Connection,
                            MessageReader, control_frame, frame)
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


class PipeConnection():
    """Connection to the runner in qemu over stdin and stdout"""
    def __init__(self, proc: 'subprocess.Popen[bytes]') -> None:
        assert proc.stdin and proc.stdout
        self.stdin = proc.stdin
        self.stdout = proc.stdout

    def write(self, data: bytes) -> None:
        self.stdin.write(data)

    def flush(self) -> None:
        self.stdin.flush()

    def read(self, size: int) -> bytes:
        return self.stdout.read(size)


def read_until(rx: MessageReader, msg_type: int) -> bytes:
    """Reads the messages of the runner up to one of msg_type and returns its payload"""
    while True:
        received, payload = rx.read()
        if received == msg_type:
            return payload


def send_stream(con: Connection, rx: MessageReader, data: bytes) -> int:
    """Sends a command stream in frames, each frame is answered by the runner

    Returns:
        State flag of the parser after the last frame (1: END_COM reached)
    """
    ret = 0
    for i in range(0, len(data), FRAME_SIZE):
        con.write(frame(data[i:i + FRAME_SIZE]))
        con.flush()
        msg_type, payload = rx.read()
        assert msg_type == MSG_DONE, f"Runner error {payload!r}\n -Output: {rx.output}"
        ret = int.from_bytes(payload, 'little', signed=True)
        if ret != 0:
            break
    return ret


def run_zephyr_program(runner: str, path: str, qemu: Sequence[str] = ()) -> str | None:
    """Run a command file on the Zephyr runner in qemu or on a board

    Returns:
        Runner output or None if runner or qemu are not available (test skipped with a warning)
    """
    with open(path, 'rb') as f:
        data = f.read()

    lines: list[str] = []
    error = ''
    ret = 0

    if SERIAL_PORT:
        import serial  # pyserial

        with serial.Serial(SERIAL_PORT, 115200, timeout=10) as ser:
            # The board may be in the middle of a command stream of an aborted run
            ser.write(control_frame(CONTROL_ABORT))
            rx = MessageReader(ser, [])
            while read_until(rx, MSG_CONTROL) != bytes((CONTROL_ABORT,)):
                pass
            rx.output = lines
            ret = send_stream(ser, rx, data)
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
            con = PipeConnection(proc)
            rx = MessageReader(con, lines)
            read_until(rx, MSG_READY)
            ret = send_stream(con, rx, data)
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

    assert ret == 1, f"\n -State flag: {ret}\n -Error occurred: {error}\n -Output: {result}"
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
