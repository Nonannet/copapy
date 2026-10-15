"""Tests for SerialTarget with the Zephyr runner (targets/zephyr).

By default the runner for mps2/an500 is run in qemu-system-arm (see
test_runner_zephyr.py), build it with: tools/build_zephyr.sh mps2/an500

To run the tests on a Teensy 4.1 set the environment variable
COPARUN_SERIAL_PORT to the serial port of the runner (requires pyserial).
"""
import os
import subprocess
import threading
import time
import warnings
from typing import Iterator

import pytest

import copapy as cp
from runner_helpers import check_for_qemu
from test_runner_zephyr import QEMU, RUNNER, SERIAL_PORT, TIMEOUT


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

    def readline(self) -> bytes:
        return self.stdout.readline()


@pytest.fixture
def target() -> Iterator[cp.SerialTarget]:
    if SERIAL_PORT:
        with cp.SerialTarget(SERIAL_PORT, cp.devices.teensy41) as tg:
            yield tg
            tg.stop()
        return

    if not check_for_qemu(QEMU):
        warnings.warn(f"{QEMU[1] if QEMU[0] == 'wsl' else QEMU[0]} not found, test skipped!", UserWarning)
        pytest.skip()
    if not os.path.isfile(RUNNER):
        warnings.warn(f"{RUNNER} not found, test skipped!", UserWarning)
        pytest.skip()

    proc = subprocess.Popen(QEMU + ['-kernel', RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    watchdog = threading.Timer(TIMEOUT, proc.kill)
    watchdog.start()
    try:
        yield cp.SerialTarget(device=cp.devices.mps2_an500, connection=PipeConnection(proc))
    finally:
        watchdog.cancel()
        proc.kill()
        proc.wait()


def approx(v: object) -> object:
    return pytest.approx(v, rel=1e-6, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.runner
def test_exec(target: cp.SerialTarget) -> None:
    a = cp.value(3.0)
    arr = cp.array([1.0, 2.0, 3.0])
    b = cp.sqrt(a) + 1
    c = arr * a

    target.deploy(b, c)
    target.exec()
    assert target.read_value(b) == approx(3.0 ** 0.5 + 1)
    assert target.read_value(c) == approx([3.0, 6.0, 9.0])

    target.write_value(a, 16.0)
    target.write_value(arr, [0.5, 1.5, 2.5])
    target.exec()
    assert target.read_value([a, b]) == approx([16.0, 5.0])
    assert target.read_value(c) == approx([8.0, 24.0, 40.0])
    assert target.stats() == {'runs': 2, 'overruns': 0}


def control_tasks() -> tuple[cp.value[float], cp.value[int], list[cp.value[float]], cp.Task, cp.Task]:
    speed_ref = cp.value(0.0)
    encoder = cp.value(0)
    err = speed_ref - encoder * 0.01
    result1 = cp.clamp(err * 5, -1.0, 1.0)
    result2 = cp.maximum(result1 * 0.99, cp.abs(err))
    task1 = cp.Task((result1, err), period=1e-3, priority=2)
    task2 = cp.Task(result2, period=0.01, priority=1)
    return speed_ref, encoder, [result1, err, result2], task1, task2


@pytest.mark.runner
def test_step_equals_local(target: cp.SerialTarget) -> None:
    # Simulated time on the device gives the same results as on the host
    speed_ref, encoder, outputs, task1, task2 = control_tasks()
    local = cp.LocalTarget()
    for tg in (target, local):
        tg.deploy(task1, task2)

    for t, ref, enc in [(0.0005, 0.1, 0), (0.0123, 0.3, 7), (0.05, -0.2, 3), (0.1, 0.25, -40)]:
        results = []
        for tg in (target, local):
            tg.write_value([speed_ref, encoder], [ref, enc])
            tg.step(t)
            results.append(tg.read_value(outputs))
        assert results[0] == results[1]
    assert target.stats(task1) == local.stats(task1) == {'runs': 100, 'overruns': 0}


@pytest.mark.runner
def test_realtime(target: cp.SerialTarget) -> None:
    speed_ref, encoder, outputs, task1, task2 = control_tasks()
    target.deploy(task1, task2)
    target.write_value(speed_ref, 0.1)

    start = time.perf_counter()
    target.run()
    time.sleep(0.3)
    assert target.read_value(outputs) == approx([0.5, 0.1, 0.495])

    target.write_value(encoder, 100)
    time.sleep(0.1)
    assert target.read_value(outputs) == approx([-1.0, -0.9, 0.9])

    with pytest.raises(RuntimeError):
        target.exec(task1)
    target.stop()
    elapsed = time.perf_counter() - start

    runs1, runs2 = target.stats(task1)['runs'], target.stats(task2)['runs']
    print(f"{elapsed=:.3f} s, {runs1=}, {runs2=}, {target.stats(task1)}")
    assert target.stats(task1)['overruns'] == 0
    assert 0 < runs1 <= elapsed / 1e-3 + 1
    if SERIAL_PORT:
        assert runs1 > 0.8 * elapsed / 1e-3  # qemu (WSL) may run slower than real time
    assert abs(runs2 - runs1 / 10) <= 1

    # Stopped: outputs do not change anymore
    assert target.stats(task1)['runs'] == runs1


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
