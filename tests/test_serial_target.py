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
from test_tasks import counts
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
    assert counts(target.stats()) == (2, 0)


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
    assert counts(target.stats(task1)) == counts(local.stats(task1)) == (100, 0)


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
    assert 0 < runs1 <= elapsed / 1e-3 + 1

    # Stopped: outputs do not change anymore
    assert target.stats(task1)['runs'] == runs1

    # Execution times
    for task, period in ((task1, 1e-3), (task2, 0.01)):
        stats = target.stats(task)
        print(task, stats)
        assert 0 <= stats['exec_time_mean'] <= stats['exec_time_max']
        assert 0 <= stats['exec_time'] <= stats['exec_time_max'] <= stats['response_time_max']
        assert stats['start_delay_max'] <= stats['response_time_max']
        if SERIAL_PORT:
            # Timing in qemu (WSL) is not real time
            assert stats['overruns'] == 0
            assert stats['response_time_max'] < period
            assert stats['exec_time_mean'] > 0
    if SERIAL_PORT:
        assert runs1 > 0.8 * elapsed / 1e-3
        assert abs(runs2 - runs1 / 10) <= 1


@pytest.mark.runner
def test_load_shedding(target: cp.SerialTarget) -> None:
    if not SERIAL_PORT:
        pytest.skip("Real-time behaviour is only tested on a board")

    # Heavy task: about 10 ms CPU time per 1 ms period
    n = 100
    m = cp.array([[0.01 * ((i * 7 + j * 3) % 11 - 5) for j in range(n)] for i in range(n)])
    v = cp.array([1.0] * n)
    x = v
    for _ in range(100):
        x = m @ x + v
    heavy = cp.Task(x.sum(), period=1e-3, priority=1)

    ref = cp.value(1.0)
    fast_out = ref * 2 + 1
    fast = cp.Task(fast_out, period=1e-3, priority=2)

    target.deploy(fast, heavy)
    target.run()
    start = time.perf_counter()
    requests = 0
    while time.perf_counter() - start < 1.0:
        # The host is served although the tasks need more than the CPU
        target.write_value(ref, float(requests))
        target.read_value(fast_out)
        requests += 1
    target.stop()

    fast_stats, heavy_stats = target.stats(fast), target.stats(heavy)
    print(requests, fast_stats, heavy_stats)
    assert requests > 100
    assert fast_stats['overruns'] == 0 and not fast_stats['degraded']
    assert heavy_stats['degraded'] and heavy_stats['degradations'] == 1
    assert heavy_stats['overruns'] > 0


@pytest.mark.runner
def test_gpio(target: cp.SerialTarget) -> None:
    if 'led' not in [c.name for c in target.io]:
        pytest.skip("No GPIO channels on this device")
    io = target.io
    print(io)

    # Inputs with pull-up, open pins read high
    in24, in25 = cp.value(False), cp.value(False)
    target.bind(in24, io.gpio('d24'))
    target.bind(in25, io.gpio('d25'))

    # Outputs from a task, read back by the next job
    on = cp.value(0)
    led = on > 0
    d10 = cp.iif(on > 1, 1, 0)
    readback = cp.value(False)
    target.bind(led, io.gpio('led'))
    target.bind(d10, io.gpio('d10'))
    target.bind(readback, io.gpio('led'))

    pins = [in24 + 1, in25 + 1, readback + 1]
    task = cp.Task([led, d10, pins], period=1e-3)
    target.deploy(task)

    for level, expected_led, expected_d10 in ((0, 0, 0), (1, 1, 0), (2, 1, 1), (0, 0, 0)):
        target.write_value(on, level)
        target.exec(task)  # writes the outputs
        target.exec(task)  # reads back the output
        assert target.read_value([led, d10]) == [expected_led, expected_d10]
        assert target.read_value(pins) == [2, 2, expected_led + 1]
        assert target.read_value([in24, in25, readback]) == [1, 1, expected_led]

    # Real time: the LED follows the input written by the host
    target.run()
    for level in (1, 0, 1, 0):
        target.write_value(on, level)
        time.sleep(0.05)
        assert target.read_value(readback) == level
    target.stop()

    with pytest.raises(ValueError):
        target.write_value(in24, True)  # read from a channel
    with pytest.raises(ValueError):
        target.bind(on > 2, io.gpio('d24'))  # input channel can not be written
    with pytest.raises(ValueError):
        target.bind(cp.value(False), io.gpio('d24'))  # already bound
    with pytest.raises(KeyError):
        io.gpio('d99')
    with pytest.raises(ValueError):
        cp.LocalTarget().bind(cp.value(False), io.gpio('d24'))  # no I/O


@pytest.mark.runner
def test_pwm_encoder_adc(target: cp.SerialTarget) -> None:
    if 'enc0' not in [c.name for c in target.io]:
        pytest.skip("No PWM, encoder and ADC channels on this device")
    io = target.io

    duty = cp.value(0.0)
    pwm = cp.clamp(duty, 0.0, 1.0)
    pos, a0, a1 = cp.value(0), cp.value(0), cp.value(0)
    target.bind(pwm, io.pwm('pwm2'))
    target.bind(pos, io.encoder('enc0'))
    target.bind(a0, io.adc('a0'))
    target.bind(a1, io.adc('a1'))
    task = cp.Task([pwm, pos + 1, a0 + 1, a1 + 1], period=1e-3)
    target.deploy(task)

    for d in (0.0, 0.25, 1.0, 2.0):
        target.write_value(duty, d)
        target.exec(task)
        assert target.read_value(pwm) == min(d, 1.0)
        # Encoder inputs with pull-up do not move, open ADC inputs give any value
        position, adc0, adc1 = target.read_value([pos, a0, a1])
        assert position == 0
        assert 0 <= adc0 <= 4095 and 0 <= adc1 <= 4095

    target.run()
    time.sleep(0.2)
    target.stop()
    stats = target.stats(task)
    print(stats)
    assert stats['overruns'] == 0 and stats['runs'] > 150

    with pytest.raises(TypeError):
        target.bind(cp.value(0.0), io.encoder('enc0'))  # int channel
    with pytest.raises(TypeError):
        target.bind(pos + 1, io.pwm('pwm3'))  # float channel
    with pytest.raises(ValueError):
        target.bind(cp.value(0.0), io.pwm('pwm3'))  # PWM can not be read
    with pytest.raises(KeyError):
        io.adc('enc0')


if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
