"""Tests for tasks on the local target: deployment, Logical Execution Time
semantics between tasks and inputs, arrays and error cases."""
from typing import Any

import pytest

import copapy as cp


def approx(v: object) -> object:
    return pytest.approx(v, rel=1e-6, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def counts(stats: dict[str, Any]) -> tuple[int, int]:
    return stats['runs'], stats['overruns']


def test_single_task_exec() -> None:
    a = cp.value(3.0)
    b = cp.sqrt(a) + 1
    c = cp.vector([a, b]) * 2

    tg = cp.LocalTarget()
    tg.deploy(b, c)
    tg.exec()
    assert tg.read_value(b) == approx(3.0 ** 0.5 + 1)
    assert tg.read_value(c).values == approx([6.0, (3.0 ** 0.5 + 1) * 2])

    tg.write_value(a, 16.0)
    assert tg.read_value(a) == 16.0
    assert tg.read_value(b) == approx(3.0 ** 0.5 + 1)  # not run yet
    tg.exec()
    assert tg.read_value(b) == approx(5.0)
    assert tg.read_value(c).values == approx([32.0, 10.0])
    assert counts(tg.stats()) == (2, 0)


def test_let_between_tasks() -> None:
    speed_ref = cp.value(0.0)
    encoder = cp.value(0)

    err = speed_ref - encoder * 0.01
    result1 = cp.clamp(err * 5, -1.0, 1.0)
    result2 = cp.maximum(result1 * 0.99, cp.abs(err))

    task1 = cp.Task((result1, err), period=1e-3, priority=2)
    task2 = cp.Task(result2, period=0.1, priority=1)

    tg = cp.LocalTarget()
    tg.deploy(task1, task2)
    tg.write_value(speed_ref, 0.1)

    # Outputs are published at the end of the period
    tg.step(0.0005)
    assert tg.read_value([result1, err, result2]) == [0.0, 0.0, 0.0]
    tg.step(0.001)
    assert tg.read_value([result1, err, result2]) == approx([0.5, 0.1, 0.0])

    # task2 reads the outputs of task1 published at its release
    tg.step(0.1)
    assert tg.read_value(result2) == 0.0
    tg.step(0.2)
    assert tg.read_value(result2) == approx(0.495)
    assert counts(tg.stats(task1)) == (200, 0)
    assert counts(tg.stats(task2)) == (2, 0)
    stats = tg.stats(task1)
    assert 0 <= stats['exec_time_mean'] <= stats['exec_time_max'] <= stats['response_time_max']

    # Inputs are read at the next release
    tg.step(0.3)
    tg.write_value(encoder, 100)
    tg.step(0.301)
    assert tg.read_value([result1, err, result2]) == approx([-1.0, -0.9, 0.495])
    tg.step(0.4)
    assert tg.read_value(result2) == approx(0.495)
    tg.step(0.5)
    assert tg.read_value(result2) == approx(0.9)
    assert tg.time == approx(0.5)


def test_shared_expression() -> None:
    # The expression is no output of task1: task2 computes it itself
    x = cp.value(2.0)
    y = x * 3
    t1 = cp.Task(y + 1, period=0.01)
    t2 = cp.Task(y * 2, period=0.02)

    tg = cp.LocalTarget()
    tg.deploy(t1, t2)
    tg.step(0.02)
    assert tg.read_value([t1.outputs, t2.outputs]) == approx([7.0, 12.0])


def test_arrays_between_tasks() -> None:
    a = cp.array([1.0, 2.0, 3.0, 4.0])
    s = cp.value(2.0)
    b = a * s
    total = b[0] + b[1] + b[2] + b[3]
    first = b[0] + 1

    t1 = cp.Task(b, period=0.01)
    t2 = cp.Task((total, first), period=0.01)

    tg = cp.LocalTarget()
    tg.deploy(t1, t2)
    tg.step(0.01)
    assert tg.read_value(b) == approx([2.0, 4.0, 6.0, 8.0])
    assert tg.read_value(b[2]) == approx(6.0)
    assert tg.read_value([total, first]) == [0.0, 1.0]  # from b = 0 before the first publish
    tg.step(0.02)
    assert tg.read_value([total, first]) == approx([20.0, 3.0])

    tg.write_value(a, [0.0, 1.0, 1.0, 1.0])
    tg.write_value(a[0], 10.0)
    assert tg.read_value(a) == approx([10.0, 1.0, 1.0, 1.0])
    tg.step(0.04)
    assert tg.read_value([total, first]) == approx([26.0, 21.0])


def test_triggered_task() -> None:
    x = cp.value(1.0)
    periodic = cp.Task(x + 1, period=0.01)
    triggered = cp.Task(x * 10)

    tg = cp.LocalTarget()
    tg.deploy([periodic, triggered])
    tg.step(0.05)
    assert tg.stats(periodic)['runs'] == 5
    assert tg.stats(triggered)['runs'] == 0
    assert tg.read_value(triggered.outputs) == 0.0
    tg.exec(triggered)
    assert tg.read_value(triggered.outputs) == approx(10.0)

    with pytest.raises(ValueError):
        tg.exec()  # two tasks deployed


def test_errors() -> None:
    x = cp.value(1.0)
    y = x + 1
    tg = cp.LocalTarget()

    with pytest.raises(ValueError):
        tg.deploy(cp.Task(y, period=0.01), cp.Task(y, period=0.02))  # two writers
    with pytest.raises(ValueError):
        tg.deploy(cp.Task(y, period=0.00015))  # no multiple of the tick
    with pytest.raises(ValueError):
        cp.Task(y, period=0)

    tg.deploy(cp.Task(y, period=0.01))
    with pytest.raises(ValueError):
        tg.write_value(y, 2.0)  # no input
    with pytest.raises(ValueError):
        tg.exec(cp.Task(x * 2))  # not deployed
    tg.step(0.1)
    with pytest.raises(ValueError):
        tg.step(0.05)  # time goes backwards


if __name__ == "__main__":
    test_single_task_exec()
    test_let_between_tasks()
    test_shared_expression()
    test_arrays_between_tasks()
    test_triggered_task()
    test_errors()
