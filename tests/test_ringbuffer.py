"""Tests for cp.ringbuffer on the local target."""
from typing import Any

import pytest

import copapy as cp
import copapy._tasks as tasks


def approx(v: object) -> object:
    return pytest.approx(v, rel=1e-6, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def run_sequence(outputs: Any, u: cp.value[Any], inputs: list[Any], period: float = 0.01) -> list[Any]:
    """Results of the periodic task computing outputs for each input value"""
    tg = cp.LocalTarget(tick=period)
    tg.deploy(cp.Task(outputs, period=period))
    results: list[Any] = []
    for k, x in enumerate(inputs):
        tg.write_value(u, x)
        tg.step((k + 1) * period)
        results.append(tg.read_value(outputs))
    return results


def history(xs: list[Any], k: int, n: int, init: Any) -> list[Any]:
    """Reference: values of the last n runs in time order after run k"""
    past = [init] * n + xs[:k + 1]
    return past[-n:]


def test_scalar() -> None:
    u = cp.value(0.0)
    n = 4
    rb = cp.ringbuffer(u, n, init=-1.0)
    lag = cp.value(2)
    outputs = [rb.data, rb.index, rb[0], rb[1], rb[lag], rb.oldest, rb.window()]
    xs = [float(i + 1) for i in range(10)]
    results = run_sequence(outputs, u, xs)
    for k, (data, index, r0, r1, rl, oldest, window) in enumerate(results):
        expected = history(xs, k, n, -1.0)
        assert index == k % n
        assert data[index] == xs[k]
        assert window == expected
        assert [r0, r1, rl, oldest] == [expected[-1], expected[-2], expected[-3], expected[0]]
        # Storage order: window rotated by the slot of the oldest value
        start = (index + 1) % n
        assert data[start:] + data[:start] == expected


def test_runtime_lag_is_limited() -> None:
    u = cp.value(0.0)
    rb = cp.ringbuffer(u, 3)
    lags = [0, 1, 2, -5, 10]
    outputs = [rb[cp.value(lag)] for lag in lags]
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = run_sequence(outputs, u, xs)
    for k, result in enumerate(results):
        expected = history(xs, k, 3, 0.0)
        assert result == [expected[-1 - min(max(lag, 0), 2)] for lag in lags]


def test_types() -> None:
    u = cp.value(0)
    rb_int = cp.ringbuffer(u * 3, 3, init=7)
    rb_bool = cp.ringbuffer(u > 1, 2)
    results = run_sequence([rb_int.window(), rb_bool.window(), rb_int[2]], u, [1, 2, 3])
    assert results == [[[7, 7, 3], [False, False], 7], [[7, 3, 6], [False, True], 7], [[3, 6, 9], [True, True], 3]]
    assert rb_int.data.dtype == 'int' and rb_bool.data.dtype == 'bool'
    with pytest.raises(TypeError, match="no integer"):
        cp.ringbuffer(u, 2, init=0.5)
    assert cp.ringbuffer(cp.value(1.0), 2, init=1).data.dtype == 'float'


def test_vector_and_tensor_samples() -> None:
    u = cp.value(0.0)
    v = cp.vector([u, u * 2, u * 3])
    t = cp.tensor([[u, u + 1.0], [u - 1.0, u]])
    q = cp.quaternion(1.0, u, 0.0, 0.0)
    rv = cp.ringbuffer(v, 3, init=[9.0, 8.0, 7.0])
    rt = cp.ringbuffer(t, 2)
    rq = cp.ringbuffer(q, 2, init=cp.quaternion.identity())
    outputs = [rv[1], rv.window(), rt.oldest, rq[1], rv.data]
    results = run_sequence(outputs, u, [1.0, 2.0, 3.0, 4.0])

    assert isinstance(results[0][0], cp.vector)
    assert isinstance(results[0][2], cp.tensor)
    assert isinstance(results[0][3], cp.quaternion)
    assert results[0][0].values == (9.0, 8.0, 7.0)
    assert results[0][1] == [[9.0, 8.0, 7.0], [9.0, 8.0, 7.0], [1.0, 2.0, 3.0]]
    assert results[0][3].values == (1.0, 0.0, 0.0, 0.0)
    assert results[3][0].values == (3.0, 6.0, 9.0)
    assert results[3][1] == [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]
    assert results[3][2].values == (3.0, 4.0, 2.0, 3.0)
    assert results[3][3].values == (1.0, 3.0, 0.0, 0.0)
    assert results[3][4] == [[4.0, 8.0, 12.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]


def test_array_operations_on_buffer() -> None:
    # data and window are standard arrays for the array stencils
    u = cp.value(0.0)
    n = 50
    rb = cp.ringbuffer(u, n)
    taps = cp.array([float(i % 7) for i in range(n)])
    outputs = [cp.filters.mean(cp.vector(rb.data)), (rb.window() * taps).sum(), rb.data.max()]
    xs = [float((i * 13) % 17) for i in range(80)]
    results = run_sequence(outputs, u, xs)
    for k, (mean, fir, peak) in enumerate(results):
        expected = history(xs, k, n, 0.0)
        assert mean == approx(sum(expected) / n)
        assert fir == approx(sum(e * float(i % 7) for i, e in enumerate(expected)))
        assert peak == max(expected)


def test_moving_sum() -> None:
    # O(1) moving sum: add the entering value and subtract the leaving value
    u = cp.value(0.0)
    n = 5
    rb = cp.ringbuffer(u, n + 1)
    total = cp.feedback(lambda s: s + u - rb.oldest, init=0.0)
    xs = [float(i * i % 11) for i in range(20)]
    results = run_sequence(total, u, xs)
    assert results == approx([sum(history(xs, k, n, 0.0)) for k in range(len(xs))])


def test_no_copy_for_update(monkeypatch: pytest.MonkeyPatch) -> None:
    # The buffer is written in place: the update after each run copies only the index
    copies: list[list[int]] = []
    original = tasks._sort_state_copies

    def recorder(c: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        result = original(c)
        copies.append([size for _, _, size in result])
        return result

    monkeypatch.setattr(tasks, '_sort_state_copies', recorder)
    rb = cp.ringbuffer(cp.value(0.0), 1000)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(cp.Task([rb.oldest, rb.window()], period=0.01))
    assert copies == [[4]]


def test_single_slot_and_constants() -> None:
    u = cp.value(0.0)
    rb = cp.ringbuffer(u, 1)
    const = cp.ringbuffer(2.5, 3, init=1.0)
    results = run_sequence([rb.window(), rb.oldest, const.window()], u, [3.0, 4.0])
    assert results == [[[3.0], 3.0, [1.0, 1.0, 2.5]], [[4.0], 4.0, [1.0, 2.5, 2.5]]]


def test_tasks() -> None:
    # Buffer used by another task: data and index are outputs of the task computing it
    u = cp.value(0.0)
    rb = cp.ringbuffer(u * 2, 3)
    window = rb.window()
    fast = cp.Task([rb.data, rb.index], period=0.01)
    slow = cp.Task(window, period=0.02)
    tg = cp.LocalTarget(tick=0.01)
    tg.deploy(fast, slow)
    for k in range(4):
        tg.write_value(u, float(k + 1))
        tg.step((k + 1) * 0.01)
    # Window of the buffer published by the fast task at the release of the slow task
    assert tg.read_value(window) == [0.0, 2.0, 4.0]

    # Triggered task: one value per exec, deployment resets the buffer
    rb2 = cp.ringbuffer(u, 2)
    tg2 = cp.LocalTarget()
    tg2.deploy(rb2.data)
    for x in [1.0, 2.0, 3.0]:
        tg2.write_value(u, x)
        tg2.exec()
    assert tg2.read_value(rb2.data) == [3.0, 2.0]
    tg2.deploy(rb2.data)
    tg2.write_value(u, 5.0)
    tg2.exec()
    assert tg2.read_value(rb2.data) == [5.0, 0.0]


def test_errors() -> None:
    u = cp.value(0.0)
    rb = cp.ringbuffer(u, 3)
    with pytest.raises(ValueError, match="require tasks"):
        cp.Target().compile(rb.data)
    with pytest.raises(ValueError, match="positive"):
        cp.ringbuffer(u, 0)
    with pytest.raises(IndexError):
        rb[3]
    with pytest.raises(TypeError, match="int value"):
        rb[u]  # type: ignore[index]
    with pytest.raises(ValueError, match="initial values"):
        cp.ringbuffer(cp.vector([u, u]), 2, init=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="numbers"):
        cp.ringbuffer(u, 2, init=u)
    with pytest.raises(TypeError, match="byte"):
        cp.ringbuffer(cp.array([1, 2], 'byte'), 2)
    with pytest.raises(TypeError, match="Unsupported"):
        cp.ringbuffer("x", 2)
