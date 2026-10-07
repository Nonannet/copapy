"""Tests for the automatic differentiation of array operations (packed tensors
and arrays) against central differences of a pure Python reference."""
import math
from typing import Any, Callable

import pytest

import copapy as cp
from copapy.backend import get_dag_stats

N = 6
DATA = [0.7, -1.3, 2.1, 0.4, -0.6, 1.5, 1.2, 0.8, -2.2, 0.3, 1.9, -0.9]


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def numeric_grad(func: Callable[[list[float]], float], data: list[float], h: float = 1e-6) -> list[float]:
    def shifted(i: int, d: float) -> list[float]:
        return [v + d if j == i else v for j, v in enumerate(data)]
    return [(func(shifted(i, h)) - func(shifted(i, -h))) / (2 * h) for i in range(len(data))]


def tensors(values: list[Any]) -> tuple[Any, Any]:
    """Two packed tensors: operations on them use array stencils"""
    return cp.tensor(values[:N], packed=True), cp.tensor(values[N:], packed=True)


def lists(values: list[float]) -> tuple[list[float], list[float]]:
    return values[:N], values[N:]


def check(f_cp: Callable[[list[Any]], Any], f_ref: Callable[[list[float]], float], data: list[float] = DATA) -> None:
    values = [cp.value(v) for v in data]
    result = f_cp(values)
    grads = cp.grad(result, values)
    out = evaluate(result, *grads)
    assert out[0] == pytest.approx(f_ref(data), rel=1e-4, abs=1e-4)
    assert out[1:] == pytest.approx(numeric_grad(f_ref, data), rel=2e-3, abs=2e-3)


# name, copapy function of two packed tensors, reference function of two lists
ELEMENTWISE: list[tuple[str, Callable[[Any, Any], Any], Callable[[float, float], float]]] = [
    ('add_sub', lambda t, u: t + u - t * 0.5, lambda p, q: p + q - p * 0.5),
    ('mul', lambda t, u: t * u * t, lambda p, q: p * q * p),
    ('div', lambda t, u: t / (u * u + 1.0), lambda p, q: p / (q * q + 1.0)),
    ('rdiv', lambda t, u: 2.0 / (u * u + 0.5) - 3.0 * t, lambda p, q: 2.0 / (q * q + 0.5) - 3.0 * p),
    ('exp_log', lambda t, u: cp.exp(t * 0.3) + cp.log(u * u + 1.0), lambda p, q: math.exp(p * 0.3) + math.log(q * q + 1.0)),
    ('sqrt_abs', lambda t, u: cp.sqrt(cp.abs(t) + 1.0) * cp.abs(u), lambda p, q: math.sqrt(abs(p) + 1.0) * abs(q)),
    ('sin_cos_tan', lambda t, u: cp.sin(t) * cp.cos(u) + cp.tan(t * 0.2), lambda p, q: math.sin(p) * math.cos(q) + math.tan(p * 0.2)),
    ('tanh', lambda t, u: cp.tanh(t * u), lambda p, q: math.tanh(p * q)),
    ('asin_acos_atan', lambda t, u: cp.asin(t * 0.2) + cp.acos(u * 0.3) * cp.atan(t),
     lambda p, q: math.asin(p * 0.2) + math.acos(q * 0.3) * math.atan(p)),
    ('atan2', lambda t, u: cp.atan2(t, u), lambda p, q: math.atan2(p, q)),
    ('pow', lambda t, u: cp.pow(cp.abs(t) + 0.5, u), lambda p, q: (abs(p) + 0.5) ** q),
    ('int_pow', lambda t, u: t ** 3 - u ** 2, lambda p, q: p ** 3 - q ** 2),
    ('min_max', lambda t, u: cp.maximum(t, u) * 2.0 + cp.minimum(t, 0.5), lambda p, q: max(p, q) * 2.0 + min(p, 0.5)),
    ('relu', lambda t, u: cp.nn.relu(t) * u, lambda p, q: max(p, 0.0) * q),
    ('sigmoid', lambda t, u: cp.nn.sigmoid(t - u), lambda p, q: 1.0 / (1.0 + math.exp(q - p))),
]


@pytest.mark.parametrize('name,f_cp,f_ref', ELEMENTWISE, ids=[c[0] for c in ELEMENTWISE])
def test_elementwise(name: str, f_cp: Callable[[Any, Any], Any], f_ref: Callable[[float, float], float]) -> None:
    weights = [0.5 + 0.25 * i for i in range(N)]  # different gradient for each element
    check(lambda v: (f_cp(*tensors(v)) * cp.tensor(weights)).sum(),
          lambda d: sum(f_ref(p, q) * w for p, q, w in zip(*lists(d), weights)))


def test_scalar_operands() -> None:
    """A scalar used for each element gets the sum of the element gradients"""
    def f_cp(v: list[Any]) -> Any:
        t, s = cp.tensor(v[:N], packed=True), v[N]
        return (t * s + s / (t * t + 2.0) - s + cp.pow(s * s + 1.0, t) + cp.atan2(s, t)).sum()

    def f_ref(d: list[float]) -> float:
        s = d[N]
        return sum(p * s + s / (p * p + 2.0) - s + (s * s + 1.0) ** p + math.atan2(s, p) for p in d[:N])

    check(f_cp, f_ref, DATA[:N + 1])


def test_reductions_and_dot() -> None:
    check(lambda v: cp.max(tensors(v)[0]) * 2.0 - cp.min(tensors(v)[1]), lambda d: max(d[:N]) * 2.0 - min(d[N:]))
    check(lambda v: tensors(v)[0] @ tensors(v)[1], lambda d: sum(p * q for p, q in zip(*lists(d))))
    check(lambda v: (tensors(v)[0] * 3.0).mean() + cp.sin(tensors(v)[1]).sum() ** 2,
          lambda d: sum(p * 3.0 for p in d[:N]) / N + sum(math.sin(q) for q in d[N:]) ** 2)


def test_matrix_products() -> None:
    def matvec(v: list[Any]) -> Any:
        w, x = cp.tensor(v[:8], (2, 4), packed=True), cp.tensor(v[8:], packed=True)
        y = w @ x
        return (y * y).sum()

    def matvec_ref(d: list[float]) -> float:
        return sum(sum(d[r * 4 + c] * d[8 + c] for c in range(4)) ** 2 for r in range(2))

    def matmul(v: list[Any]) -> Any:
        a, b = cp.tensor(v[:6], (2, 3), packed=True), cp.tensor(v[6:], (3, 2), packed=True)
        return (cp.tanh(a @ b) * cp.tensor([[1.0, 2.0], [3.0, 4.0]])).sum()

    def matmul_ref(d: list[float]) -> float:
        return sum(math.tanh(sum(d[r * 3 + k] * d[6 + k * 2 + c] for k in range(3))) * (r * 2 + c + 1.0)
                   for r in range(2) for c in range(2))

    check(matvec, matvec_ref)
    check(matmul, matmul_ref)


def test_single_elements() -> None:
    """Elements of an array result used by scalar operations"""
    def f_cp(v: list[Any]) -> Any:
        r = cp.sin(cp.tensor(v[:N], packed=True))  # only available as array
        assert r._packed_array() is not None
        return r.values[0] * r.values[2] + r.values[2] * v[N] + cp.exp(r.values[5]) + r.sum() * 0.5

    def f_ref(d: list[float]) -> float:
        r = [math.sin(p) for p in d[:N]]
        return r[0] * r[2] + r[2] * d[N] + math.exp(r[5]) + sum(r) * 0.5

    check(f_cp, f_ref, DATA[:N + 1])


def test_grad_of_arrays() -> None:
    data, weight = [0.5, -1.5, 2.0, 3.0], cp.value(0.75)
    a = cp.array(data)
    result = (a * a * weight + cp.sin(a)).sum()
    da = cp.grad(result, a)
    dw = cp.grad(result, weight)
    assert isinstance(da, cp.array) and da.shape == (4,)
    out_a, out_w = evaluate(da, dw)
    assert out_a == pytest.approx([2 * p * 0.75 + math.cos(p) for p in data], rel=1e-5)
    assert out_w == pytest.approx(sum(p * p for p in data), rel=1e-5)

    # Arrays of values and tensors
    values = [cp.value(p) for p in data]
    packed = cp.array(values)
    grads = cp.grad((packed * packed).sum(), values)
    matrix = cp.tensor([[cp.value(p) for p in data[:2]], [cp.value(p) for p in data[2:]]], packed=True)
    dm = cp.grad((matrix * matrix).sum(), matrix)
    assert isinstance(dm, cp.tensor) and dm.shape == (2, 2)
    out_values, out_m = evaluate(grads, dm)
    assert out_values == pytest.approx([2 * p for p in data])
    assert list(out_m.values) == pytest.approx([2 * p for p in data])


def test_gradient_code_size() -> None:
    """The gradient of array operations is computed by array stencils"""
    values = [cp.value(float(i % 7) * 0.1) for i in range(200)]
    t = cp.tensor(values)  # packed by the pack threshold
    grads = cp.grad((cp.sin(t) * t).sum(), values)
    stats = get_dag_stats([sum(grads[1:], grads[0]).net])
    assert stats.get('cos_floatarr') == 1
    assert 'cos_float' not in stats and 'mul_float_float' not in stats


def test_unsupported_operations() -> None:
    t = cp.tensor([[cp.value(1.0), cp.value(2.0)], [cp.value(3.0), cp.value(4.0)]], packed=True)
    with pytest.raises(NotImplementedError):
        cp.grad(((t * 2.0).T * t).sum(), t)  # strided copy of an array result
    x = cp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    with pytest.raises(NotImplementedError):
        cp.grad(cp.nn.conv2d(x, cp.array([[[[1.0, 2.0], [3.0, 4.0]]]])).sum(), x)
