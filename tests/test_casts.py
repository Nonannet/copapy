"""Tests for casting between float, int and bool: numbers, values, vectors, tensors and arrays"""
from typing import Any

import pytest

import copapy as cp

PY_TYPES: dict[str, Any] = {'float': float, 'int': int, 'bool': bool}
DATA: dict[str, list[Any]] = {
    'float': [2.7, -2.7, 0.0, 0.5, -0.5, 100.25],
    'int': [3, -3, 0, 1, -1, 1000],
    'bool': [True, False, False, True, True, False],
}


def evaluate(*exprs: Any) -> list[Any]:
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def check(result: list[Any], data: list[Any], dtype: str) -> None:
    expected = [PY_TYPES[dtype](x) for x in data]
    assert result == pytest.approx(expected)
    assert all(type(r) is PY_TYPES[dtype] for r in result)


@pytest.mark.parametrize('dtype', ['float', 'int', 'bool'])
@pytest.mark.parametrize('src', ['float', 'int', 'bool'])
def test_numbers(src: str, dtype: str) -> None:
    check([cp.cast(x, dtype) for x in DATA[src]], DATA[src], dtype)
    check([cp.cast(x, PY_TYPES[dtype]) for x in DATA[src]], DATA[src], dtype)


@pytest.mark.parametrize('dtype', ['float', 'int', 'bool'])
@pytest.mark.parametrize('src', ['float', 'int', 'bool'])
def test_values(src: str, dtype: str) -> None:
    values = [cp.value(x) for x in DATA[src]]
    casted = [cp.cast(v, dtype) for v in values]
    assert all(c.dtype == dtype for c in casted)
    if src == dtype:
        assert all(c is v for c, v in zip(casted, values))
    check(evaluate(*casted), DATA[src], dtype)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('dtype', ['float', 'int', 'bool'])
@pytest.mark.parametrize('src', ['float', 'int', 'bool'])
def test_vector_tensor(src: str, dtype: str, packed: bool) -> None:
    data = DATA[src]
    v = cp.vector((cp.value(x) for x in data), packed=packed)
    t = cp.tensor([cp.value(x) for x in data], (2, 3), packed=packed)

    cv, ct = cp.cast(v, dtype), cp.cast(t, dtype)
    assert isinstance(cv, cp.vector) and isinstance(ct, cp.tensor)
    assert cv.dtype == dtype and ct.dtype == dtype and ct.shape == (2, 3)
    if src == dtype:
        assert cv is v and ct is t

    out_v, out_t = evaluate(cv, ct)
    check(list(out_v.values), data, dtype)
    check(list(out_t.values), data, dtype)


@pytest.mark.parametrize('dtype', ['float', 'int', 'bool'])
@pytest.mark.parametrize('src', ['float', 'int', 'bool'])
def test_array(src: str, dtype: str) -> None:
    data = DATA[src]
    a = cp.array([data[:3], data[3:]])
    ca = cp.cast(a, dtype)
    assert isinstance(ca, cp.array) and ca.dtype == dtype and ca.shape == (2, 3)

    tg = cp.Target()
    tg.compile(ca)
    tg.run()
    result = tg.read_value(ca)
    check(result[0] + result[1], data, dtype)


@pytest.mark.parametrize('dtype', ['float', 'int', 'bool'])
def test_constants(dtype: str) -> None:
    """Vectors and tensors of numbers are converted at trace time"""
    v = cp.cast(cp.vector(DATA['float']), dtype)
    t = cp.cast(cp.tensor(DATA['int'], (3, 2)), dtype)
    check(list(v.values), DATA['float'], dtype)
    check(list(t.values), DATA['int'], dtype)
    assert isinstance(v, cp.vector) and isinstance(t, cp.tensor) and t.shape == (3, 2)


def test_mixed_elements() -> None:
    v = cp.vector([cp.value(2.5), 7.9, cp.value(-1.5)])
    c = cp.to_int(v)
    assert c.dtype == 'int' and c.values[1] == 7
    out, = evaluate(c)
    assert list(out.values) == [2, 7, -1]


def test_quaternion() -> None:
    q = cp.quaternion(1, cp.value(2), 0, 0)
    c = cp.to_float(q)
    assert isinstance(c, cp.quaternion)
    assert type(c.values[0]) is float and c.values[1].dtype == 'float'


def test_helpers() -> None:
    v = cp.value(-3.9)
    f, i, b = cp.to_float(cp.value(4)), cp.to_int(v), cp.to_bool(v)
    assert (f.dtype, i.dtype, b.dtype) == ('float', 'int', 'bool')
    assert evaluate(f, i, b) == [4.0, -3, True]
    assert (cp.to_float(2), cp.to_int(-2.9), cp.to_bool(0.0)) == (2.0, -2, False)


def test_generated_code() -> None:
    """Bool to int generates no code, a packed tensor is converted by one array stencil"""
    b = cp.value(1.5) > 1
    assert cp.to_int(b).net is b.net

    t = cp.tensor([cp.value(float(i)) for i in range(8)], (2, 4), packed=True)
    arr = cp.to_int(t)._packed_array()
    assert arr is not None
    assert arr.net.source.name == 'int_floatarr'


def test_grad() -> None:
    """The derivative of a conversion to float is 1, to int it is 0"""
    n, x = cp.value(3), cp.value(1.5)
    dn, dx = evaluate(cp.grad(cp.to_float(n) * 2.5, n), cp.grad(x * 2.5 + cp.to_int(x), x))
    assert dn == pytest.approx(2.5) and dx == pytest.approx(2.5)


def test_errors() -> None:
    with pytest.raises(ValueError):
        cp.cast(1, 'double')
    with pytest.raises(ValueError):
        cp.cast(1, str)
    with pytest.raises(TypeError):
        cp.cast('1', 'int')
