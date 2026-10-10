"""Tests for byte arrays: arrays storing numbers from 0 to 255 in one byte per
element (e.g. image data), converted to float or int for operations."""
from typing import Any

import pytest

import copapy as cp

IMAGE = [[0, 10, 255, 3], [7, 8, 9, 200], [128, 1, 2, 64]]
FLAT = [v for row in IMAGE for v in row]


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def flat(data: Any) -> list[Any]:
    return [x for d in data for x in flat(d)] if isinstance(data, list) else [data]


def op_names(result: Any) -> set[str]:
    """Names of all operations the result depends on"""
    seen: set[int] = set()
    names: set[str] = set()
    stack = [result.net]
    while stack:
        net = stack.pop()
        if id(net) not in seen:
            seen.add(id(net))
            names.add(net.source.name)
            stack += list(net.source.args)
    return names


def test_create_and_read() -> None:
    img = cp.array(IMAGE, 'byte')
    assert img.dtype == 'byte' and img.shape == (3, 4) and img.size == 12
    tg = cp.Target()
    assert tg.read_value(img) == IMAGE  # Not compiled: read from the constants
    tg.compile(img.sum())
    tg.run()
    assert tg.read_value(img) == IMAGE
    assert tg._values[img.net][1] == 12  # One byte per element


def test_write_at_runtime() -> None:
    """A new frame is written to the array for each run"""
    img = cp.array(IMAGE, 'byte')
    result = cp.to_float(img).sum()
    tg = cp.Target()
    tg.compile(result)
    for frame in (IMAGE, [[255] * 4] * 3, [[0, 1, 2, 3]] * 3):
        tg.write_value(img, frame)
        tg.run()
        assert tg.read_value(result) == sum(flat(frame))
        assert tg.read_value(img) == frame


def test_errors() -> None:
    for values in ([0, 256], [-1, 5], [0.5, 1.0], [cp.value(1), 2]):
        with pytest.raises(ValueError):
            cp.array(values, 'byte')
    with pytest.raises(ValueError):
        cp.cast(cp.value(5), 'byte')
    with pytest.raises(TypeError):
        cp.to_bytes(cp.value(5))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cp.cast(cp.vector([cp.value(5), cp.value(6)]), 'byte')


def test_casts_from_bytes() -> None:
    img = cp.array(IMAGE, 'byte')
    as_float, as_int, as_bool = cp.to_float(img), cp.to_int(img), cp.to_bool(img)
    assert (as_float.dtype, as_int.dtype, as_bool.dtype) == ('float', 'int', 'bool')
    # Converted directly, not by an int array in between
    assert as_float.net.source.name == 'float_bytearr' and as_int.net.source.name == 'int_bytearr'
    assert cp.cast(img, 'byte') is img
    res = evaluate(as_float, as_int, as_bool)
    assert res[0] == IMAGE and all(isinstance(v, float) for v in flat(res[0]))
    assert res[1] == IMAGE and all(isinstance(v, int) for v in flat(res[1]))
    assert flat(res[2]) == [v != 0 for v in FLAT]


def test_cast_to_bytes_saturates() -> None:
    ints = [-1000, -1, 0, 1, 254, 255, 256, 100000]
    floats = [-3.5, -0.1, 0.0, 0.9, 17.99, 254.5, 255.0, 255.9, 1e9, float('inf'), float('-inf'), float('nan')]
    from_int = cp.cast(cp.array([cp.value(v) for v in ints]), 'byte')
    from_float = cp.cast(cp.array([cp.value(v) for v in floats]), 'byte')
    from_bool = cp.cast(cp.array([cp.value(v) for v in ints]) > 0, 'byte')
    assert from_int.dtype == from_float.dtype == from_bool.dtype == 'byte'
    assert cp.to_bytes(cp.array([300, 5])).dtype == 'byte'
    res = evaluate(from_int, from_float, from_bool, cp.to_bytes(cp.array([cp.value(300), cp.value(5)])))
    assert res[3] == [255, 5]
    assert res[0] == [0, 0, 0, 1, 254, 255, 255, 255]
    assert res[1] == [0, 0, 0, 0, 17, 254, 255, 255, 255, 255, 0, 0]
    assert res[2] == [int(v > 0) for v in ints]


def test_operations_convert_to_int() -> None:
    img = cp.array(IMAGE, 'byte')
    results: list[Any] = [img + 1, img + img, 300 - img, img * cp.value(2), -img, img > 9, img == 255,
                          cp.minimum(img, 100), img.sum(), img.min(), img.max(), img.flatten().sort(),
                          img.flatten().argsort(), cp.abs(img)]
    assert all(r.dtype in ('int', 'bool') for r in results)
    assert 'float_bytearr' not in set().union(*(op_names(r) for r in results))
    res = evaluate(*results)
    refs: list[Any] = [[v + 1 for v in FLAT], [2 * v for v in FLAT], [300 - v for v in FLAT], [2 * v for v in FLAT],
                       [-v for v in FLAT], [v > 9 for v in FLAT], [v == 255 for v in FLAT], [min(v, 100) for v in FLAT],
                       sum(FLAT), min(FLAT), max(FLAT), sorted(FLAT), sorted(range(12), key=lambda i: FLAT[i]), FLAT]
    assert [flat(r) if isinstance(r, list) else r for r in res] == refs


def test_operations_convert_to_float() -> None:
    """With a float operand the bytes are converted to float directly"""
    img = cp.array(IMAGE, 'byte')
    weights = cp.array([[0.5] * 4] * 3)
    results: list[Any] = [img * (1.0 / 255.0), img - cp.value(0.5), 2.5 + img, img * weights, img / 2,
                          img.flatten().dot(weights.flatten()), cp.sqrt(img)]
    assert all(r.dtype == 'float' for r in results)
    assert all('int_bytearr' not in op_names(r) for r in results[:4] + results[5:6])
    res = evaluate(*results)
    refs: list[Any] = [[v / 255.0 for v in FLAT], [v - 0.5 for v in FLAT], [2.5 + v for v in FLAT],
                       [0.5 * v for v in FLAT], [v / 2 for v in FLAT], 0.5 * sum(FLAT), [v ** 0.5 for v in FLAT]]
    for r, ref in zip(res, refs):
        assert (flat(r) if isinstance(r, list) else r) == pytest.approx(ref, rel=1e-6)


def test_views_stay_bytes() -> None:
    img = cp.array(IMAGE, 'byte')
    crop, row, transposed, reshaped = img[1:, 1:3], img[2], img.T, img.reshape(2, 6)
    broadcasted = cp.array([1, 2, 3, 4], 'byte').broadcast_to((2, 4))
    views = [crop, row, transposed, reshaped, broadcasted]
    assert all(v.dtype == 'byte' for v in views)
    assert crop.net.source.name == 'copy8_arr' and reshaped.net is img.net
    res = evaluate(*views)
    assert res == [[r[1:3] for r in IMAGE[1:]], IMAGE[2], [list(c) for c in zip(*IMAGE)],
                   [FLAT[:6], FLAT[6:]], [[1, 2, 3, 4]] * 2]


def test_elements_are_int_values() -> None:
    img = cp.array(IMAGE, 'byte')
    index = cp.value(1)
    elements = [img[0, 2], img[2, 0], img[index, 3], img.flatten()[-1], cp.array([42], 'byte')[0]]
    assert all(isinstance(e, cp.value) and e.dtype == 'int' for e in elements)
    assert evaluate(*elements, img[0, 2] + 1.5) == [255, 128, 200, 64, 42, 256.5]


def test_convolution() -> None:
    """A byte image is converted to float once and convolved by the float stencil"""
    image = [IMAGE]  # One channel
    weight = [[[[1.0, -1.0], [0.5, 2.0]]]]
    result = cp.nn.conv2d(cp.array(image, 'byte'), weight)
    names = op_names(result)
    assert 'float_bytearr' in names and 'int_bytearr' not in names
    assert not any(n.startswith('mul') for n in names)
    res, ref = evaluate(result, cp.nn.conv2d(cp.array(image, 'float'), weight))
    assert res == ref
    assert res[0][0][0] == pytest.approx(0 - 10 + 0.5 * 7 + 2.0 * 8)


def test_grad() -> None:
    img = cp.array(IMAGE, 'byte')
    gain = cp.value(0.5)
    loss = ((img * gain) ** 2).sum()
    assert evaluate(cp.grad(loss, gain))[0] == pytest.approx(2 * 0.5 * sum(v * v for v in FLAT))
