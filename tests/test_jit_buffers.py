"""Tests for the exchange of array data with the target without conversion of the
elements: Target.buffer, map_to_buffer, map_to_numpy and buffer arguments of jit."""
import array as pyarray
from typing import Any

import pytest

import copapy as cp


def float_view(values: list[float], shape: tuple[int, ...]) -> Any:
    data: Any = memoryview(pyarray.array('f', values)).cast('B')
    return data.cast('f', shape)


def test_target_buffer() -> None:
    img = cp.array([[1, 2, 3], [4, 5, 6]], 'byte')
    scaled = cp.to_float(img) * 0.5
    flags = img > 2
    tg = cp.Target()
    tg.compile(scaled, flags)
    tg.run()
    view: Any = tg.buffer(img)
    result: Any = tg.buffer(scaled)
    assert (view.format, view.shape, result.format, result.shape) == ('B', (2, 3), 'f', (2, 3))
    assert tg.buffer(flags).format == 'i' and list(tg.buffer(flags).cast('B').cast('i')) == [0, 0, 1, 1, 1, 1]
    assert view.tolist() == [[1, 2, 3], [4, 5, 6]] and result.tolist() == [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]
    # Writing to the buffer changes the array for the next run, the result buffer is updated
    view.cast('B')[:] = bytes([10, 20, 30, 40, 50, 60])
    tg.run()
    assert result.tolist() == [[5.0, 10.0, 15.0], [20.0, 25.0, 30.0]]
    assert tg.read_value(img) == [[10, 20, 30], [40, 50, 60]]
    with pytest.raises(ValueError):
        tg.buffer(cp.array([1.0, 2.0]))  # Not compiled


def test_no_compile_with_buffers_in_use() -> None:
    """The memory of the target is not replaced while a buffer refers to it"""
    a = cp.array([cp.value(1.0), cp.value(2.0)])
    tg = cp.Target()
    tg.compile(a * 2.0)
    view = tg.buffer(a)
    part = view.cast('B')[2:6]  # Views of the buffer keep it in use as well
    with pytest.raises(RuntimeError):
        tg.compile(a * 3.0)
    view.release()
    with pytest.raises(RuntimeError):
        tg.compile(a * 3.0)
    part.release()
    result = a * 3.0
    tg.compile(result)
    tg.run()
    assert tg.buffer(result).tolist() == [3.0, 6.0]


def test_read_mapped_and_nested() -> None:
    a = cp.array([cp.value(1.0), cp.value(2.0), cp.value(3.0)])
    doubled = a * 2.0
    structure: Any = (cp.map_to_buffer(doubled), [a.sum(), (doubled, 7)], {'max': cp.max(a), 'vec': cp.vector([cp.value(1), cp.value(2)])})
    tg = cp.Target()
    tg.compile(structure)
    tg.run()
    view, items, named = tg.read_value(structure)
    assert isinstance(view, memoryview) and view.tolist() == [2.0, 4.0, 6.0]
    assert items == [6.0, [[2.0, 4.0, 6.0], 7]]
    assert named['max'] == 3.0 and named['vec'].values == (1, 2)
    # The mark is not passed on to results of operations
    assert cp.map_to_buffer(doubled)._export == 'buffer' and (cp.map_to_buffer(doubled) + 1)._export is None
    with pytest.raises(TypeError):
        cp.map_to_buffer(cp.value(1.0))  # type: ignore[arg-type]


def test_map_vector_and_tensor() -> None:
    @cp.jit
    def func(x: Any, y: Any) -> Any:
        vec = cp.vector([x, y, x + y]) * 2.0
        packed = cp.vector([x, y], packed=True) + 1.0
        weights: Any = cp.tensor([[1.0, 2.0], [3.0, 4.0]])
        mat = cp.tensor([[x, y], [y, x]]) @ weights
        return cp.map_to_buffer(vec), cp.map_to_buffer(packed), cp.map_to_buffer(mat), cp.map_to_buffer(cp.vector([x, 2]))

    vec, packed, mat, ints = func(1.5, 2.0)
    assert vec.tolist() == [3.0, 4.0, 7.0] and packed.tolist() == [2.5, 3.0]
    assert mat.shape == (2, 2) and mat.tolist() == [[7.5, 11.0], [6.5, 10.0]]
    assert ints.format == 'f' and ints.tolist() == [1.5, 2.0]


def test_jit_bytes() -> None:
    @cp.jit
    def scale(img: Any, gain: Any) -> Any:
        x = cp.to_float(img) * gain
        return cp.map_to_buffer(cp.to_byte(x)), x.max(), img + 1

    out, peak, plus = scale(bytes([1, 2, 3, 250]), 2.0)
    assert bytes(out) == bytes([2, 4, 6, 255]) and peak == 500.0 and plus == [2, 3, 4, 251]
    # Same types and shapes: the program is reused, the buffer shows the new result
    out2, peak, _ = scale(bytearray([8, 8, 8, 8]), 0.5)
    assert bytes(out2) == bytes([4, 4, 4, 4]) and bytes(out) == bytes(out2) and peak == 4.0
    # A new program for another size or type of an argument
    assert bytes(scale(bytes(6), 2.0)[0]) == bytes(6)
    assert scale(bytes([1, 2, 3, 250]), 2)[1] == 500.0
    assert bytes(out) == bytes([4, 4, 4, 4])


def test_jit_typed_buffers() -> None:
    @cp.jit
    def func(a: Any, idx: Any) -> Any:
        return cp.map_to_buffer(a.T * 2.0), a.sum(), cp.map_to_buffer(idx + 1), a

    data = float_view([1, 2, 3, 4, 5, 6], (2, 3))
    indices = memoryview(pyarray.array('i', [5, -6, 7]))
    transposed, total, plus, same = func(data, indices)
    assert transposed.shape == (3, 2) and transposed.tolist() == [[2.0, 8.0], [4.0, 10.0], [6.0, 12.0]]
    assert total == 21.0 and plus.format == 'i' and plus.tolist() == [6, -5, 8]
    assert same == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # Not contiguous: copied in the order of the elements
    assert func(data, indices[::-1])[2].tolist() == [8, -5, 6]
    with pytest.raises(TypeError):
        func(memoryview(pyarray.array('d', [1.0] * 6)), indices)


def test_jit_numbers_and_sequences() -> None:
    @cp.jit
    def func(x: Any, values: Any) -> Any:
        return x * 2 + sum(values)

    assert func(1.5, [1, 2]) == 6.0
    assert func(2.5, [3, 4]) == 12.0
    assert func(2, (1, 2, 3)) == 10  # Other types and length: compiled again


def test_jit_numpy() -> None:
    np = pytest.importorskip('numpy')

    @cp.jit
    def func(img: Any, weights: Any) -> Any:
        x = cp.to_float(img) * weights
        return cp.map_to_numpy(cp.to_byte(x)), cp.map_to_numpy(x, copy=True), cp.map_to_numpy(img > 100)

    frame = np.array([[0, 50, 100], [150, 200, 250]], dtype=np.uint8)
    weights = np.array([[1.0, 2.0, 0.5]] * 2)  # float64: converted
    out, product, flags = func(frame, weights)
    assert (out.dtype, product.dtype, flags.dtype) == (np.uint8, np.float32, np.int32)
    assert out.shape == (2, 3) and out.tolist() == [[0, 100, 50], [150, 255, 125]]
    assert product.tolist() == [[0.0, 100.0, 50.0], [150.0, 400.0, 125.0]]
    assert flags.tolist() == [[0, 0, 0], [1, 1, 1]]
    # The view is updated by the next call, the copy is not
    out2, product2, _ = func(frame[::-1].T.copy().T, weights)
    assert out.tolist() == out2.tolist() == [[150, 255, 125], [0, 100, 50]]
    assert product.tolist() == [[0.0, 100.0, 50.0], [150.0, 400.0, 125.0]] and product2[0, 1] == 400.0
    # Not contiguous and other int types
    assert func(frame[:, ::-1], weights)[0].tolist() == [[100, 100, 0], [250, 255, 75]]
    assert func(frame.astype(np.int64), weights)[1].tolist() == product.tolist()
    with pytest.raises(TypeError):
        func(frame > 0, weights)
