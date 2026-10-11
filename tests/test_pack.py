"""Tests for pack and unpack: int and float values converted to and from
their bytes, with the format strings of the struct module."""
import struct
from typing import Any

import pytest

import copapy as cp

FMT = 'bBhHiIf'
NUMBERS = (-5, 200, -1234, 54321, -123456789, 0x12345678, 3.25)


def evaluate(*exprs: Any) -> list[Any]:
    """Compile and run the expressions, return their results."""
    tg = cp.Target()
    tg.compile(*exprs)
    tg.run()
    return [tg.read_value(e) for e in exprs]


def variables(numbers: Any) -> list[Any]:
    return [cp.value(n) for n in numbers]


def test_numbers_at_trace_time() -> None:
    for order in ('<', '>'):
        data = cp.pack(order + FMT, *NUMBERS)
        assert data == list(struct.pack(order + FMT, *NUMBERS))
        assert cp.unpack(order + FMT, data) == NUMBERS
    assert cp.pack('2B x h', 1, 2, 3) == [1, 2, 0, 3, 0]  # Default is little-endian
    assert cp.unpack('2B x h', [1, 2, 99, 3, 0]) == (1, 2, 3)


def test_pack_values() -> None:
    for order in ('<', '>'):
        data = cp.pack(order + FMT, *variables(NUMBERS))
        assert all(isinstance(b, cp.value) for b in data)
        assert evaluate(data)[0] == list(struct.pack(order + FMT, *NUMBERS))


def test_unpack_values() -> None:
    for order in ('<', '>'):
        data = variables(struct.pack(order + FMT, *NUMBERS))
        assert evaluate(cp.unpack(order + FMT, data))[0] == list(NUMBERS)


def test_roundtrip() -> None:
    numbers = (-1, 255, -32768, 65535, -2**31, 2**31 - 1, -0.1)
    result = evaluate(cp.unpack('>' + FMT, cp.pack('>' + FMT, *variables(numbers))))[0]
    assert result[:-1] == list(numbers[:-1])
    assert result[-1] == struct.unpack('f', struct.pack('f', numbers[-1]))[0]


def test_special_floats() -> None:
    numbers = (float('inf'), -0.0, 1e-45)
    data, = evaluate(cp.pack('3f', *variables(numbers)))
    assert data == list(struct.pack('<3f', *numbers))
    (nan,), = evaluate(cp.unpack('f', variables(struct.pack('<f', float('nan')))))
    assert nan != nan


def test_truncation_and_wrap() -> None:
    assert cp.pack('Bh', 0x1FF, -1) == [0xFF, 0xFF, 0xFF]
    assert evaluate(cp.pack('Bh', cp.value(0x1FF), cp.value(0x12345)))[0] == [0xFF, 0x45, 0x23]
    assert evaluate(cp.unpack('I', variables([0, 0, 0, 0x80])))[0] == [-2**31]


def test_mixed_and_offset() -> None:
    data: list[cp.value[int] | int] = [99, 0x34, cp.value(0x12), 0xFF, 0xFF, 0xFF, cp.value(0xFF)]
    assert evaluate(cp.unpack('<Hi', data, offset=1))[0] == [0x1234, -1]
    assert evaluate(cp.unpack('<f', cp.pack('<i', cp.value(1))))[0][0] == struct.unpack('<f', b'\x01\0\0\0')[0]
    assert evaluate(cp.pack('f', cp.value(2)))[0] == list(struct.pack('<f', 2.0))  # int is converted


def test_byte_array() -> None:
    frame = cp.array(list(struct.pack('<hf', -2, 0.5)), 'byte')
    assert evaluate(cp.unpack('<hf', frame))[0] == [-2, 0.5]


def test_errors() -> None:
    with pytest.raises(ValueError):
        cp.pack('q', 1)
    with pytest.raises(ValueError):
        cp.pack('hh', 1)
    with pytest.raises(ValueError):
        cp.unpack('i', [1, 2, 3])
    with pytest.raises(ValueError):
        cp.unpack('h', [1, 2, 3], offset=2)
    with pytest.raises(TypeError):
        cp.pack('h', 1.5)
    with pytest.raises(TypeError):
        cp.pack('h', cp.value(1.5))
