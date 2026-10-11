from typing import Any, Sequence, overload
import re
import struct
from ._basic_types import NumLike, uniint, value, Net, Constant, ArrayNet, ArrayPack, ArrayElement, transl_type, value_from_number, add_op
from ._basic_types import to_float as _value_to_float
from ._arrays import array, ArrayType, _add_array_op
from ._vectors import vector
from ._tensors import tensor

_dtype_names: dict[Any, str] = {float: 'float', int: 'int', bool: 'bool'}
_number_types: dict[str, Any] = {'float': float, 'int': int, 'bool': bool}

# Size in bytes of the fields supported by pack and unpack (struct format characters)
_field_sizes: dict[str, int] = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4, 'f': 4, 'x': 1}


def _cast_value(x: value[Any], dtype: str) -> value[Any]:
    """Convert a copapy value: constants at trace time, variables by a stencil.
    Bool values are stored as int, so bool to int generates no code."""
    if x.dtype == dtype:
        return x
    if dtype == 'float':
        return _value_to_float(x)
    source = x.net.source
    if dtype == 'int':
        if x.dtype == 'bool':
            return value(x.net, 'int')
        if isinstance(source, Constant):
            return value_from_number(int(source.value))
        return add_op('int', [x])
    if isinstance(source, Constant):
        return value(value_from_number(int(bool(source.value))).net, 'bool')
    return x != 0


def _cast_scalar(x: Any, dtype: str) -> Any:
    if isinstance(x, value):
        return _cast_value(x, dtype)
    return _number_types[dtype](x)


def _cast_array(x: array[Any], dtype: str) -> array[Any]:
    """Convert an array by a single array stencil"""
    if x.dtype == dtype:
        return x
    if dtype == 'int' and x.dtype == 'bool':
        return x._computed()
    n = value_from_number(x.size).net
    node = _add_array_op(f"{dtype}_{x.net.dtype}arr", [x.net, n], x.size)
    assert isinstance(node.result, ArrayNet)
    return array._from_net(node.result, x.shape, dtype)


def _cast_array_type(x: ArrayType[Any], dtype: str) -> ArrayType[Any]:
    """Convert the elements, by an array stencil if x is packed. The result
    has the class of x."""
    arr = None if x._is_constant() else x._get_array()
    if arr is not None:
        converted = _cast_array(arr, dtype)
        return x if converted is arr else type(x)._from_array(converted)
    values = tuple(_cast_scalar(v, dtype) for v in x.values)
    if all(new is old for new, old in zip(values, x.values)):
        return x
    ret = type(x).__new__(type(x))
    ret._set_elements(values, x.shape, x._packed)
    return ret


def cast(x: Any, dtype: 'str | type') -> Any:
    """Convert a number, a copapy value or the elements of a vector, tensor or
    array to another type.

    Numbers and constants are converted at trace time, copapy values by a
    stencil. Float to int truncates towards zero (like Python int()), a bool
    is True for all values other than zero. Bool values are stored as int,
    a conversion from bool to int generates no code.

    Arguments:
        x: Number, copapy value, vector, tensor or array
        dtype: Target type: 'float', 'int' or 'bool' or the Python
            types float, int or bool

    Returns:
        x converted to dtype, x itself if it already has the type
    """
    name: str = _dtype_names.get(dtype, '') if isinstance(dtype, type) else dtype
    if name == 'byte' and isinstance(x, array):
        return _cast_array(x, name)
    if name not in _number_types:
        raise ValueError(f"Unsupported type {dtype}, expected 'float', 'int' or 'bool' ('byte' for arrays)")
    if isinstance(x, array):
        return _cast_array(x, name)
    if isinstance(x, ArrayType):
        return _cast_array_type(x, name)
    if isinstance(x, (value, int, float)):
        return _cast_scalar(x, name)
    raise TypeError(f"Can not cast {type(x).__name__} to {name}")


@overload
def to_float(x: float | int) -> float: ...
@overload
def to_float(x: value[Any]) -> value[float]: ...
@overload
def to_float(x: vector[Any]) -> vector[float]: ...
@overload
def to_float(x: tensor[Any]) -> tensor[float]: ...
@overload
def to_float(x: array[Any]) -> array[float]: ...
@overload
def to_float(x: ArrayType[Any]) -> ArrayType[float]: ...
def to_float(x: Any) -> Any:
    """Convert a number, a copapy value or the elements of a vector, tensor
    or array to float.

    Arguments:
        x: Number, copapy value, vector, tensor or array

    Returns:
        x as float
    """
    return cast(x, 'float')


@overload
def to_int(x: float | int) -> int: ...
@overload
def to_int(x: value[Any]) -> value[int]: ...
@overload
def to_int(x: vector[Any]) -> vector[int]: ...
@overload
def to_int(x: tensor[Any]) -> tensor[int]: ...
@overload
def to_int(x: array[Any]) -> array[int]: ...
@overload
def to_int(x: ArrayType[Any]) -> ArrayType[int]: ...
def to_int(x: Any) -> Any:
    """Convert a number, a copapy value or the elements of a vector, tensor
    or array to int. Floats are truncated towards zero (like Python int()).

    Arguments:
        x: Number, copapy value, vector, tensor or array

    Returns:
        x as int
    """
    return cast(x, 'int')


@overload
def to_bool(x: float | int) -> bool: ...
@overload
def to_bool(x: value[Any]) -> value[int]: ...
@overload
def to_bool(x: vector[Any]) -> vector[int]: ...
@overload
def to_bool(x: tensor[Any]) -> tensor[int]: ...
@overload
def to_bool(x: array[Any]) -> array[int]: ...
@overload
def to_bool(x: ArrayType[Any]) -> ArrayType[int]: ...
def to_bool(x: Any) -> Any:
    """Convert a number, a copapy value or the elements of a vector, tensor
    or array to bool: True for all values other than zero.

    Arguments:
        x: Number, copapy value, vector, tensor or array

    Returns:
        x as bool
    """
    return cast(x, 'bool')


def to_byte(x: array[Any]) -> array[int]:
    """Convert the elements of an array to bytes, limited to the range 0 to
    255. Floats are truncated towards zero, nan is converted to 0.

    Arguments:
        x: Array

    Returns:
        x as byte array
    """
    if not isinstance(x, array):
        raise TypeError(f"Can not convert {type(x).__name__} to byte, only arrays are supported")
    return _cast_array(x, 'byte')


def _parse_format(fmt: str) -> tuple[bool, list[str]]:
    """Byte order (True for big-endian) and the field characters of a struct
    format string, repeat counts are expanded."""
    fmt = ''.join(fmt.split())
    big = fmt.startswith('>')
    fields: list[str] = []
    for count, char in re.findall(r'(\d*)(.)', fmt[1:] if fmt[:1] in ('<', '>') else fmt):
        if char not in _field_sizes:
            raise ValueError(f"Unsupported format character '{char}', expected one of {' '.join(_field_sizes)}")
        fields += [char] * (int(count) if count else 1)
    return big, fields


def _reinterpret(x: value[Any], dtype: str) -> value[Any]:
    """Bit pattern of a float value as int or of an int value as float. It
    requires no stencil: the value is stored to memory and read with the other type."""
    stored = ArrayPack([x.net], transl_type(x.dtype))
    return value(Net(dtype, ArrayElement(stored.result, 0)))


def _pack_field(x: NumLike, char: str, big: bool) -> list[uniint]:
    size = _field_sizes[char]
    if char == 'f':
        if not isinstance(x, value):
            return list(struct.pack('>f' if big else '<f', x))
        bits = _reinterpret(_value_to_float(x), 'int')
    elif isinstance(x, value):
        if x.dtype == 'float':
            raise TypeError(f"Format '{char}' requires an int value, not float")
        bits = x
    elif isinstance(x, int):
        return list((x & (256 ** size - 1)).to_bytes(size, 'big' if big else 'little'))
    else:
        raise TypeError(f"Format '{char}' requires an int, not {type(x).__name__}")
    ret: list[uniint] = [(bits >> 8 * i if i else bits) & 0xFF for i in range(size)]
    return ret[::-1] if big else ret


def _unpack_field(data: Sequence[uniint], char: str, big: bool) -> NumLike:
    data = data[::-1] if big else data  # Least significant byte first
    terms = [b << 8 * i if i else b for i, b in enumerate(data) if isinstance(b, value)]
    if not terms:
        number: int | float = struct.unpack('<' + char, bytes(b for b in data if not isinstance(b, value)))[0]
        return number

    # Constant bytes are combined at trace time (as signed 32 bit int)
    const = sum(b << 8 * i for i, b in enumerate(data) if not isinstance(b, value))
    const = (const ^ 0x80000000) - 0x80000000
    ret = terms[0]
    for term in terms[1:]:
        ret = ret | term
    if const:
        ret = ret | const
    if char in ('b', 'h'):
        sign = 1 << (8 * len(data) - 1)
        ret = (ret ^ sign) - sign
    return _reinterpret(ret, 'float') if char == 'f' else ret


def pack(fmt: str, *values: NumLike) -> list[uniint]:
    """Convert numbers or copapy values to their bytes, like struct.pack.

    The format is a struct format string limited to 32 bit types: 'b' and 'B'
    (8 bit int), 'h' and 'H' (16 bit int), 'i' and 'I' (32 bit int), 'f'
    (32 bit float) and 'x' (pad byte), with optional repeat counts. It starts
    with '<' for little-endian (default) or '>' for big-endian byte order.

    Int values that do not fit into the field are truncated to its lowest bits.
    Numbers and constants are converted at trace time.

    Arguments:
        fmt: Format string, e.g. '<Hhf'
        values: One int or float number or copapy value per field

    Returns:
        List of bytes: copapy int values or ints from 0 to 255
    """
    big, fields = _parse_format(fmt)
    if len(values) != sum(f != 'x' for f in fields):
        raise ValueError(f"Format '{fmt}' requires {sum(f != 'x' for f in fields)} values, got {len(values)}")
    remaining = iter(values)
    ret: list[uniint] = []
    for char in fields:
        ret += [0] if char == 'x' else _pack_field(next(remaining), char, big)
    return ret


def unpack(fmt: str, data: 'Sequence[uniint] | array[int]', offset: int = 0) -> tuple[Any, ...]:
    """Convert bytes to int and float values, like struct.unpack_from.

    The format is described at pack. Unsigned 32 bit fields ('I') are
    returned as int, values above 2**31 - 1 are therefore negative.

    Arguments:
        fmt: Format string, e.g. '<Hhf'
        data: Bytes as ints or copapy int values from 0 to 255, e.g. the
            result of pack or a byte array
        offset: Index of the first byte in data

    Returns:
        Tuple with one copapy value per field, numbers if all its bytes are numbers
    """
    big, fields = _parse_format(fmt)
    size = sum(_field_sizes[f] for f in fields)
    if offset < 0 or offset + size > len(data):
        raise ValueError(f"Format '{fmt}' requires {size} bytes at offset {offset}, data has {len(data)}")
    ret: list[NumLike] = []
    for char in fields:
        if char != 'x':
            ret.append(_unpack_field([data[offset + i] for i in range(_field_sizes[char])], char, big))
        offset += _field_sizes[char]
    return tuple(ret)
