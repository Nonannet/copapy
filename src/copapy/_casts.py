from typing import Any, overload
from ._basic_types import value, Constant, ArrayNet, transl_type, value_from_number, add_op
from ._basic_types import to_float as _value_to_float
from ._arrays import array, ArrayType, _add_array_op
from ._vectors import vector
from ._tensors import tensor

_dtype_names: dict[Any, str] = {float: 'float', int: 'int', bool: 'bool'}
_number_types: dict[str, Any] = {'float': float, 'int': int, 'bool': bool}


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
    if name not in _number_types:
        raise ValueError(f"Unsupported type {dtype}, expected 'float', 'int' or 'bool'")
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
