from typing import Any, Sequence, overload
import math
from ._basic_types import value, ArrayNet, to_float, value_from_number
from ._arrays import array, ArrayType, _add_array_op
from ._vectors import vector
from ._tensors import tensor
from ._casts import cast, _cast_array
from ._math import minimum, maximum

_METHODS = ('linear', 'previous', 'nearest')

# Relative deviation of the step sizes up to which a grid is uniform
_UNIFORM_TOLERANCE = 1e-6


def _index_op(op: str, table: array[Any], pos: Any) -> Any:
    """Add a lerp or bsearch stencil for a scalar or an array of positions. The
    result is a value or an array with the shape of the positions."""
    n = value_from_number(table.size).net
    if isinstance(pos, array):
        node = _add_array_op(f"{op}_floatarr_floatarr",
                             [table.net, _cast_array(pos, 'float').net, n, value_from_number(pos.size).net], pos.size)
        assert isinstance(node.result, ArrayNet)
        return array._from_net(node.result, pos.shape)
    scalar = to_float(pos) if isinstance(pos, value) else value_from_number(float(pos))
    return value(_add_array_op(f"{op}_floatarr_float", [table.net, scalar.net, n]).result)


def _lerp(table: array[Any], t: Any) -> Any:
    """Element of a 1D float array at the fractional index t (value, array or
    number) by linear interpolation, t is clamped to the array"""
    return _index_op('lerp', table, t)


def _bsearch(grid: array[Any], x: Any) -> Any:
    """Fractional index of x (value, array or number) in an increasing 1D float
    array, clamped to the array"""
    return _index_op('bsearch', grid, x)


def _floor_index(t: Any, n: int) -> Any:
    """Fractional index rounded down to an element index (as float), clamped
    to a table of n elements"""
    if not isinstance(t, value | array):
        return float(math.floor(min(max(t, 0.0), n - 1.0)))
    return cast(cast(minimum(maximum(t, 0.0), n - 1.0), 'int'), 'float')


def get_table_delta(table: array[Any], t: Any) -> tuple[Any, Any]:
    """Get difference of the two elements of a table around the fractional index
    t and the int flag (1 or 0) if t is inside of the table and not clamped.
    
    Arguments:
        table: 1D float array with at least two elements
        t: Fractional index (value, array or number) in the table
    
    Returns:
        diff: Difference of the two elements around t, with the shape of t
        inside: Boolean flag indicating if t is inside the table and not clamped
    """
    n = table.size
    lower = minimum(_floor_index(t, n), n - 2.0)
    diff = _lerp(table, lower + 1.0) - _lerp(table, lower)
    return diff, (t > 0.0) * (t < n - 1.0)


def _numbers(v: Any, name: str) -> list[float] | None:
    """Elements of a table given as numbers, None if it is an array or contains copapy values"""
    if isinstance(v, array):
        return None
    elements: Any = v.values if isinstance(v, ArrayType) else v
    if isinstance(v, ArrayType) and v.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {v.shape}")
    if not isinstance(elements, Sequence):
        raise TypeError(f"{name} must be a sequence, vector, tensor or array, not {type(v).__name__}")
    if any(isinstance(e, value) for e in elements):
        return None
    if not all(isinstance(e, int | float) for e in elements):
        raise ValueError(f"{name} must be one-dimensional and contain numbers or copapy values")
    return [float(e) for e in elements]


def _table_array(v: Any, numbers: list[float] | None, name: str) -> array[Any]:
    """Table as 1D float array"""
    if numbers is not None:
        return array(numbers, 'float')
    arr = v if isinstance(v, array) else (v._force_array() if isinstance(v, ArrayType) else array(list(v)))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return _cast_array(arr, 'float')


def _interp_number(x: float, xp: list[float], fp: list[float], method: str) -> float:
    """Interpolation of numbers at trace time"""
    if x != x:
        return x
    n = len(xp)
    i = 0
    while i < n - 2 and xp[i + 1] <= x:
        i += 1
    f = min(max((x - xp[i]) / (xp[i + 1] - xp[i]), 0.0), 1.0)
    if method == 'previous':
        return fp[i + 1] if f == 1.0 else fp[i]
    if method == 'nearest':
        return fp[i + 1] if f >= 0.5 else fp[i]
    return fp[i] * (1.0 - f) + fp[i + 1] * f


@overload
def interp(x: float | int, xp: Any, fp: Any, method: str = 'linear') -> 'value[float] | float': ...
@overload
def interp(x: value[Any], xp: Any, fp: Any, method: str = 'linear') -> value[float]: ...
@overload
def interp(x: vector[Any], xp: Any, fp: Any, method: str = 'linear') -> vector[float]: ...
@overload
def interp(x: tensor[Any], xp: Any, fp: Any, method: str = 'linear') -> tensor[float]: ...
@overload
def interp(x: array[Any], xp: Any, fp: Any, method: str = 'linear') -> array[float]: ...
def interp(x: Any, xp: Any, fp: Any, method: str = 'linear') -> Any:
    """One-dimensional interpolation of a table.

    Outside of the range of xp the first or last value of fp is returned. The
    position in the table is computed by two scalar operations if xp is a
    uniform grid of numbers, otherwise by a binary search. For the method
    'linear' x = nan results in nan.

    Arguments:
        x: Position(s) to evaluate
        xp: Increasing points of the table: a sequence of numbers or copapy
            values, a vector, 1D tensor or 1D array with at least two elements
        fp: Values of the table at the points xp, with the same number of
            elements as xp.
        method: 'linear' for the linear interpolation between two points,
            'previous' for the value of the last point not larger than x or
            'nearest' for the value of the point closest to x

    Returns:
        Interpolated value(s) with the shape of x.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}, expected one of {', '.join(_METHODS)}")

    grid_numbers, table_numbers = _numbers(xp, 'xp'), _numbers(fp, 'fp')
    grid = None if grid_numbers is not None else _table_array(xp, None, 'xp')
    table = None if table_numbers is not None else _table_array(fp, None, 'fp')
    n = len(grid_numbers) if grid_numbers is not None else grid.size  # type: ignore[union-attr]
    n_table = len(table_numbers) if table_numbers is not None else table.size  # type: ignore[union-attr]
    if n != n_table:
        raise ValueError(f"xp and fp must have the same number of elements, got {n} and {n_table}")
    if n < 2:
        raise ValueError("The table requires at least two points")

    uniform = False
    if grid_numbers is not None:
        steps = [b - a for a, b in zip(grid_numbers, grid_numbers[1:])]
        if not all(s > 0 for s in steps):
            raise ValueError("xp must be strictly increasing")
        mean_step = (grid_numbers[-1] - grid_numbers[0]) / (n - 1)
        uniform = all(abs(s - mean_step) <= _UNIFORM_TOLERANCE * mean_step for s in steps)

    def evaluate(pos: Any) -> Any:
        """Table at one position: a number, value or array"""
        nonlocal grid, table
        if not isinstance(pos, value | array):
            if grid_numbers is not None and table_numbers is not None:
                return _interp_number(float(pos), grid_numbers, table_numbers, method)
            pos = float(pos)
        if uniform:
            assert grid_numbers is not None
            t = (pos - grid_numbers[0]) * (1.0 / mean_step)
        else:
            if grid is None:
                grid = array(grid_numbers, 'float')  # type: ignore[arg-type]
            t = _bsearch(grid, pos)
        if method == 'previous':
            t = _floor_index(t, n)
        elif method == 'nearest':
            t = _floor_index(t + 0.5, n)
        if table is None:
            table = array(table_numbers, 'float')  # type: ignore[arg-type]
        return _lerp(table, t)

    if isinstance(x, array):
        return evaluate(x)
    if isinstance(x, ArrayType):
        arr = None if x._is_constant() else x._get_array()
        if arr is not None:
            return type(x)._from_array(evaluate(arr))
        results = [evaluate(v) for v in x.values]
        return vector(results) if isinstance(x, vector) else tensor(results, x.shape)
    if not isinstance(x, value | int | float):
        raise TypeError(f"Can not interpolate at {type(x).__name__}")
    return evaluate(x)
