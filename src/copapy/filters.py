from . import value, vector
from ._basic_types import iif, unifloat
from._helper_types import TNum
from typing import Any, Iterable


def homogenize_vector(input_values: Iterable[TNum | value[TNum]]) -> Iterable[TNum] | Iterable[value[TNum]]:
    input_list = list(input_values)
    if any(isinstance(val, value) for val in input_list):
        return (v if isinstance(v, value) else value(v) for v in input_list)
    else:
        return (v for v in input_list if not isinstance(v, value))


def _inv_argsort(input_vector: vector[TNum]) -> vector[int]:
    positions = (sum((v1 > v2) for v2 in input_vector) for v1 in input_vector)
    return vector(positions)


def argsort(input_vector: vector[TNum]) -> vector[int]:
    """
    Perform an indirect sort. It returns an array of indices that index data
    in sorted order.

    Args:
        input_vector: The input vector containing numerical values.

    Returns:
        Index array.
    """
    return _inv_argsort(_inv_argsort(input_vector))


def median(input_vector: vector[TNum]) -> TNum | value[TNum]:
    """
    Applies a median filter to the input vector and returns the median as a unifloat.

    Args:
        input_vector: The input vector containing numerical values.

    Returns:
        The median value of the input vector.
    """
    vec = input_vector
    ret = vec[0]
    for v1 in vec:
        n2 = len(vec) // 2 + 1
        lt = sum(v1 < v2 for v2 in vec)
        gt = sum(v1 > v2 for v2 in vec)
        ret = iif((lt < n2) & (gt < n2), v1, ret)

    return ret


def mean(input_vector: vector[Any]) -> unifloat:
    """
    Applies a mean filter to the input vector and returns the mean as a unifloat.

    Args:
        input_vector (vector): The input vector containing numerical values.

    Returns:
        unifloat: The mean value of the input vector.
    """
    return input_vector.sum() / len(input_vector)
