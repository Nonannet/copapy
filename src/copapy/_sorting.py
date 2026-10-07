from ._vectors import vector
from ._arrays import array, ArrayType, TArrayType
from typing import Any, cast


def _check_1d(input_vector: ArrayType[Any]) -> None:
    if input_vector.ndim != 1:
        raise ValueError(f"Expected a 1D vector or tensor, got shape {input_vector.shape}")


def _as_array(input_vector: ArrayType[Any]) -> array[Any]:
    """Elements of the input as array, packed independent of the pack threshold"""
    _check_1d(input_vector)
    if not input_vector.shape[0]:
        raise ValueError("Empty vectors are not supported")
    return input_vector._force_array()


def sort(input_vector: TArrayType) -> TArrayType:
    """
    Sort the elements of a vector in ascending order.

    Arguments:
        input_vector: The input vector (or 1D tensor) containing numerical values.

    Returns:
        Sorted vector.
    """
    _check_1d(input_vector)
    if input_vector._is_constant():
        constants: list[Any] = list(input_vector.values)
        array_type: Any = type(input_vector)
        return cast(TArrayType, array_type(sorted(constants)))
    return type(input_vector)._from_array(_as_array(input_vector).sort())


def argsort(input_vector: ArrayType[Any]) -> vector[int]:
    """
    Perform an indirect sort. It returns a vector of indices that index data
    in sorted order. Equal elements are in order of their index (like a stable sort).

    Arguments:
        input_vector: The input vector (or 1D tensor) containing numerical values.

    Returns:
        Index vector.
    """
    _check_1d(input_vector)
    if input_vector._is_constant():
        data: list[Any] = list(input_vector.values)
        indices: list[int] = sorted(range(len(data)), key=lambda i: data[i])
        return vector(indices)
    return vector._from_array(_as_array(input_vector).argsort())
