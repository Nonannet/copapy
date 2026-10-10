from typing import Any, TypeVar, overload
from ._basic_types import value, uniint, unifloat, to_float, select_by_mask
from ._helper_types import TNum
from ._arrays import array, ArrayType

TContainer = TypeVar("TContainer", bound='array[Any] | ArrayType[Any]')


@overload
def iif(expression: value[Any], true_result: uniint, false_result: uniint) -> value[int]: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def iif(expression: value[Any], true_result: unifloat, false_result: unifloat) -> value[float]: ...
@overload
def iif(expression: float | int, true_result: TNum, false_result: TNum) -> TNum: ...
@overload
def iif(expression: float | int, true_result: TNum | value[TNum], false_result: value[TNum]) -> value[TNum]: ...
@overload
def iif(expression: float | int, true_result: value[TNum], false_result: TNum | value[TNum]) -> value[TNum]: ...
@overload
def iif(expression: float | int | value[Any], true_result: TNum | value[TNum], false_result: TNum | value[TNum]) -> value[TNum] | TNum: ...
@overload
def iif(expression: float | int | value[Any], true_result: TContainer, false_result: Any) -> TContainer: ...
@overload
def iif(expression: float | int | value[Any], true_result: Any, false_result: TContainer) -> TContainer: ...
def iif(expression: Any, true_result: Any, false_result: Any) -> Any:
    """Inline if-else operation. Returns true_result if expression is non-zero,
    else returns false_result.

    Both results are computed if expression is not known at trace time. The selection
    is done without a branching. The result is exact and inf or nan in the
    result that is not selected have no effect.

    Vectors, tensors and arrays as results are selected as a whole by the scalar
    expression. They are blended by multiplications with 1 and 0 (array stencils
    for arrays and packed vectors or tensors): inf or nan in an element of the
    result that is not selected gives nan.

    Arguments:
        expression: The condition to evaluate: a number or copapy value.
        true_result: The result if expression is non-zero.
        false_result: The result if expression is zero.

    Returns:
        The selected result based on the evaluation of expression.
    """
    allowed_type = (value, int, float)
    assert isinstance(expression, allowed_type), "The expression must be a number or a copapy value"
    if not (isinstance(true_result, allowed_type) and isinstance(false_result, allowed_type)):
        # Vectors, tensors and arrays
        if not isinstance(expression, value):
            return true_result if expression != 0 else false_result
        flag: value[int] = value((expression if expression.dtype == 'bool' else expression != 0).net, 'int')
        return true_result * flag + false_result * (1 - flag)

    def is_float(v: Any) -> bool:
        return v.dtype == 'float' if isinstance(v, value) else isinstance(v, float)

    if is_float(true_result) or is_float(false_result):
        true_result = to_float(true_result) if isinstance(true_result, value) else float(true_result)
        false_result = to_float(false_result) if isinstance(false_result, value) else float(false_result)

    if not isinstance(expression, value):
        return true_result if expression != 0 else false_result

    # Results of comparisons are 0 or 1: negated to a mask with no or all bits set
    condition = expression if expression.dtype == 'bool' else expression != 0
    return select_by_mask(-condition, true_result, false_result)
