import functools
import pkgutil
from typing import Any, Sequence, TypeVar, overload, TypeAlias, Generic, Callable
from ._stencils import stencil_database, detect_process_arch
import copapy as cp
from ._helper_types import TNum

NumLike: TypeAlias = 'value[int] | value[float] | int | float'
unifloat: TypeAlias = 'value[float] | float'
uniint: TypeAlias = 'value[int] | int'

TCPNum = TypeVar("TCPNum", bound='value[Any]')
TVarNumb: TypeAlias = 'value[Any] | int | float'

stencil_cache: dict[tuple[str, str], stencil_database] = {}


TFunc = TypeVar("TFunc", bound=Callable[..., Any])


def scalar_op(func: TFunc) -> TFunc:
    """Decorator for binary operators of value: returns NotImplemented for
    non-scalar operands, so Python falls back to the reflected operator
    of the other operand (e.g. vector.__rsub__ for value - vector)."""
    @functools.wraps(func)
    def wrapper(self: Any, other: Any) -> Any:
        if not isinstance(other, (value, int, float)):
            return NotImplemented
        return func(self, other)
    return wrapper  # type: ignore[return-value]


def get_var_name(var: Any, scope: dict[str, Any] = globals()) -> list[str]:
    return [name for name, value in scope.items() if value is var]


def stencil_db_from_package(arch: str = 'native', optimization: str = 'O3') -> stencil_database:
    global stencil_cache
    ci = (arch, optimization)
    if ci in stencil_cache:
        return stencil_cache[ci]  # return cached stencil db
    if arch == 'native':
        arch = detect_process_arch()
    stencil_data = pkgutil.get_data(__name__, f"obj/stencils_{arch}_{optimization}.o")
    assert stencil_data, f"stencils_{arch}_{optimization} not found"
    sdb = stencil_database(stencil_data)
    stencil_cache[ci] = sdb
    return sdb


generic_sdb = stencil_db_from_package()


def transl_type(t: str) -> str:
    return {'bool': 'int'}.get(t, t)


class Node:
    """A Node represents an computational operation like ADD or other operations
    like read and write from or to the memory or IOs. In the computation graph
    Nodes are connected via Nets.

    Attributes:
        args (list[Net]): The input Nets to this Node.
        name (str): The name of the operation this Node represents.
    """
    def __init__(self) -> None:
        self.args: tuple[Net, ...] = ()
        self.name: str = ''
        self.node_hash = 0

    def __repr__(self) -> str:
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, HeadNode) else '')})"


class Net:
    """A Net represents a scalar type in the computation graph - or more generally it
    connects Nodes together.

    Attributes:
        dtype (str): The data type of this Net.
        source (Node): The Node that produces the value for this Net.
    """
    def __init__(self, dtype: str, source: Node):
        self.dtype = dtype
        self.source = source

    def __repr__(self) -> str:
        names = get_var_name(self)
        return f"{'name:' + names[0] if names else 'h:' + str(hash(self))[-5:]}"

    def __hash__(self) -> int:
        return self.source.node_hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Net) and self.source == other.source


class value(Generic[TNum]):
    """A "value" represents a typed scalar variable. It supports arithmetic and
    comparison operations.

    Attributes:
        dtype (str): Data type of this value.
    """
    def __init__(self, source: TNum | Net, dtype: str | None = None):
        """Instance a value.

        Arguments:
            dtype: Data type of this value.
            net: Reference to the underlying Net in the graph
        """
        if isinstance(source, Net):
            self.net: Net = source
            if dtype:
                assert transl_type(dtype) == source.dtype, f"Type of Net ({source.dtype}) does not match {dtype}"
                self.dtype: str = dtype
            else:
                self.dtype = source.dtype
        elif dtype == 'int' or dtype == 'bool':
            new_node = Input(int(source))
            self.net = Net(new_node.dtype, new_node)
            self.dtype = dtype
        elif dtype == 'float':
            new_node = Input(float(source))
            self.net = Net(new_node.dtype, new_node)
            self.dtype = dtype
        elif dtype is None:
            if isinstance(source, bool):
                new_node = Input(source)
                self.net = Net(new_node.dtype, new_node)
                self.dtype = 'bool'
            else:
                new_node = Input(source)
                self.net = Net(new_node.dtype, new_node)
                self.dtype = new_node.dtype
        else:
            raise ValueError('Unknown type: {dtype}')

    def __repr__(self) -> str:
        names = get_var_name(self)
        return f"{'name:' + names[0] if names else 'h:' + str(self.net.source.node_hash)[-5:]}"

    @overload
    def __add__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __add__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __add__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __add__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __add__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    @scalar_op
    def __add__(self, other: TVarNumb) -> Any:
        if not isinstance(other, value) and other == 0:
            return self
        return add_op('add', [self, other], True)

    @overload
    def __radd__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __radd__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __radd__(self, other: float) -> 'value[float]': ...
    @scalar_op
    def __radd__(self, other: NumLike) -> Any:
        return self + other

    @overload
    def __sub__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __sub__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __sub__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __sub__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __sub__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    @scalar_op
    def __sub__(self, other: TVarNumb) -> Any:
        if isinstance(other, int | float) and other == 0:
            return self
        return add_op('sub', [self, other])

    @overload
    def __rsub__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rsub__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rsub__(self, other: float) -> 'value[float]': ...
    @scalar_op
    def __rsub__(self, other: NumLike) -> Any:
        return add_op('sub', [other, self])

    @overload
    def __mul__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __mul__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __mul__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __mul__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __mul__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    @scalar_op
    def __mul__(self, other: TVarNumb) -> Any:
        if self.dtype == 'float' and isinstance(other, int):
            other = float(other)  # Prevent runtime conversion of consts; TODO: add this for other operations
        if not isinstance(other, value):
            if other == 1:
                return self
            elif other == 0:
                return 0
        if self is other:
            return add_op('square', [self])
        return add_op('mul', [self, other], True)

    @overload
    def __rmul__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rmul__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rmul__(self, other: float) -> 'value[float]': ...
    @scalar_op
    def __rmul__(self, other: NumLike) -> Any:
        return self * other

    @scalar_op
    def __truediv__(self, other: NumLike) -> 'value[float]':
        return add_op('div', [self, other])

    @scalar_op
    def __rtruediv__(self, other: NumLike) -> 'value[float]':
        return add_op('div', [other, self])

    @overload
    def __floordiv__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __floordiv__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __floordiv__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __floordiv__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __floordiv__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    @scalar_op
    def __floordiv__(self, other: TVarNumb) -> Any:
        return add_op('floordiv', [self, other])

    @overload
    def __rfloordiv__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rfloordiv__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rfloordiv__(self, other: float) -> 'value[float]': ...
    @scalar_op
    def __rfloordiv__(self, other: NumLike) -> Any:
        return add_op('floordiv', [other, self])

    def __abs__(self: 'value[TNum]') -> 'value[TNum]':
        return cp.abs(self)

    def __neg__(self: 'value[TNum]') -> 'value[TNum]':
        return add_op('neg', [self])

    @scalar_op
    def __gt__(self, other: TVarNumb) -> 'value[int]':
        return add_op('gt', [self, other], dtype='bool')

    @scalar_op
    def __lt__(self, other: TVarNumb) -> 'value[int]':
        return add_op('gt', [other, self], dtype='bool')

    @scalar_op
    def __ge__(self, other: TVarNumb) -> 'value[int]':
        return add_op('ge', [self, other], dtype='bool')

    @scalar_op
    def __le__(self, other: TVarNumb) -> 'value[int]':
        return add_op('ge', [other, self], dtype='bool')

    @scalar_op
    def __eq__(self, other: TVarNumb) -> 'value[int]':  # type: ignore
        return add_op('eq', [self, other], True, dtype='bool')

    @scalar_op
    def __ne__(self, other: TVarNumb) -> 'value[int]':  # type: ignore
        return add_op('ne', [self, other], True, dtype='bool')

    @overload
    def __mod__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __mod__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __mod__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __mod__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __mod__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    @scalar_op
    def __mod__(self, other: TVarNumb) -> Any:
        return add_op('mod', [self, other])

    @overload
    def __rmod__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rmod__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rmod__(self, other: float) -> 'value[float]': ...
    @scalar_op
    def __rmod__(self, other: NumLike) -> Any:
        return add_op('mod', [other, self])

    @overload
    def __pow__(self: 'value[TNum]', other: 'value[TNum] | TNum') -> 'value[TNum]': ...
    @overload
    def __pow__(self: 'value[int]', other: uniint) -> 'value[int]': ...
    @overload
    def __pow__(self, other: unifloat) -> 'value[float]': ...
    @overload
    def __pow__(self: 'value[float]', other: NumLike) -> 'value[float]': ...
    @overload
    def __pow__(self, other: TVarNumb) -> 'value[float] | value[int]': ...
    def __pow__(self, other: TVarNumb) -> Any:
        return cp.pow(self, other)

    @overload
    def __rpow__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rpow__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rpow__(self, other: float) -> 'value[float]': ...
    def __rpow__(self, other: NumLike) -> Any:
        return cp.pow(other, self)

    def __hash__(self) -> int:
        return id(self)

    # Bitwise and shift operations for cp[int]
    @scalar_op
    def __lshift__(self, other: uniint) -> 'value[int]':
        return add_op('lshift', [self, other])

    @scalar_op
    def __rlshift__(self, other: uniint) -> 'value[int]':
        return add_op('lshift', [other, self])

    @scalar_op
    def __rshift__(self, other: uniint) -> 'value[int]':
        return add_op('rshift', [self, other])

    @scalar_op
    def __rrshift__(self, other: uniint) -> 'value[int]':
        return add_op('rshift', [other, self])

    @scalar_op
    def __and__(self, other: uniint) -> 'value[int]':
        return add_op('bwand', [self, other], True)

    @scalar_op
    def __rand__(self, other: uniint) -> 'value[int]':
        return add_op('bwand', [other, self], True)

    @scalar_op
    def __or__(self, other: uniint) -> 'value[int]':
        return add_op('bwor', [self, other], True)

    @scalar_op
    def __ror__(self, other: uniint) -> 'value[int]':
        return add_op('bwor', [other, self], True)

    @scalar_op
    def __xor__(self, other: uniint) -> 'value[int]':
        return add_op('bwxor', [self, other], True)

    @scalar_op
    def __rxor__(self, other: uniint) -> 'value[int]':
        return add_op('bwxor', [other, self], True)


class HeadNode(Node):
    """Base class for scalars stored in the data memory.

    Attributes:
        value: Numeric value written to the memory when loading the program.
        dtype: Data type of the value.
    """
    def __init__(self, value: Any):
        if isinstance(value, int):
            self.value: int | float =  value
            self.dtype = 'int'
        elif isinstance(value, float):
            self.value =  value
            self.dtype = 'float'
        else:
            raise ValueError(f'Non supported data type: {type(value).__name__}')

        self.name = 'const_' + self.dtype
        self.args = ()


class Constant(HeadNode):
    """Anonymous constant. Constants of equal value and type are the same
    node and can be removed during optimization."""
    def __init__(self, value: Any):
        super().__init__(value)
        self.node_hash = hash(value) ^ hash(self.dtype)

    def __eq__(self, other: object) -> bool:
        return (self is other) or (isinstance(other, Constant) and
                                   self.value == other.value and
                                   self.dtype == other.dtype)

    def __hash__(self) -> int:
        return self.node_hash


class Input(HeadNode):
    """Named value (copapy value) that can be changed by Target.write_value.
    Each Input is a distinct node, independent of its initial value."""
    def __init__(self, value: Any):
        super().__init__(value)
        self.node_hash = id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return self.node_hash


class Store(Node):
    def __init__(self, input: value[Any] | Net | int | float):
        if isinstance(input, value):
            net = input.net
        elif isinstance(input, Net):
            net = input
        else:
            node = Constant(input)
            net = Net(node.dtype, node)

        self.name = 'store_' + transl_type(net.dtype)
        self.args = (net,)
        self.node_hash = hash(self.name) ^ hash(net.source.node_hash)


class Op(Node):
    def __init__(self, typed_op_name: str, args: Sequence[Net], commutative: bool = False):
        self.name: str = typed_op_name
        self.args: tuple[Net, ...] = tuple(args)
        self.node_hash = self.get_node_hash(commutative)
        self.commutative = commutative

    def get_node_hash(self, commutative: bool = False) -> int:
        if commutative:
            h = hash(self.name) ^ hash(frozenset(a.source.node_hash for a in self.args))
        else:
            h = hash(self.name) ^ hash(tuple(a.source.node_hash for a in self.args))
        return h if h != -1 else -2

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Op):
            return NotImplemented

        # Traverse graph for both notes. Return false on first difference.
        # A false inequality result in seldom cases is ok, whereas a false
        # equality result leads to wrong computation results.
        nodes: list[tuple[Node, Node]] = [(self, other)]
        seen: set[tuple[int, int]] = set()
        while(nodes):
            s_node, o_node = nodes.pop()

            if s_node.node_hash != o_node.node_hash:
                return False
            key = (id(s_node), id(o_node))
            if key in seen:
                continue
            if isinstance(s_node, Op):
                if (s_node.name.split('_')[0] != o_node.name.split('_')[0] or
                    len(o_node.args) != len(s_node.args)):
                    return False
                if s_node.commutative:
                    for s_net, o_net in zip(sorted(s_node.args, key=hash),
                                            sorted(o_node.args, key=hash)):
                        if s_net is not o_net:
                            nodes.append((s_net.source, o_net.source))
                else:
                    for s_net, o_net in zip(s_node.args, o_node.args):
                        if s_net is not o_net:
                            nodes.append((s_net.source, o_net.source))
            elif s_node != o_node:
                return False
            seen.add(key)
        return True

    def __hash__(self) -> int:
        return self.node_hash

class ArrayNet(Net):
    """A Net representing a contiguous array of scalars in the heap memory.

    Attributes:
        dtype (str): The element data type.
        length (int): Number of elements.
    """
    def __init__(self, dtype: str, source: Node, length: int):
        super().__init__(dtype, source)
        self.length = length


class ArrayConst(Node):
    """Array with values set at compile time or written by the host."""
    def __init__(self, values: Sequence[int | float], dtype: str):
        self.values = tuple(values)
        self.dtype = dtype
        self.name = 'const_' + dtype + 'arr'
        self.args = ()
        self.node_hash = id(self)


class ArrayOp(Op):
    """Operation reading and writing heap memory: the arguments are
    accessed by the stencil through the ref_arg<n> symbols, the result
    through ref_out. Array ops do not preserve the register contents.

    Attributes:
        result: Net the operation writes to (array or scalar)
    """
    def __init__(self, typed_op_name: str, args: Sequence[Net], result_dtype: str, length: int | None = None):
        super().__init__(typed_op_name, args)
        self.result: Net = ArrayNet(result_dtype, self, length) if length else Net(result_dtype, self)


class ArrayPack(Node):
    """Array built from scalar values. It is no stencil, the compiler
    stores each scalar into the memory of the corresponding array element.

    Attributes:
        result: The array net
        elements: Nets of the array elements, aliases into the array memory
    """
    def __init__(self, args: Sequence[Net], dtype: str):
        assert all(transl_type(a.dtype) == dtype for a in args), "All values must have the type of the array"
        self.name = 'pack_' + dtype + 'arr'
        self.args = tuple(args)
        self.node_hash = id(self)
        self.result = ArrayNet(dtype, self, len(args))
        self.elements = [Net(dtype, ArrayElement(self.result, i)) for i in range(len(args))]


class ArrayElement(Node):
    """Scalar element of an array at a constant index. It is no stencil
    but an alias to the memory of the array element."""
    def __init__(self, array_net: ArrayNet, index: int):
        assert 0 <= index < array_net.length, f"Index {index} out of bounds for length {array_net.length}"
        self.name = 'element_' + array_net.dtype
        self.args = (array_net,)
        self.index = index
        self.node_hash = hash((array_net.source.node_hash, index, 'element'))

    def __eq__(self, other: object) -> bool:
        return self is other or (isinstance(other, ArrayElement) and
                                 self.index == other.index and
                                 self.args[0] == other.args[0])

    def __hash__(self) -> int:
        return self.node_hash


def value_from_number(val: Any) -> value[Any]:
    # Create anonymous constant that can be removed during optimization
    new_node = Constant(val)
    new_net = Net(new_node.dtype, new_node)
    return value(new_net)


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
def iif(expression: Any, true_result: Any, false_result: Any) -> Any:
    """Inline if-else operation. Returns true_result if expression is non-zero,
    else returns false_result.

    Both results are computed if expression is not known at trace time. The selection
    is done without a branching. The result is exact and inf or nan in the
    result that is not selected have no effect.

    Arguments:
        expression: The condition to evaluate.
        true_result: The result if expression is non-zero.
        false_result: The result if expression is zero.

    Returns:
        The selected result based on the evaluation of expression.
    """
    allowed_type = (value, int, float)
    assert isinstance(true_result, allowed_type) and isinstance(false_result, allowed_type), "Result type not supported"

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


def select_by_mask(mask: value[int], x: Any, y: Any) -> Any:
    """x if all bits of the int mask are set, y if no bit is set. The operand that
    is not selected is replaced by zero by its bit pattern, so the result is exact
    and inf or nan in that operand have no effect. Constant zeros generate no code."""
    kept_x = x if not isinstance(x, value) and x == 0 else add_op('mask', [x, mask])
    kept_y = y if not isinstance(y, value) and y == 0 else add_op('masknot', [y, mask])
    return kept_x + kept_y


def to_float(val: value[Any]) -> value[Any]:
    """Convert an int value to float: constants at trace time, variables by
    the float_int stencil. Float values are returned unchanged."""
    if transl_type(val.dtype) != 'int':
        return val
    source = val.net.source
    if isinstance(source, Constant):
        return value_from_number(float(source.value))
    return add_op('float', [val])


def add_op(op: str, args: list[value[Any] | int | float], commutative: bool = False, dtype: str | None = None) -> value[Any]:
    arg_values = [a if isinstance(a, value) else value_from_number(a) for a in args]
    arg_dtypes = [a.dtype for a in arg_values]

    typed_op = '_'.join([op] + [transl_type(a.dtype) for a in arg_values])

    if typed_op not in generic_sdb.stencil_definitions and any(transl_type(a.dtype) == 'int' for a in arg_values):
        # Float-only operations and operations with mixed argument types
        # have only float stencils: convert the int arguments
        converted = [to_float(a) for a in arg_values]
        if commutative:
            # A converted variable is computed last and is still in register 0
            def computed(new: value[Any], old: value[Any]) -> bool:
                return new is not old and not isinstance(new.net.source, HeadNode)
            pairs = list(zip(converted, arg_values))
            converted = [n for n, o in pairs if computed(n, o)] + [n for n, o in pairs if not computed(n, o)]
        arg_values = converted
        typed_op = '_'.join([op] + [transl_type(a.dtype) for a in arg_values])

    if typed_op not in generic_sdb.stencil_definitions:
        raise NotImplementedError(f"Operation {op} not implemented for {' and '.join(arg_dtypes)}")

    result_type = generic_sdb.stencil_definitions[typed_op].split('_')[0]

    result_net = Net(result_type, Op(typed_op, [av.net for av in arg_values], commutative))

    if dtype:
        result_type = dtype

    return value(result_net, result_type)
