import functools
import pkgutil
import struct
import weakref
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
        return f"{'name:' + names[0] if names else 'h:' + str(hash(self.net.source))[-5:]}"

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
    def __init__(self, value: int | float):
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
    """Anonymous constant. Constants of equal value and type are merged
    to one node when the graph is canonicalized."""


class Input(HeadNode):
    """Named value (copapy value) that can be changed by Target.write_value.
    Each Input is a distinct node, independent of its initial value."""


class Delay(HeadNode):
    """Value of a net at the previous run of the task computing it (unit
    delay). For the computation it is a head node in the memory of the task:
    it is initialized with the initial value and updated by the task after
    each run. The delayed net is no argument, so feedback loops through
    delays do not make the graph cyclic.

    Attributes:
        input: The delayed net, None until it is set by feedback
    """
    def __init__(self, init: int | float, input: 'Net | None' = None):
        super().__init__(init)
        self.name = 'delay_' + self.dtype
        self.input = input


class TimeStep(HeadNode):
    """Time constant factor * period^exponent of the task (cp.dt for the
    period in seconds), written to the memory of each task using it at
    deployment."""
    def __init__(self, factor: float = 1.0, exponent: int = 1) -> None:
        super().__init__(0.0)
        self.name = 'dt'
        self.factor = factor
        self.exponent = exponent


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


class Op(Node):
    def __init__(self, typed_op_name: str, args: Sequence[Net], commutative: bool = False):
        self.name: str = typed_op_name
        self.args: tuple[Net, ...] = tuple(args)
        self.commutative = commutative


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


class ArrayOp(Op):
    """Operation reading and writing heap memory: the arguments are
    accessed by the stencil through the ref_arg<n> symbols, the result
    through ref_out. Array ops do not preserve the register contents.

    Attributes:
        result: Net the operation writes to (array or scalar)
    """
    def __init__(self, typed_op_name: str, args: Sequence[Net], result_dtype: str, length: int | None = None,
                 inplace: bool = False):
        super().__init__(typed_op_name, args)
        self.result: Net = ArrayNet(result_dtype, self, length) if length else Net(result_dtype, self)
        self.inplace = inplace
        """The result is the memory of the first argument, which the operation
        modifies. It must be the only operation using the first argument."""


class ArrayDelay(Node):
    """Array state of a task, the array counterpart of Delay: the value of an
    array net at the previous run of the task. The memory is initialized with
    the initial values and updated by the task after each run. If the delayed
    net is the result of an in-place operation on the state, the update needs
    no copy.

    Attributes:
        values: Initial values
        dtype: Element type
        input: The delayed array net, None until it is set
    """
    def __init__(self, values: Sequence[int | float], dtype: str, input: 'ArrayNet | None' = None):
        self.values = tuple(values)
        self.dtype = dtype
        self.name = 'delay_' + dtype + 'arr'
        self.args = ()
        self.input = input


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
        self.result = ArrayNet(dtype, self, len(args))
        self.elements = [Net(dtype, ArrayElement(self.result, i)) for i in range(len(args))]


class ArrayElement(Node):
    """Scalar element of an array at a constant index. It is no stencil
    but an alias to the memory of the array element.

    Attributes:
        index: Index of the element in the array
        dtype: Type the element is read as, it differs from the type of the
            array if the bit pattern of the element is reinterpreted
    """
    def __init__(self, array_net: ArrayNet, index: int, dtype: str | None = None):
        assert 0 <= index < array_net.length, f"Index {index} out of bounds for length {array_net.length}"
        self.name = 'element_' + array_net.dtype
        self.args = (array_net,)
        self.index = index
        self.dtype = dtype or array_net.dtype


# Nets of the anonymous constants in use, by type and value
_constant_nets: 'weakref.WeakValueDictionary[tuple[type, int | bytes], Net]' = weakref.WeakValueDictionary()


@overload
def value_from_number(val: int) -> value[int]: ...
@overload
def value_from_number(val: float) -> value[float]: ...
def value_from_number(val: int | float) -> value[Any]:
    """Anonymous constant that can be removed during optimization. Constants
    of the same type and value share one node: they can not change, so the
    canonicalization of the graph has nothing to merge and to rebuild for them."""
    # Floats by bit pattern: 0.0 and -0.0 are different constants
    key = (float, struct.pack('d', val)) if isinstance(val, float) else (type(val), val)
    net = _constant_nets.get(key)
    if net is None:
        new_node = Constant(val)
        net = _constant_nets[key] = Net(new_node.dtype, new_node)
    return value(net)


def select_by_mask(mask: value[int], x: Any, y: Any) -> Any:
    """x if all bits of the int mask are set, y if no bit is set. The operand that
    is not selected is replaced by zero by its bit pattern, so the result is exact
    and inf or nan in that operand have no effect. Constant zeros generate no code."""
    kept_x = x if not isinstance(x, value) and x == 0 else add_op('mask', [x, mask])
    kept_y = y if not isinstance(y, value) and y == 0 else add_op('masknot', [y, mask])
    return kept_x + kept_y


def to_float(val: value[Any]) -> value[float]:
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
