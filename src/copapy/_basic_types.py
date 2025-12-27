import pkgutil
from typing import Any, Sequence, TypeVar, overload, TypeAlias, Generic, cast
from ._stencils import stencil_database, detect_process_arch
import copapy as cp
from ._helper_types import TNum

NumLike: TypeAlias = 'value[int] | value[float] | int | float'
unifloat: TypeAlias = 'value[float] | float'
uniint: TypeAlias = 'value[int] | int'

TCPNum = TypeVar("TCPNum", bound='value[Any]')
TVarNumb: TypeAlias = 'value[Any] | int | float'

stencil_cache: dict[tuple[str, str], stencil_database] = {}


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
        self.args: tuple[Net, ...] = tuple()
        self.name: str = ''
        self.node_hash = 0

    def __repr__(self) -> str:
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, CPConstant) else '')})"


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
            new_node = CPConstant(int(source), False)
            self.net = Net(new_node.dtype, new_node)
            self.dtype = dtype
        elif dtype == 'float':
            new_node = CPConstant(float(source), False)
            self.net = Net(new_node.dtype, new_node)
            self.dtype = dtype
        elif dtype is None:
            if isinstance(source, bool):
                new_node = CPConstant(source, False)
                self.net = Net(new_node.dtype, new_node)
                self.dtype = 'bool'
            else:
                new_node = CPConstant(source, False)
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
    def __mul__(self, other: TVarNumb) -> Any:
        if self.dtype == 'float' and isinstance(other, int):
            other = float(other)  # Prevent runtime conversion of consts; TODO: add this for other operations
        if not isinstance(other, value):
            if other == 1:
                return self
            elif other == 0:
                return 0
        return add_op('mul', [self, other], True)

    @overload
    def __rmul__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rmul__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rmul__(self, other: float) -> 'value[float]': ...
    def __rmul__(self, other: NumLike) -> Any:
        return self * other

    def __truediv__(self, other: NumLike) -> 'value[float]':
        return add_op('div', [self, other])

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
    def __floordiv__(self, other: TVarNumb) -> Any:
        return add_op('floordiv', [self, other])

    @overload
    def __rfloordiv__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rfloordiv__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rfloordiv__(self, other: float) -> 'value[float]': ...
    def __rfloordiv__(self, other: NumLike) -> Any:
        return add_op('floordiv', [other, self])

    def __abs__(self: TCPNum) -> TCPNum:
        return cp.abs(self)  # type: ignore

    def __neg__(self: TCPNum) -> TCPNum:
        if self.dtype == 'float':
            return cast(TCPNum, add_op('sub', [value(0.0), self]))
        return cast(TCPNum, add_op('sub', [value(0), self]))

    def __gt__(self, other: TVarNumb) -> 'value[int]':
        return add_op('gt', [self, other], dtype='bool')

    def __lt__(self, other: TVarNumb) -> 'value[int]':
        return add_op('gt', [other, self], dtype='bool')

    def __ge__(self, other: TVarNumb) -> 'value[int]':
        return add_op('ge', [self, other], dtype='bool')

    def __le__(self, other: TVarNumb) -> 'value[int]':
        return add_op('ge', [other, self], dtype='bool')

    def __eq__(self, other: TVarNumb) -> 'value[int]':  # type: ignore
        return add_op('eq', [self, other], True, dtype='bool')

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
    def __mod__(self, other: TVarNumb) -> Any:
        return add_op('mod', [self, other])

    @overload
    def __rmod__(self: 'value[TNum]', other: TNum) -> 'value[TNum]': ...
    @overload
    def __rmod__(self: 'value[int]', other: int) -> 'value[int]': ...
    @overload
    def __rmod__(self, other: float) -> 'value[float]': ...
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
    def __lshift__(self, other: uniint) -> 'value[int]':
        return add_op('lshift', [self, other])

    def __rlshift__(self, other: uniint) -> 'value[int]':
        return add_op('lshift', [other, self])

    def __rshift__(self, other: uniint) -> 'value[int]':
        return add_op('rshift', [self, other])

    def __rrshift__(self, other: uniint) -> 'value[int]':
        return add_op('rshift', [other, self])

    def __and__(self, other: uniint) -> 'value[int]':
        return add_op('bwand', [self, other], True)

    def __rand__(self, other: uniint) -> 'value[int]':
        return add_op('bwand', [other, self], True)

    def __or__(self, other: uniint) -> 'value[int]':
        return add_op('bwor', [self, other], True)

    def __ror__(self, other: uniint) -> 'value[int]':
        return add_op('bwor', [other, self], True)

    def __xor__(self, other: uniint) -> 'value[int]':
        return add_op('bwxor', [self, other], True)

    def __rxor__(self, other: uniint) -> 'value[int]':
        return add_op('bwxor', [other, self], True)


class CPConstant(Node):
    def __init__(self, value: Any, anonymous: bool = True):
        if isinstance(value, int):
            self.value: int | float =  value
            self.dtype = 'int'
        elif isinstance(value, float):
            self.value =  value
            self.dtype = 'float'
        else:
            raise ValueError(f'Non supported data type: {type(value).__name__}')

        self.name = 'const_' + self.dtype
        self.args = tuple()
        self.node_hash = hash(value) ^ hash(self.dtype) if anonymous else id(self)
        self.anonymous = anonymous

    def __eq__(self, other: object) -> bool:
        return (self is other) or (self.anonymous and
                                   isinstance(other, CPConstant) and
                                   other.anonymous and
                                   self.value == other.value and
                                   self.dtype == other.dtype)

    def __hash__(self) -> int:
        return self.node_hash


class Write(Node):
    def __init__(self, input: value[Any] | Net | int | float):
        if isinstance(input, value):
            net = input.net
        elif isinstance(input, Net):
            net = input
        else:
            node = CPConstant(input)
            net = Net(node.dtype, node)

        self.name = 'write_' + transl_type(net.dtype)
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


def value_from_number(val: Any) -> value[Any]:
    # Create anonymous constant that can be removed during optimization
    new_node = CPConstant(val)
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
    
    Arguments:
        expression: The condition to evaluate.
        true_result: The result if expression is non-zero.
        false_result: The result if expression is zero.

    Returns:
        The selected result based on the evaluation of expression.
    """
    allowed_type = (value, int, float)
    assert isinstance(true_result, allowed_type) and isinstance(false_result, allowed_type), "Result type not supported"
    return (expression != 0) * true_result + (expression == 0) * false_result


def add_op(op: str, args: list[value[Any] | int | float], commutative: bool = False, dtype: str | None = None) -> value[Any]:
    arg_values = [a if isinstance(a, value) else value_from_number(a) for a in args]

    if commutative:
        arg_values = sorted(arg_values, key=lambda a: a.dtype)  # TODO: update the stencil generator to generate only sorted order

    typed_op = '_'.join([op] + [transl_type(a.dtype) for a in arg_values])

    if typed_op not in generic_sdb.stencil_definitions:
        raise NotImplementedError(f"Operation {op} not implemented for {' and '.join([a.dtype for a in arg_values])}")

    result_type = generic_sdb.stencil_definitions[typed_op].split('_')[0]

    result_net = Net(result_type, Op(typed_op, [av.net for av in arg_values], commutative))

    if dtype:
        result_type = dtype

    return value(result_net, result_type)
