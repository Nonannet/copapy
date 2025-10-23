import pkgutil
from typing import Any, TypeVar, overload, TypeAlias, Generic, cast
from ._stencils import stencil_database
import platform

NumLike: TypeAlias = 'variable[int] | variable[float] | variable[bool] | int | float | bool'
NumLikeAndNet: TypeAlias = 'variable[int] | variable[float] | variable[bool] | int | float | bool | Net'
NetAndNum: TypeAlias = 'Net | int | float'

unifloat: TypeAlias = 'variable[float] | float'
uniint: TypeAlias = 'variable[int] | int'
unibool: TypeAlias = 'variable[bool] | bool'

TNumber = TypeVar("TNumber", bound='CPNumber')
T = TypeVar("T")


def get_var_name(var: Any, scope: dict[str, Any] = globals()) -> list[str]:
    return [name for name, value in scope.items() if value is var]


def stencil_db_from_package(arch: str = 'native', optimization: str = 'O3') -> stencil_database:
    if arch == 'native':
        arch = platform.machine()
    stencil_data = pkgutil.get_data(__name__, f"obj/stencils_{arch}_{optimization}.o")
    assert stencil_data, f"stencils_{arch}_{optimization} not found"
    return stencil_database(stencil_data)


generic_sdb = stencil_db_from_package()


def transl_type(t: str) -> str:
    return {'bool': 'int'}.get(t, t)


class Node:
    def __init__(self) -> None:
        self.args: list[Net] = []
        self.name: str = ''

    def __repr__(self) -> str:
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, CPConstant) else '')})"


class Device():
    pass


class Net:
    def __init__(self, dtype: str, source: Node):
        self.dtype = dtype
        self.source = source

    def __repr__(self) -> str:
        names = get_var_name(self)
        return f"{'name:' + names[0] if names else 'id:' + str(id(self))[-5:]}"

    def __hash__(self) -> int:
        return id(self)


class CPNumber(Net):
    def __init__(self, dtype: str, source: Node):
        self.dtype = dtype
        self.source = source

    @overload
    def __mul__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __mul__(self, other: unifloat) -> 'variable[float]':
        ...

    def __mul__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mul', [self, other], True)

    @overload
    def __rmul__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rmul__(self, other: unifloat) -> 'variable[float]':
        ...

    def __rmul__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mul', [self, other], True)

    @overload
    def __add__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __add__(self, other: unifloat) -> 'variable[float]':
        ...

    def __add__(self, other: NumLike) -> 'CPNumber':
        return _add_op('add', [self, other], True)

    @overload
    def __radd__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __radd__(self, other: unifloat) -> 'variable[float]':
        ...

    def __radd__(self, other: NumLike) -> 'CPNumber':
        return _add_op('add', [self, other], True)

    @overload
    def __sub__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __sub__(self, other: unifloat) -> 'variable[float]':
        ...

    def __sub__(self, other: NumLike) -> 'CPNumber':
        return _add_op('sub', [self, other])

    @overload
    def __rsub__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rsub__(self, other: unifloat) -> 'variable[float]':
        ...

    def __rsub__(self, other: NumLike) -> 'CPNumber':
        return _add_op('sub', [other, self])

    def __truediv__(self, other: NumLike) -> 'variable[float]':
        return _add_op('div', [self, other])

    def __rtruediv__(self, other: NumLike) -> 'variable[float]':
        return _add_op('div', [other, self])

    @overload
    def __floordiv__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __floordiv__(self, other: unifloat) -> 'variable[float]':
        ...

    def __floordiv__(self, other: NumLike) -> 'CPNumber':
        return _add_op('floordiv', [self, other])

    @overload
    def __rfloordiv__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rfloordiv__(self, other: unifloat) -> 'variable[float]':
        ...

    def __rfloordiv__(self, other: NumLike) -> 'CPNumber':
        return _add_op('floordiv', [other, self])

    def __neg__(self: TNumber) -> TNumber:
        assert isinstance(T, variable)
        return cast(TNumber, _add_op('sub', [variable(0), self]))

    def __gt__(self, other: NumLike) -> 'variable[bool]':
        ret = _add_op('gt', [self, other])
        return variable(ret.source, dtype='bool')

    def __lt__(self, other: NumLike) -> 'variable[bool]':
        ret = _add_op('gt', [other, self])
        return variable(ret.source, dtype='bool')

    def __eq__(self, other: NumLike) -> 'variable[bool]':  # type: ignore
        ret = _add_op('eq', [self, other], True)
        return variable(ret.source, dtype='bool')

    def __ne__(self, other: NumLike) -> 'variable[bool]':  # type: ignore
        ret = _add_op('ne', [self, other], True)
        return variable(ret.source, dtype='bool')

    @overload
    def __mod__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __mod__(self, other: unifloat) -> 'variable[float]':
        ...

    def __mod__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mod', [self, other])

    @overload
    def __rmod__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rmod__(self, other: unifloat) -> 'variable[float]':
        ...

    def __rmod__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mod', [other, self])

    @overload
    def __pow__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __pow__(self, other: unifloat) -> 'variable[float]':
        ...

    def __pow__(self, other: NumLike) -> 'CPNumber':
        return _add_op('pow', [other, self])

    @overload
    def __rpow__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rpow__(self, other: unifloat) -> 'variable[float]':
        ...

    def __rpow__(self, other: NumLike) -> 'CPNumber':
        return _add_op('rpow', [self, other])

    def __hash__(self) -> int:
        return super().__hash__()


class variable(Generic[T], CPNumber):
    def __init__(self, source: T | Node, dtype: str | None = None):
        if isinstance(source, Node):
            self.source = source
            assert dtype, 'For source type Node a dtype argument is required.'
            self.dtype = dtype
        elif isinstance(source, bool):
            self.source = CPConstant(source)
            self.dtype = 'bool'
        elif isinstance(source, int):
            self.source = CPConstant(source)
            self.dtype = 'int'
        elif isinstance(source, float):
            self.source = CPConstant(source)
            self.dtype = 'float'
        else:
            raise ValueError(f'Non supported data type: {type(source).__name__}')

    # Bitwise and shift operations for cp[int]
    def __lshift__(self, other: uniint) -> 'variable[int]':
        return _add_op('lshift', [self, other])

    def __rlshift__(self, other: uniint) -> 'variable[int]':
        return _add_op('lshift', [other, self])

    def __rshift__(self, other: uniint) -> 'variable[int]':
        return _add_op('rshift', [self, other])

    def __rrshift__(self, other: uniint) -> 'variable[int]':
        return _add_op('rshift', [other, self])

    def __and__(self, other: uniint) -> 'variable[int]':
        return _add_op('bwand', [self, other], True)

    def __rand__(self, other: uniint) -> 'variable[int]':
        return _add_op('rwand', [other, self], True)

    def __or__(self, other: uniint) -> 'variable[int]':
        return _add_op('bwor', [self, other], True)

    def __ror__(self, other: uniint) -> 'variable[int]':
        return _add_op('bwor', [other, self], True)

    def __xor__(self, other: uniint) -> 'variable[int]':
        return _add_op('bwxor', [self, other], True)

    def __rxor__(self, other: uniint) -> 'variable[int]':
        return _add_op('bwxor', [other, self], True)


class cpvector:
    def __init__(self, *value: NumLike):
        self.value = value

    def __add__(self, other: 'cpvector') -> 'cpvector':
        assert len(self.value) == len(other.value)
        tup = (a + b for a, b in zip(self.value, other.value))
        return cpvector(*(v for v in tup if isinstance(v, CPNumber)))


class CPConstant(Node):
    def __init__(self, value: int | float):
        self.dtype, self.value = _get_data_and_dtype(value)
        self.name = 'const_' + self.dtype
        self.args = []


class Write(Node):
    def __init__(self, input: NetAndNum):
        if isinstance(input, Net):
            net = input
        else:
            node = CPConstant(input)
            net = Net(node.dtype, node)

        self.name = 'write_' + transl_type(net.dtype)
        self.args = [net]


class Op(Node):
    def __init__(self, typed_op_name: str, args: list[Net]):
        assert not args or any(isinstance(t, Net) for t in args), 'args parameter must be of type list[Net]'
        self.name: str = typed_op_name
        self.args: list[Net] = args


def net_from_value(value: Any) -> Net:
    vi = CPConstant(value)
    return Net(vi.dtype, vi)


@overload
def iif(expression: CPNumber, true_result: unibool, false_result: unibool) -> variable[bool]:  # pyright: ignore[reportOverlappingOverload]
    ...


@overload
def iif(expression: CPNumber, true_result: uniint, false_result: uniint) -> variable[int]:
    ...


@overload
def iif(expression: CPNumber, true_result: unifloat, false_result: unifloat) -> variable[float]:
    ...


@overload
def iif(expression: NumLike, true_result: T, false_result: T) -> T:
    ...


def iif(expression: Any, true_result: Any, false_result: Any) -> Any:
    allowed_type = (variable, int, float, bool)
    assert isinstance(true_result, allowed_type) and isinstance(false_result, allowed_type), "Result type not supported"
    if isinstance(expression, CPNumber):
        return (expression != 0) * true_result + (expression == 0) * false_result
    else:
        return true_result if expression else false_result


def _add_op(op: str, args: list[CPNumber | int | float], commutative: bool = False) -> variable[Any]:
    arg_nets = [a if isinstance(a, Net) else net_from_value(a) for a in args]

    if commutative:
        arg_nets = sorted(arg_nets, key=lambda a: a.dtype)

    typed_op = '_'.join([op] + [transl_type(a.dtype) for a in arg_nets])

    if typed_op not in generic_sdb.stencil_definitions:
        raise NotImplementedError(f"Operation {op} not implemented for {' and '.join([a.dtype for a in arg_nets])}")

    result_type = generic_sdb.stencil_definitions[typed_op].split('_')[0]

    if result_type == 'int':
        return variable[int](Op(typed_op, arg_nets), result_type)
    else:
        return variable[float](Op(typed_op, arg_nets), result_type)


def _get_data_and_dtype(value: Any) -> tuple[str, float | int]:
    if isinstance(value, bool):
        return ('bool', int(value))
    elif isinstance(value, int):
        return ('int', int(value))
    elif isinstance(value, float):
        return ('float', float(value))
    else:
        raise ValueError(f'Non supported data type: {type(value).__name__}')
