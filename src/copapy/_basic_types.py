import pkgutil
from typing import Any, TypeVar, overload, TypeAlias
from ._stencils import stencil_database
import platform

NumLike: TypeAlias = 'cpint | cpfloat | cpbool | int | float| bool'
NumLikeAndNet: TypeAlias = 'cpint | cpfloat | cpbool | int | float | bool | Net'
NetAndNum: TypeAlias = 'Net | int | float'

unifloat: TypeAlias = 'cpfloat | float'
uniint: TypeAlias = 'cpint | int'
unibool: TypeAlias = 'cpbool | bool'

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
        return f"Node:{self.name}({', '.join(str(a) for a in self.args) if self.args else (self.value if isinstance(self, InitVar) else '')})"


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
    def __mul__(self, other: unifloat) -> 'cpfloat':
        ...

    def __mul__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mul', [self, other], True)

    @overload
    def __rmul__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rmul__(self, other: unifloat) -> 'cpfloat':
        ...

    def __rmul__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mul', [self, other], True)

    @overload
    def __add__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __add__(self, other: unifloat) -> 'cpfloat':
        ...

    def __add__(self, other: NumLike) -> 'CPNumber':
        return _add_op('add', [self, other], True)

    @overload
    def __radd__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __radd__(self, other: unifloat) -> 'cpfloat':
        ...

    def __radd__(self, other: NumLike) -> 'CPNumber':
        return _add_op('add', [self, other], True)

    @overload
    def __sub__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __sub__(self, other: unifloat) -> 'cpfloat':
        ...

    def __sub__(self, other: NumLike) -> 'CPNumber':
        return _add_op('sub', [self, other])

    @overload
    def __rsub__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rsub__(self, other: unifloat) -> 'cpfloat':
        ...

    def __rsub__(self, other: NumLike) -> 'CPNumber':
        return _add_op('sub', [other, self])

    def __truediv__(self, other: NumLike) -> 'cpfloat':
        ret = _add_op('div', [self, other])
        assert isinstance(ret, cpfloat)
        return ret

    def __rtruediv__(self, other: NumLike) -> 'cpfloat':
        ret = _add_op('div', [other, self])
        assert isinstance(ret, cpfloat)
        return ret

    @overload
    def __floordiv__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __floordiv__(self, other: unifloat) -> 'cpfloat':
        ...

    def __floordiv__(self, other: NumLike) -> 'CPNumber':
        return _add_op('floordiv', [self, other])

    @overload
    def __rfloordiv__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rfloordiv__(self, other: unifloat) -> 'cpfloat':
        ...

    def __rfloordiv__(self, other: NumLike) -> 'CPNumber':
        return _add_op('floordiv', [other, self])

    def __neg__(self: TNumber) -> TNumber:
        ret = _add_op('sub', [cpvalue(0), self])
        assert isinstance(ret, type(self))
        return ret

    def __gt__(self, other: NumLike) -> 'cpbool':
        ret = _add_op('gt', [self, other])
        return cpbool(ret.source)

    def __lt__(self, other: NumLike) -> 'cpbool':
        ret = _add_op('gt', [other, self])
        return cpbool(ret.source)

    def __eq__(self, other: NumLike) -> 'cpbool':  # type: ignore
        ret = _add_op('eq', [self, other], True)
        return cpbool(ret.source)

    def __ne__(self, other: NumLike) -> 'cpbool':  # type: ignore
        ret = _add_op('ne', [self, other], True)
        return cpbool(ret.source)

    @overload
    def __mod__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __mod__(self, other: unifloat) -> 'cpfloat':
        ...

    def __mod__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mod', [self, other])

    @overload
    def __rmod__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rmod__(self, other: unifloat) -> 'cpfloat':
        ...

    def __rmod__(self, other: NumLike) -> 'CPNumber':
        return _add_op('mod', [other, self])

    @overload
    def __pow__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __pow__(self, other: unifloat) -> 'cpfloat':
        ...

    def __pow__(self, other: NumLike) -> 'CPNumber':
        return _add_op('pow', [other, self])

    @overload
    def __rpow__(self: TNumber, other: uniint) -> TNumber:
        ...

    @overload
    def __rpow__(self, other: unifloat) -> 'cpfloat':
        ...

    def __rpow__(self, other: NumLike) -> 'CPNumber':
        return _add_op('rpow', [self, other])

    def __hash__(self) -> int:
        return super().__hash__()


class cpint(CPNumber):
    def __init__(self, source: int | Node):
        if isinstance(source, Node):
            self.source = source
        else:
            self.source = InitVar(int(source))
        self.dtype = 'int'

    def __lshift__(self, other: uniint) -> 'cpint':
        ret = _add_op('lshift', [self, other])
        assert isinstance(ret, cpint)
        return ret

    def __rlshift__(self, other: uniint) -> 'cpint':
        ret = _add_op('lshift', [other, self])
        assert isinstance(ret, cpint)
        return ret

    def __rshift__(self, other: uniint) -> 'cpint':
        ret = _add_op('rshift', [self, other])
        assert isinstance(ret, cpint)
        return ret

    def __rrshift__(self, other: uniint) -> 'cpint':
        ret = _add_op('rshift', [other, self])
        assert isinstance(ret, cpint)
        return ret

    def __and__(self, other: uniint) -> 'cpint':
        ret = _add_op('bwand', [self, other], True)
        assert isinstance(ret, cpint)
        return ret

    def __rand__(self, other: uniint) -> 'cpint':
        ret = _add_op('rwand', [other, self], True)
        assert isinstance(ret, cpint)
        return ret

    def __or__(self, other: uniint) -> 'cpint':
        ret = _add_op('bwor', [self, other], True)
        assert isinstance(ret, cpint)
        return ret

    def __ror__(self, other: uniint) -> 'cpint':
        ret = _add_op('bwor', [other, self], True)
        assert isinstance(ret, cpint)
        return ret

    def __xor__(self, other: uniint) -> 'cpint':
        ret = _add_op('bwxor', [self, other], True)
        assert isinstance(ret, cpint)
        return ret

    def __rxor__(self, other: uniint) -> 'cpint':
        ret = _add_op('bwxor', [other, self], True)
        assert isinstance(ret, cpint)
        return ret


class cpfloat(CPNumber):
    def __init__(self, source: float | Node | CPNumber):
        if isinstance(source, Node):
            self.source = source
        elif isinstance(source, CPNumber):
            self.source = _add_op('cast_float', [source]).source
        else:
            self.source = InitVar(float(source))
        self.dtype = 'float'


class cpbool(cpint):
    def __init__(self, source: bool | Node):
        if isinstance(source, Node):
            self.source = source
        else:
            self.source = InitVar(bool(source))
        self.dtype = 'bool'


class InitVar(Node):
    def __init__(self, value: int | float):
        self.dtype, self.value = _get_data_and_dtype(value)
        self.name = 'const_' + self.dtype
        self.args = []


class Write(Node):
    def __init__(self, input: NetAndNum):
        if isinstance(input, Net):
            net = input
        else:
            node = InitVar(input)
            net = Net(node.dtype, node)

        self.name = 'write_' + transl_type(net.dtype)
        self.args = [net]


class Op(Node):
    def __init__(self, typed_op_name: str, args: list[Net]):
        assert not args or any(isinstance(t, Net) for t in args), 'args parameter must be of type list[Net]'
        self.name: str = typed_op_name
        self.args: list[Net] = args


def net_from_value(value: Any) -> Net:
    vi = InitVar(value)
    return Net(vi.dtype, vi)


@overload
def iif(expression: CPNumber, true_result: unibool, false_result: unibool) -> cpbool:  # pyright: ignore[reportOverlappingOverload]
    ...


@overload
def iif(expression: CPNumber, true_result: uniint, false_result: uniint) -> cpint:
    ...


@overload
def iif(expression: CPNumber, true_result: unifloat, false_result: unifloat) -> cpfloat:
    ...


@overload
def iif(expression: NumLike, true_result: T, false_result: T) -> T:
    ...


def iif(expression: Any, true_result: Any, false_result: Any) -> Any:
    # TODO: check that input types are matching
    alowed_type = cpint | cpfloat | cpbool | int | float | bool
    assert isinstance(true_result, alowed_type) and isinstance(false_result, alowed_type), "Result type not supported"
    if isinstance(expression, CPNumber):
        return (expression != 0) * true_result + (expression == 0) * false_result
    else:
        return true_result if expression else false_result


def _add_op(op: str, args: list[CPNumber | int | float], commutative: bool = False) -> CPNumber:
    arg_nets = [a if isinstance(a, Net) else net_from_value(a) for a in args]

    if commutative:
        arg_nets = sorted(arg_nets, key=lambda a: a.dtype)

    typed_op = '_'.join([op] + [transl_type(a.dtype) for a in arg_nets])

    if typed_op not in generic_sdb.stencil_definitions:
        raise NotImplementedError(f"Operation {op} not implemented for {' and '.join([a.dtype for a in arg_nets])}")

    result_type = generic_sdb.stencil_definitions[typed_op].split('_')[0]

    if result_type == 'int':
        return cpint(Op(typed_op, arg_nets))
    else:
        return cpfloat(Op(typed_op, arg_nets))


@overload
def cpvalue(value: bool) -> cpbool:  # pyright: ignore[reportOverlappingOverload]
    ...


@overload
def cpvalue(value: int) -> cpint:
    ...


@overload
def cpvalue(value: float) -> cpfloat:
    ...


def cpvalue(value: bool | int | float) -> cpbool | cpint | cpfloat:
    vi = InitVar(value)

    if isinstance(value, bool):
        return cpbool(vi)
    elif isinstance(value, float):
        return cpfloat(vi)
    else:
        return cpint(vi)


def _get_data_and_dtype(value: Any) -> tuple[str, float | int]:
    if isinstance(value, bool):
        return ('bool', int(value))
    elif isinstance(value, int):
        return ('int', int(value))
    elif isinstance(value, float):
        return ('float', float(value))
    else:
        raise ValueError(f'Non supported data type: {type(value).__name__}')
