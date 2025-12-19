from typing import Iterable, overload, TypeVar, Any, Callable, TypeAlias
from . import _binwrite as binw
from coparun_module import coparun, read_data_mem, create_target, clear_target
import struct
from ._basic_types import stencil_db_from_package
from ._basic_types import value, Net, Node, Write, NumLike
from ._compiler import compile_to_dag

T = TypeVar("T", int, float)
Values: TypeAlias = 'Iterable[NumLike] | NumLike'
ArgType: TypeAlias = int | float | Iterable[int | float]
TRet = TypeVar("TRet", Iterable[int | float], int, float)

_jit_cache: dict[Any, tuple['Target', tuple[value[Any] | Iterable[value[Any]], ...], NumLike | Iterable[NumLike]]] = {}


def add_read_command(dw: binw.data_writer, variables: dict[Net, tuple[int, int, str]], net: Net) -> None:
    assert net in variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = variables[net]
    dw.write_com(binw.Command.READ_DATA)
    dw.write_int(addr)
    dw.write_int(lengths)


def jit(func: Callable[..., TRet]) -> Callable[..., TRet]:
    def call_helper(*args: ArgType) -> TRet:
        if func in _jit_cache:
            tg, inputs, out = _jit_cache[func]
            for input, arg in zip(inputs, args):
                tg.write_value(input, arg)
        else:
            tg = Target()
            inputs = tuple(
                tuple(value(ai) for ai in a) if isinstance(a, Iterable) else value(a) for a in args)
            out = func(*inputs)  # type: ignore
            tg.compile(out)
            _jit_cache[func] = (tg, inputs, out)
        tg.run()
        return tg.read_value(out)  # type: ignore

    return call_helper


class Target():
    """Target device for compiling for and running on copapy code.
    """
    def __init__(self, arch: str = 'native', optimization: str = 'O3') -> None:
        """Initialize Target object

        Arguments:
            arch: Target architecture
            optimization: Optimization level
        """
        self.sdb = stencil_db_from_package(arch, optimization)
        self._values: dict[Net, tuple[int, int, str]] = {}
        self._context = create_target()

    def __del__(self) -> None:
        clear_target(self._context)

    def compile(self, *values: int | float | value[int] | value[float] | Iterable[int | float | value[int] | value[float]]) -> None:
        """Compiles the code to compute the given values.

        Arguments:
            values: Values to compute
        """
        nodes: list[Node] = []
        for s in values:
            if isinstance(s, Iterable):
                for net in s:
                    if isinstance(net, Net):
                        nodes.append(Write(net))
            else:
                if isinstance(s, Net):
                    nodes.append(Write(s))

        dw, self._values = compile_to_dag(nodes, self.sdb)
        dw.write_com(binw.Command.END_COM)
        assert coparun(self._context, dw.get_data()) > 0

    def run(self) -> None:
        """Runs the compiled code on the target device.
        """
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_PROG)
        dw.write_com(binw.Command.END_COM)
        assert coparun(self._context, dw.get_data()) > 0

    @overload
    def read_value(self, net: value[T]) -> T: ...
    @overload
    def read_value(self, net: NumLike) -> float | int | bool: ...
    @overload
    def read_value(self, net: Iterable[T | value[T]]) -> list[T]: ...
    def read_value(self, net: NumLike | value[T] | Iterable[T | value[T]]) -> Any:
        """Reads the numeric value of a copapy type.

        Arguments:
            net: Values to read

        Returns:
            Numeric value
        """
        if isinstance(net, Iterable):
            return [self.read_value(ni) if isinstance(ni, value) else ni for ni in net]

        if isinstance(net, float | int):
            print("Warning: value is not a copypy value")
            return net

        assert isinstance(net, Net), "Argument must be a copapy value"
        assert net in self._values, f"Value {net} not found. It might not have been compiled for the target."
        addr, lengths, var_type = self._values[net]
        assert lengths > 0
        data = read_data_mem(self._context, addr, lengths)
        assert data is not None and len(data) == lengths, f"Failed to read value {net}"
        en = {'little': '<', 'big': '>'}[self.sdb.byteorder]
        if var_type == 'float':
            if lengths == 4:
                val = struct.unpack(en + 'f', data)[0]
            elif lengths == 8:
                val = struct.unpack(en + 'd', data)[0]
            else:
                raise ValueError(f"Unsupported float length: {lengths} bytes")
            assert isinstance(val, float)
            return val
        elif var_type == 'int':
            assert lengths in (1, 2, 4, 8), f"Unsupported int length: {lengths} bytes"
            val = int.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
            return val
        elif var_type == 'bool':
            assert lengths in (1, 2, 4, 8), f"Unsupported int length: {lengths} bytes"
            val = bool.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
            return val
        else:
            raise ValueError(f"Unsupported value type: {var_type}")
        
    def write_value(self, net: value[Any] | Iterable[value[Any]], value: int | float | Iterable[int | float]) -> None:
        """Reads the numeric value of a copapy type.

        Arguments:
            net: Variable to overwrite
            value: Value
        """
        if isinstance(net, Iterable):
            assert isinstance(value, Iterable), "If net is iterable, value must be iterable too"
            for ni, vi in zip(net, value):
                self.write_value(ni, vi)
            return
        
        assert not isinstance(value, Iterable), "If net is not iterable, value must not be iterable"

        assert isinstance(net, Net), "Argument must be a copapy value"
        assert net in self._values, f"Value {net} not found. It might not have been compiled for the target."
        addr, lengths, var_type = self._values[net]
        assert lengths > 0

        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(addr)
        dw.write_int(lengths)

        if var_type == 'float':
            dw.write_value(float(value), lengths)
        elif var_type == 'int' or var_type == 'bool':
            dw.write_value(int(value), lengths)
        else:
            raise ValueError(f"Unsupported value type: {var_type}")
        
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

    def read_value_remote(self, net: Net) -> None:
        """Reads the raw data of a value by the runner."""
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_command(dw, self._values, net)
        assert coparun(self._context, dw.get_data()) > 0
