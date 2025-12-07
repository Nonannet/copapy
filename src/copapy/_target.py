from typing import Iterable, overload, TypeVar, Any
from . import _binwrite as binw
from coparun_module import coparun, read_data_mem
import struct
from ._basic_types import stencil_db_from_package
from ._basic_types import value, Net, Node, Write, NumLike
from ._compiler import compile_to_dag

T = TypeVar("T", int, float)


def add_read_command(dw: binw.data_writer, variables: dict[Net, tuple[int, int, str]], net: Net) -> None:
    assert net in variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = variables[net]
    dw.write_com(binw.Command.READ_DATA)
    dw.write_int(addr)
    dw.write_int(lengths)


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
        assert coparun(dw.get_data()) > 0

    def run(self) -> None:
        """Runs the compiled code on the target device.
        """
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_PROG)
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

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
        data = read_data_mem(addr, lengths)
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

    def read_value_remote(self, net: Net) -> None:
        """Reads the raw data of a value by the runner."""
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_command(dw, self._values, net)
        assert coparun(dw.get_data()) > 0
