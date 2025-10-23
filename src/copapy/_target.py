from typing import Iterable, overload
from . import _binwrite as binw
from coparun_module import coparun, read_data_mem
import struct
from ._basic_types import stencil_db_from_package
from ._basic_types import cpbool, cpint, cpfloat, Net, Node, Write, NumLike
from ._compiler import compile_to_instruction_list


def add_read_command(dw: binw.data_writer, variables: dict[Net, tuple[int, int, str]], net: Net) -> None:
    assert net in variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = variables[net]
    dw.write_com(binw.Command.READ_DATA)
    dw.write_int(addr)
    dw.write_int(lengths)


class Target():
    def __init__(self, arch: str = 'native', optimization: str = 'O3') -> None:
        self.sdb = stencil_db_from_package(arch, optimization)
        self._variables: dict[Net, tuple[int, int, str]] = dict()

    def compile(self, *variables: int | float | cpint | cpfloat | cpbool | Iterable[int | float | cpint | cpfloat | cpbool]) -> None:
        nodes: list[Node] = []
        for s in variables:
            if isinstance(s, Iterable):
                for net in s:
                    assert isinstance(net, Net), f"The folowing element is not a Net: {net}"
                    nodes.append(Write(net))
            else:
                nodes.append(Write(s))

        dw, self._variables = compile_to_instruction_list(nodes, self.sdb)
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

    def run(self) -> None:
        # set entry point and run code
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_PROG)
        dw.write_com(binw.Command.END_COM)
        assert coparun(dw.get_data()) > 0

    @overload
    def read_value(self, net: cpbool) -> bool:
        ...

    @overload
    def read_value(self, net: cpfloat) -> float:
        ...

    @overload
    def read_value(self, net: cpint) -> int:
        ...

    @overload
    def read_value(self, net: NumLike) -> float | int | bool:
        ...

    def read_value(self, net: NumLike) -> float | int | bool:
        assert isinstance(net, Net), "Variable must be a copapy variable object"
        assert net in self._variables, f"Variable {net} not found"
        addr, lengths, var_type = self._variables[net]
        assert lengths > 0
        data = read_data_mem(addr, lengths)
        assert data is not None and len(data) == lengths, f"Failed to read variable {net}"
        en = {'little': '<', 'big': '>'}[self.sdb.byteorder]
        if var_type == 'float':
            if lengths == 4:
                value = struct.unpack(en + 'f', data)[0]
            elif lengths == 8:
                value = struct.unpack(en + 'd', data)[0]
            else:
                raise ValueError(f"Unsupported float length: {lengths} bytes")
            assert isinstance(value, float)
            return value
        elif var_type == 'int':
            assert lengths in (1, 2, 4, 8), f"Unsupported int length: {lengths} bytes"
            value = int.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
            return value
        elif var_type == 'bool':
            assert lengths in (1, 2, 4, 8), f"Unsupported int length: {lengths} bytes"
            value = bool.from_bytes(data, byteorder=self.sdb.byteorder, signed=True)
            return value
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def read_value_remote(self, net: Net) -> None:
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_command(dw, self._variables, net)
        assert coparun(dw.get_data()) > 0
