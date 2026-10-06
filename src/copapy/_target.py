from typing import Iterable, overload, TypeVar, Any, Callable, TypeAlias
from . import _binwrite as binw
from coparun_module import coparun, read_data_mem, create_target, clear_target
import struct
from ._basic_types import value, Net, Node, Store, NumLike, ArrayType, stencil_db_from_package, ArrayOp, ArrayElement, ArrayConst, ArrayPack, transl_type
from ._arrays import array
from ._compiler import compile_to_dag

T = TypeVar("T", int, float)
Values: TypeAlias = 'Iterable[NumLike] | NumLike'
ArgType: TypeAlias = int | float | Iterable[int | float]
TRet = TypeVar("TRet", Iterable[int | float], int, float)

_jit_cache: dict[Any, tuple['Target', tuple[value[Any] | Iterable[value[Any]], ...], NumLike | Iterable[NumLike]]] = {}


def add_read_value_remote(dw: binw.data_writer, variables: dict[Net, tuple[int, int, str]], net: Net) -> None:
    """Adds a read memory to stdout command to the data_writer dw.

    Arguments:
        dw: data_writer to add the command to
        variables: A dict for looking up variables by Net. The value is a tuple.
            of relative address in memory, size in bytes and data type.
        net: Variable specified by Net to read from memory and write to stdout.
    """
    assert net in variables, f"Variable {net} not found in data writer variables"
    addr, lengths, _ = variables[net]
    dw.write_com(binw.Command.READ_DATA)
    dw.write_int(addr)
    dw.write_int(lengths)


def jit(func: Callable[..., TRet]) -> Callable[..., TRet]:
    """Just-in-time compile a function for the copapy target.

    Arguments:
        func: Function to compile

    Returns:
        A callable that runs the compiled function.
    """
    def call_helper(*args: ArgType) -> TRet:
        if func in _jit_cache:
            tg, inputs, out = _jit_cache[func]
            for input, arg in zip(inputs, args):
                tg.write_value(input, arg)
        else:
            tg = Target()
            inputs = tuple(
                tuple(value(ai) for ai in a) if isinstance(a, Iterable) else value(a) for a in args)
            out = func(*inputs)
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

    def compile(self, *values: NumLike | value[T] | ArrayType[T] | Iterable[T | value[T]]) -> None:
        """Compiles the code to compute the given values.

        Arguments:
            values: Values to compute
        """
        nodes: list[Node] = []

        def add_root(net: Net) -> None:
            if isinstance(net.source, ArrayElement):
                # Element is an alias to the array memory
                net = net.source.args[0]
            if isinstance(net.source, ArrayOp | ArrayPack):
                # Result is already written to heap by the array op or pack
                nodes.append(net.source)
            elif not isinstance(net.source, ArrayConst):
                nodes.append(Store(net))

        for input in values:
            if isinstance(input, array):
                add_root(input.net)
            elif isinstance(input, ArrayType):
                packed = input._packed_array()
                if packed is not None:
                    add_root(packed.net)
                else:
                    for v in input.values:
                        if isinstance(v, value):
                            add_root(v.net)
            elif isinstance(input, Iterable):
                for v in input:
                    if isinstance(v, value):
                        add_root(v.net)
            elif isinstance(input, value):
                add_root(input.net)

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
    def read_value(self, variables: value[T]) -> T: ...
    @overload
    def read_value(self, variables: NumLike) -> float | int | bool: ...
    @overload
    def read_value(self, variables: Iterable[T | value[T]]) -> list[T]: ...
    @overload
    def read_value(self, variables: ArrayType[T]) -> ArrayType[T]: ...
    @overload
    def read_value(self, variables: array[T]) -> list[Any]: ...
    def read_value(self, variables: NumLike | value[T] | ArrayType[T] | array[T] | Iterable[T | value[T]]) -> Any:
        """Reads the numeric value of a copapy type.

        Arguments:
            variables: Variable or multiple variables to read

        Returns:
            Numeric value or values, arrays are returned as nested lists
        """
        if isinstance(variables, array):
            if variables.net in self._values:
                addr, lengths, _ = self._values[variables.net]
                data = read_data_mem(self._context, addr, lengths)
                assert data is not None and len(data) == lengths, f"Failed to read array {variables}"
                flat = struct.unpack(binw.array_format(variables.net.dtype, variables.size, self.sdb.byteorder), data)
            else:
                source = variables.net.source
                assert isinstance(source, ArrayConst), f"Array {variables} not found. It might not have been compiled for the target."
                flat = source.values
            if variables.dtype == 'bool':
                flat = tuple(bool(v) for v in flat)
            return variables.get_values(flat)

        if isinstance(variables, value) and variables.net not in self._values and \
           isinstance(variables.net.source, ArrayElement):
            # Element not accessed by the compiled program, read it from its array
            element = variables.net.source
            array_net = element.args[0]
            if array_net in self._values:
                base, _, _ = self._values[array_net]
                size = self.sdb.get_type_size(transl_type(array_net.dtype))
                self._values[variables.net] = (base + element.index * size, size, array_net.dtype)
            else:
                assert isinstance(array_net.source, ArrayConst), f"Value {variables} not found. It might not have been compiled for the target."
                const = array_net.source.values[element.index]
                return bool(const) if variables.dtype == 'bool' else const

        if isinstance(variables, ArrayType):
            packed = variables._packed_array()
            if packed is not None:
                # Read all elements at once
                return type(variables)(self.read_value(packed))
            return variables.map(lambda v: self.read_value(v))

        if isinstance(variables, Iterable):
            return [self.read_value(ni) if isinstance(ni, value) else ni for ni in variables]

        if isinstance(variables, float | int):
            return variables

        assert isinstance(variables, value), "Argument must be a copapy value"
        assert variables.net in self._values, f"Value {variables} not found. It might not have been compiled for the target."
        addr, lengths, _ = self._values[variables.net]
        var_type = variables.dtype
        assert lengths > 0
        data = read_data_mem(self._context, addr, lengths)
        assert data is not None and len(data) == lengths, f"Failed to read value {variables}"
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

    def write_value(self, variables: value[Any] | array[Any] | Iterable[value[Any]], data: int | float | Iterable[Any]) -> None:
        """Write to a copapy value on the target.

        Arguments:
            variables: Singe variable, array or multiple variables to overwrite
            value: Singe value or multiple values to write, (nested) sequences for arrays
        """
        if isinstance(variables, array):
            assert isinstance(data, Iterable), "Data for an array must be a sequence"
            assert variables.net in self._values, f"Array {variables} not found. It might not have been compiled for the target."
            addr, lengths, _ = self._values[variables.net]
            flat = array.flatten_data(data)
            assert len(flat) == variables.size, f"Data size {len(flat)} does not match array size {variables.size}"
            conv = float if variables.net.dtype == 'float' else int
            dw = binw.data_writer(self.sdb.byteorder)
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(addr)
            dw.write_int(lengths)
            dw.write_bytes(binw.pack_array([conv(v) for v in flat], variables.net.dtype, self.sdb.byteorder))
            dw.write_com(binw.Command.END_COM)
            assert coparun(self._context, dw.get_data()) > 0
            return

        if isinstance(variables, Iterable):
            assert isinstance(data, Iterable), "If net is iterable, value must be iterable too"
            for ni, vi in zip(variables, data):
                self.write_value(ni, vi)
            return

        assert not isinstance(data, Iterable), "If net is not iterable, value must not be iterable"

        assert isinstance(variables, value), "Argument must be a copapy value"
        assert variables.net in self._values, f"Value {variables} not found. It might not have been compiled for the target."
        addr, lengths, var_type = self._values[variables.net]
        assert lengths > 0

        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(addr)
        dw.write_int(lengths)

        if var_type == 'float':
            dw.write_value(float(data), lengths)
        elif var_type == 'int' or var_type == 'bool':
            dw.write_value(int(data), lengths)
        else:
            raise ValueError(f"Unsupported value type: {var_type}")

        dw.write_com(binw.Command.END_COM)
        assert coparun(self._context, dw.get_data()) > 0

    def read_value_remote(self, variable: value[Any]) -> None:
        """Reads the raw data of a value by the runner."""
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_value_remote(dw, self._values, variable.net)
        assert coparun(self._context, dw.get_data()) > 0
