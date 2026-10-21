from typing import Iterable, overload, TypeVar, Any, Callable, TypeAlias
from . import _binwrite as binw
from coparun_module import coparun, read_data_mem, data_mem, create_target, clear_target
import importlib
import struct
from ._basic_types import value, Net, Node, Store, NumLike, stencil_db_from_package, ArrayOp, ArrayElement, ArrayConst, ArrayPack, Delay, ArrayDelay, TimeStep, transl_type
from ._arrays import array, ArrayType
from ._compiler import compile_to_dag, get_head_nets
from ._stencils import stencil_database

T = TypeVar("T", int, float)
Values: TypeAlias = 'Iterable[NumLike] | NumLike'

# Cached programs by function and the kinds, types and shapes of the arguments
_jit_cache: dict[Any, '_JitProgram'] = {}

# struct format characters of the elements of arrays
_BUFFER_FORMATS = {'float': 'f', 'int': 'i', 'byte': 'B'}


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


def _buffer_dtype(view: 'memoryview[Any]') -> str | None:
    """Array type for the elements of a buffer, None if they need to be converted"""
    fmt = view.format.lstrip('@=')
    if fmt == 'B':
        return 'byte'
    if fmt == 'f':
        return 'float'
    return 'int' if fmt in ('i', 'l') and view.itemsize == 4 else None


def _as_buffer(arg: Any) -> 'memoryview[Any] | None':
    """Buffer of an argument with elements of an array type (bytes, bytearray,
    memoryview, numpy array ...), None if the argument is no buffer. Numpy arrays
    of other types are converted to 32 bit float or int."""
    try:
        view = memoryview(arg)
    except TypeError:
        return None
    if _buffer_dtype(view) is None:
        kind = view.format.lstrip('@=<>!')
        if not hasattr(arg, 'astype') or kind == '?' or kind not in 'efdgbhHiIlLqQ':
            raise TypeError(f"Unsupported element type '{view.format}' of a {type(arg).__name__} argument")
        view = memoryview(arg.astype('float32' if kind in 'efdg' else 'int32'))
    return view


def jit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Just-in-time compile a function for the copapy target.

    Arguments with the buffer protocol (bytes, bytearray, memoryview,
    numpy arrays) are mapped to copapy arrays. Byte, 32 bit int and
    32 bit float arrays are supported without conversion.

    The function is compiled for each combination of types and shapes of the
    arguments. Arrays are returned as nested lists, arrays marked with
    map_to_buffer or map_to_numpy as memoryview or numpy array.

    Arguments:
        func: Function to compile

    Returns:
        A callable that runs the compiled function.
    """
    def call_helper(*args: Any) -> Any:
        buffers: list['memoryview[Any] | None'] = []
        signature: list[Any] = []
        for a in args:
            if isinstance(a, int | float):
                buffers.append(None)
                signature.append(type(a))
                continue
            b = _as_buffer(a)
            buffers.append(b)
            signature.append((b.format, b.shape) if b is not None else
                             tuple(type(ai) for ai in a) if isinstance(a, Iterable) else type(a))
        key = (func, tuple(signature))
        program = _jit_cache.get(key)
        if program is None:
            program = _JitProgram(func, args, buffers)
            _jit_cache[key] = program
        return program(args, buffers)

    return call_helper


class _JitProgram:
    """Program of a jit function compiled for one signature of the arguments.

    The data memory of the target is not replaced after compiling, so the
    arguments are written directly into it and scalar results read from it.
    """
    def __init__(self, func: Callable[..., Any], args: tuple[Any, ...], buffers: 'list[memoryview[Any] | None]') -> None:
        tg = Target()
        inputs = tuple(
            _placeholder(b) if b is not None else
            tuple(value(ai) for ai in a) if isinstance(a, Iterable) else value(a)
            for a, b in zip(args, buffers))
        out = func(*inputs)
        tg.compile(out)
        self.tg = tg
        self.out = out
        self._mem = tg._memory()
        self._writers = [self._writer(input) for input in inputs]
        self._reader = self._result_reader(out)

    def _slot(self, input: value[Any]) -> 'tuple[struct.Struct, int] | None':
        """Format and address of a scalar input, None if it is not used"""
        net = input.net
        if net not in self.tg._values:
            return None
        addr, size, dtype = self.tg._values[net]
        en = {'little': '<', 'big': '>'}[self.tg.sdb.byteorder]
        fmt = {4: 'f', 8: 'd'}[size] if dtype == 'float' else {1: 'b', 2: 'h', 4: 'i', 8: 'q'}[size]
        return struct.Struct(en + fmt), addr

    def _writer(self, input: Any) -> Callable[[Any, Any], None] | None:
        """Function for writing an argument to the data memory, None if the
        argument is not used by the function"""
        mem = self._mem
        if isinstance(input, array):
            if input.net not in self.tg._values:
                return None
            dest = self.tg.buffer(input).cast('B')

            def write_buffer(arg: Any, buffer: 'memoryview[Any]') -> None:
                dest[:] = buffer.cast('B') if buffer.c_contiguous else buffer.tobytes()
            return write_buffer

        if isinstance(input, tuple):
            slots = [self._slot(v) for v in input]
            if not any(slots):
                return None

            def write_items(arg: Any, buffer: None) -> None:
                for slot, item in zip(slots, arg):
                    if slot is not None:
                        slot[0].pack_into(mem, slot[1], item)
            return write_items

        slot = self._slot(input)
        if slot is None:
            return None
        pack_into, addr = slot[0].pack_into, slot[1]

        def write_scalar(arg: Any, buffer: None) -> None:
            pack_into(mem, addr, arg)
        return write_scalar

    def _result_reader(self, out: Any) -> Callable[[], Any]:
        """Function for reading the result of a run"""
        if isinstance(out, value) and out.net in self.tg._values:
            unpack_from, addr = self._slot(out) or (None, 0)
            assert unpack_from is not None
            mem = self._mem
            if out.dtype == 'bool':
                return lambda: bool(unpack_from.unpack_from(mem, addr)[0])
            return lambda: unpack_from.unpack_from(mem, addr)[0]
        if isinstance(out, array) and not out._export and out.net in self.tg._values:
            addr, _, _ = self.tg._values[out.net]
            unpack_array = struct.Struct(binw.array_format(out.net.dtype, out.size, self.tg.sdb.byteorder)).unpack_from
            mem = self._mem
            if out.dtype == 'bool':
                return lambda: out.get_values([bool(v) for v in unpack_array(mem, addr)])
            return lambda: out.get_values(unpack_array(mem, addr))
        return lambda: self.tg.read_value(out)

    def __call__(self, args: tuple[Any, ...], buffers: 'list[memoryview[Any] | None]') -> Any:
        for writer, arg, buffer in zip(self._writers, args, buffers):
            if writer is not None:
                writer(arg, buffer)
        self.tg.run()
        return self._reader()


def _placeholder(buffer: 'memoryview[Any]') -> array[Any]:
    """Array for the data of a buffer argument"""
    dtype = _buffer_dtype(buffer)
    assert dtype is not None
    size = buffer.nbytes // buffer.itemsize
    return array([0.0 if dtype == 'float' else 0] * size, dtype).reshape(*(buffer.shape or ()))


def _mapped(arr: array[Any] | ArrayType[Any], export: str) -> array[Any]:
    if isinstance(arr, ArrayType):
        arr = arr._force_array()
    if not isinstance(arr, array):
        raise TypeError(f"Only arrays, vectors and tensors can be mapped, not {type(arr).__name__}")
    ret = array._from_net(arr.net, arr.shape, arr.dtype)
    ret._export = export
    return ret


def map_to_buffer(arr: array[T] | ArrayType[T]) -> array[T]:
    """Mark an array, vector or tensor to be read out as memoryview
    instead of nested lists.

    The memoryview is the memory of the array on the target, the data is not
    copied: it changes with the next run.

    Arguments:
        arr: Array, vector or tensor

    Returns:
        A marked array, further operations with it return unmarked arrays
    """
    return _mapped(arr, 'buffer')


def map_to_numpy(arr: array[T] | ArrayType[T], copy: bool = False) -> array[T]:
    """Mark an array, vector or tensor to be returned by read out as
    numpy array (uint8, int32 or float32) instead of nested lists.

    Without copy the numpy array is a view of the memory of the array on the
    target: it changes with the next run.

    Arguments:
        arr: Array, vector or tensor
        copy: Return a copy of the data that is independent of further runs

    Returns:
        The marked array, further operations with it return unmarked arrays
    """
    return _mapped(arr, 'numpy_copy' if copy else 'numpy')


def collect_roots(values: Any) -> list[Node]:
    """End nodes of the DAG for computing and storing the given values

    Arguments:
        values: Values, vectors, tensors and arrays to compute, also in
            (nested) sequences or as values of dicts
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

    def add(input: Any) -> None:
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
        elif isinstance(input, value):
            add_root(input.net)
        elif isinstance(input, dict):
            add(list(input.values()))
        elif isinstance(input, Iterable) and not isinstance(input, str | bytes):
            for v in input:
                add(v)

    add(values)
    return nodes


class _TargetBase():
    """Reading and writing of values on a target. The memory of the target is
    accessed by _read_mem and by commands sent with _send."""
    sdb: stencil_database
    _values: dict[Net, tuple[int, int, str]]  # Address, size and type by net

    def _read_mem(self, addr: int, lengths: int) -> bytes:
        """Reads data memory of the target"""
        raise NotImplementedError()

    def _send(self, dw: binw.data_writer) -> None:
        """Sends commands to the target, END_COM is appended"""
        raise NotImplementedError()

    def _check_writable(self, net: Net) -> None:
        """Raises an error if the net can not be written by write_value"""

    def _release_memory(self) -> None:
        """Called before the memory of the target is freed for a new program"""

    def buffer(self, variable: array[Any]) -> 'memoryview[Any]':
        """Returns memory of an array on the target as memoryview.

        Only available for the local target.
        """
        raise NotImplementedError(f"No buffer access on {type(self).__name__}")

    def _locate_element(self, net: Net) -> bool:
        """Returns if the net is stored on the target. An array element that is
        not accessed by the compiled program is located in the memory of its array."""
        if net not in self._values and isinstance(net.source, ArrayElement):
            array_net = net.source.args[0]
            if array_net in self._values:
                base, _, _ = self._values[array_net]
                size = self.sdb.get_type_size(transl_type(array_net.dtype))
                self._values[net] = (base + net.source.index * size, size, net.dtype)
        return net in self._values

    @overload
    def read_value(self, variables: value[T]) -> T: ...
    @overload
    def read_value(self, variables: NumLike) -> float | int | bool: ...
    @overload
    def read_value(self, variables: ArrayType[T]) -> ArrayType[T]: ...
    @overload
    def read_value(self, variables: Iterable[T | value[T]]) -> list[T]: ...
    @overload
    def read_value(self, variables: array[T]) -> Any: ...
    @overload
    def read_value(self, variables: Any) -> Any: ...
    def read_value(self, variables: Any) -> Any:
        """Reads the numeric value of a copapy type.

        Arguments:
            variables: Variable or multiple variables to read

        Returns:
            Numeric value or values. Arrays are returned as nested lists, arrays marked
            with map_to_buffer or map_to_numpy as memoryview or numpy array. Sequences
            and dicts are returned as lists and dicts of the values of their items.
        """
        if isinstance(variables, array) and variables._export:
            view = self.buffer(variables)
            if variables._export == 'buffer':
                return view
            mapped = importlib.import_module('numpy').asarray(view)
            return mapped.copy() if variables._export == 'numpy_copy' else mapped

        if isinstance(variables, array):
            if variables.net in self._values:
                addr, lengths, _ = self._values[variables.net]
                data = self._read_mem(addr, lengths)
                assert data is not None and len(data) == lengths, f"Failed to read array {variables}"
                flat = struct.unpack(binw.array_format(variables.net.dtype, variables.size, self.sdb.byteorder), data)
            else:
                source = variables.net.source
                assert isinstance(source, ArrayConst), f"Array {variables} not found. It might not have been compiled for the target."
                flat = source.values
            if variables.dtype == 'bool':
                flat = tuple(bool(v) for v in flat)
            return variables.get_values(flat)

        if isinstance(variables, value) and not self._locate_element(variables.net) and \
           isinstance(variables.net.source, ArrayElement):
            # Element of an array that is not stored on the target
            element = variables.net.source
            array_net = element.args[0]
            assert isinstance(array_net.source, ArrayConst), f"Value {variables} not found. It might not have been compiled for the target."
            const = array_net.source.values[element.index]
            return bool(const) if variables.dtype == 'bool' else const

        if isinstance(variables, ArrayType):
            packed = variables._packed_array()
            if packed is not None:
                # Read all elements at once
                array_type: Any = type(variables)
                return array_type(self.read_value(packed))
            return variables.map(lambda v: self.read_value(v))

        if isinstance(variables, dict):
            return {k: self.read_value(v) for k, v in variables.items()}

        if isinstance(variables, Iterable):
            return [self.read_value(ni) if isinstance(ni, value | array | ArrayType | Iterable) else ni for ni in variables]

        if isinstance(variables, float | int):
            return variables

        assert isinstance(variables, value), "Argument must be a copapy value"
        assert variables.net in self._values, f"Value {variables} not found. It might not have been compiled for the target."
        addr, lengths, _ = self._values[variables.net]
        var_type = variables.dtype
        assert lengths > 0
        data = self._read_mem(addr, lengths)
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
            variables: Single variable, array or multiple variables to overwrite
            value: Single value or multiple values to write, (nested) sequences for arrays
        """
        if isinstance(variables, array):
            assert isinstance(data, Iterable), "Data for an array must be a sequence"
            assert variables.net in self._values, f"Array {variables} not found. It might not have been compiled for the target."
            self._check_writable(variables.net)
            addr, lengths, _ = self._values[variables.net]
            flat = array.flatten_data(data)
            assert len(flat) == variables.size, f"Data size {len(flat)} does not match array size {variables.size}"
            conv = float if variables.net.dtype == 'float' else int
            dw = binw.data_writer(self.sdb.byteorder)
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(addr)
            dw.write_int(lengths)
            dw.write_bytes(binw.pack_array([conv(v) for v in flat], variables.net.dtype, self.sdb.byteorder))
            self._send(dw)
            return

        if isinstance(variables, Iterable):
            assert isinstance(data, Iterable), "If net is iterable, value must be iterable too"
            for ni, vi in zip(variables, data):
                self.write_value(ni, vi)
            return

        assert not isinstance(data, Iterable), "If net is not iterable, value must not be iterable"

        assert isinstance(variables, value), "Argument must be a copapy value"
        assert self._locate_element(variables.net), f"Value {variables} not found. It might not have been compiled for the target."
        self._check_writable(variables.net)
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

        self._send(dw)


class _LocalMemory(_TargetBase):
    """Memory access for targets running in this process (coparun_module)"""
    def __init__(self, arch: str = 'native', optimization: str = 'O3') -> None:
        self.sdb = stencil_db_from_package(arch, optimization)
        self._values = {}
        self._data_mem: Any = None  # Data memory of the target with the buffer protocol
        self._context = create_target()

    def __del__(self) -> None:
        if self._data_mem is not None and self._data_mem.exports:
            return  # The memory is still in use (e.g. by a numpy array): it is not freed
        clear_target(self._context)

    def _read_mem(self, addr: int, lengths: int) -> bytes:
        data: bytes = read_data_mem(self._context, addr, lengths)
        return data

    def _send(self, dw: binw.data_writer) -> None:
        dw.write_com(binw.Command.END_COM)
        ret = coparun(self._context, dw.get_data())
        if ret <= 0:
            raise RuntimeError(f"Runner error {ret}")

    def _release_memory(self) -> None:
        """Checks that the data memory can be freed for a new program"""
        if self._data_mem is not None and self._data_mem.exports:
            raise RuntimeError("A buffer or numpy array mapped to the memory of the target is still in use")
        self._data_mem = None

    def buffer(self, variable: array[Any]) -> 'memoryview[Any]':
        """Returns memory of an array on the target as memoryview.

        Only available for the local target. All buffers must be released or
        deleted before a new program is compiled on the same target.

        Arguments:
            variable: Array-variable to get the memory of

        Returns:
            Writable memoryview of the array memory
        """
        net = variable.net
        if net not in self._values:
            raise ValueError(f"Array {variable} not found. It might not have been compiled for the target or is not used.")
        addr, lengths, _ = self._values[net]
        view: Any = self._memory()[addr:addr + lengths]
        ret: memoryview[Any] = view.cast(_BUFFER_FORMATS[net.dtype], variable.shape)
        return ret

    def _memory(self) -> 'memoryview[Any]':
        """Writable memoryview of the whole data memory, like buffer
        it must be released before a new program is compiled"""
        if self._data_mem is None:
            self._data_mem = data_mem(self._context)
        return memoryview(self._data_mem)


class Target(_LocalMemory):
    """Target device for compiling for and running on copapy code.
    """
    def __init__(self, arch: str = 'native', optimization: str = 'O3') -> None:
        """Initialize Target object

        Arguments:
            arch: Target architecture
            optimization: Optimization level
        """
        super().__init__(arch, optimization)
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.RUN_PROG)
        dw.write_com(binw.Command.END_COM)
        self._run_command = dw.get_data()

    def compile(self, *values: Any) -> None:
        """Compiles the code to compute the given values.

        Arguments:
            values: Values, vectors, tensors and arrays to compute, also in
                (nested) sequences or as values of dicts
        """
        nodes = collect_roots(values)
        if get_head_nets(nodes, (Delay, ArrayDelay, TimeStep)):
            raise ValueError("delay, integrate, feedback and dt require tasks: "
                             "deploy the values on a LocalTarget or SerialTarget")
        self._release_memory()
        dw, self._values = compile_to_dag(nodes, self.sdb)
        self._send(dw)

    def run(self) -> None:
        """Runs the compiled code on the target device.
        """
        ret = coparun(self._context, self._run_command)
        if ret <= 0:
            raise RuntimeError(f"Runner error {ret}")

    def read_value_remote(self, variable: value[Any]) -> None:
        """Reads the raw data of a value by the runner."""
        dw = binw.data_writer(self.sdb.byteorder)
        add_read_value_remote(dw, self._values, variable.net)
        self._send(dw)
