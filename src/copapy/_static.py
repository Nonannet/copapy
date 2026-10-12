import os
import re
import sys
from typing import Any, NamedTuple
import coparun_module as runner
from . import _binwrite as binw
from ._basic_types import value, Net, Input, HeadNode, ArrayNet, ArrayConst, ArrayElement, stencil_db_from_package, transl_type
from ._stencils import stencil_database
from ._arrays import array, ArrayType
from ._compiler import compile_to_dag
from ._target import root_nodes

# C types of the elements by copapy type (bool is stored as int)
_C_TYPES = {'float': 'float', 'int': 'int32_t', 'bool': 'int32_t', 'byte': 'uint8_t'}

# Alignment of the code and data memory expected by the memory layout
MEM_ALIGNMENT = 16


class StaticProgram(NamedTuple):
    """Result of export_static"""
    c_file: str
    h_file: str
    code_size: int  # Size of the code (read only) in bytes
    data_size: int  # Size of the data memory (RAM) in bytes


class _Const(NamedTuple):
    """Element with a value known at compile time, it is not in the data memory"""
    value: int | float


# Location of an element: offset in the data memory, constant or None if not in memory
_Loc = int | _Const | None


class _Field(NamedTuple):
    """Member of the input or output struct"""
    name: str
    dtype: str
    shape: tuple[int, ...]  # Empty for scalars
    locs: list[_Loc]  # Location of each element (row-major)


def _array_locs(net: Net, size: int, variables: dict[Net, tuple[int, int, str]],
                sdb: stencil_database, name: str, is_input: bool) -> list[_Loc]:
    """Locations of all elements of an array"""
    if not isinstance(net, ArrayNet) or net.length != size:
        raise TypeError(f"'{name}': the array must cover its whole memory (no slice or view)")
    if is_input and not isinstance(net.source, ArrayConst):
        raise TypeError(f"Input '{name}' must be an array of numbers, not a computed array")
    if net in variables:
        elem_size = sdb.get_type_size(net.dtype)
        return [variables[net][0] + i * elem_size for i in range(size)]
    if is_input:
        return [None] * size  # Not used by the program
    if isinstance(net.source, ArrayConst):
        return [_Const(v) for v in net.source.values]
    raise ValueError(f"Array '{name}' not found in the compiled program")


def _value_loc(v: Any, variables: dict[Net, tuple[int, int, str]],
               sdb: stencil_database, name: str, is_input: bool) -> _Loc:
    """Location of a scalar"""
    if is_input:
        if not isinstance(v, value) or not isinstance(v.net.source, Input):
            raise TypeError(f"Input '{name}' must be a cp.value created from a number (or a vector, tensor or array of them)")
        return variables[v.net][0] if v.net in variables else None  # None: not used by the program

    if isinstance(v, int | float):
        return _Const(v)
    if not isinstance(v, value):
        raise TypeError(f"Output '{name}': unsupported type {type(v).__name__}")
    net = v.net
    if net in variables:
        return variables[net][0]
    source = net.source
    if isinstance(source, ArrayElement):
        # Element not accessed by the compiled program, it is located in its array
        array_net = source.args[0]
        if array_net in variables:
            return variables[array_net][0] + source.index * sdb.get_type_size(array_net.dtype)
        if isinstance(array_net.source, ArrayConst):
            return _Const(array_net.source.values[source.index])
    if isinstance(source, HeadNode):
        return _Const(source.value)
    raise ValueError(f"Output '{name}' not found in the compiled program")


def _field(name: str, obj: Any, variables: dict[Net, tuple[int, int, str]],
           sdb: stencil_database, is_input: bool) -> _Field:
    if isinstance(obj, ArrayType):
        packed = obj._packed_array()
        if packed is None:
            locs = [_value_loc(v, variables, sdb, name, is_input) for v in obj.values]
            return _Field(name, obj.dtype, obj.shape, locs)
        obj = packed
    if isinstance(obj, array):
        return _Field(name, obj.dtype, obj.shape, _array_locs(obj.net, obj.size, variables, sdb, name, is_input))
    dtype = obj.dtype if isinstance(obj, value) else 'float' if isinstance(obj, float) else 'int'
    return _Field(name, dtype, (), [_value_loc(obj, variables, sdb, name, is_input)])


def _struct(type_name: str, fields: list[_Field]) -> str:
    members = ''.join(f"    {_C_TYPES[f.dtype]} {f.name}{''.join(f'[{n}]' for n in f.shape)};\n" for f in fields)
    return f"typedef struct {{\n{members}}} {type_name};\n"


def _literal(val: int | float, dtype: str) -> str:
    if dtype != 'float':
        return str(int(val))
    text = repr(float(val))
    if text in ('inf', '-inf', 'nan'):
        raise ValueError(f"Constant {text} can not be exported")
    return text + 'f'


def _copies(fields: list[_Field], sdb: stencil_database, struct_var: str, mem: str, to_mem: bool) -> list[str]:
    """C statements copying between the struct members and the data memory.
    Elements located contiguously in the data memory are copied at once."""
    lines: list[str] = []
    for f in fields:
        elem_size = sdb.get_type_size(transl_type(f.dtype))
        member = f"{struct_var}->{f.name}"
        if not any(loc is not None for loc in f.locs):
            lines.append(f"/* {f.name} is not used by the program */")
        i = 0
        while i < len(f.locs):
            loc = f.locs[i]
            if loc is None:
                i += 1
            elif isinstance(loc, _Const):
                index = ''
                rest = i
                for n in reversed(f.shape):
                    index = f"[{rest % n}]" + index
                    rest //= n
                lines.append(f"{member}{index} = {_literal(loc.value, f.dtype)};")
                i += 1
            else:
                j = i + 1
                while j < len(f.locs) and f.locs[j] == loc + (j - i) * elem_size:
                    j += 1
                if not f.shape:
                    ptr = '&' + member
                elif i == 0:
                    ptr = member
                else:
                    ptr = f"({'const ' if to_mem else ''}uint8_t *){member} + {i * elem_size}"
                size = (j - i) * elem_size
                if to_mem:
                    lines.append(f"memcpy({mem} + {loc}, {ptr}, {size});")
                else:
                    lines.append(f"memcpy({ptr}, {mem} + {loc}, {size});")
                i = j
    return lines


def _byte_array(data: bytes) -> str:
    rows = (', '.join(f"0x{b:02x}" for b in data[i:i + 12]) for i in range(0, len(data), 12))
    return '{\n' + ''.join(f"    {row},\n" for row in rows) + '}'


def export_static(name: str, inputs: dict[str, Any], outputs: dict[str, Any], out_dir: str,
                  arch: str, code_addr: int, data_addr: int, optimization: str = 'O3') -> StaticProgram:
    """Compiles the program that computes the outputs from the inputs for fixed
    addresses on an embedded target and writes it as C source: <name>.c and <name>.h.

    The C source contains the patched machine code as read only array (it can be
    executed from flash), the data memory and the functions <name>_init and
    <name>_step. The inputs and outputs are passed to <name>_step as structs
    with one member per input or output.

    The firmware must place the section .copapy_code at code_addr and the
    section .copapy_data at data_addr (linker script), <name>_init checks this.

    Arguments:
        name: Name of the program, prefix of the files and of all C identifiers
        inputs: Inputs by name: cp.value, cp.array of numbers or vectors and
            tensors of cp.value. Their values are the initial values.
        outputs: Outputs by name: values, vectors, tensors or arrays
        out_dir: Directory for the generated files
        arch: Target architecture (e.g. 'armv7mthumb')
        code_addr: Address of the code on the target
        data_addr: Address of the data memory on the target
        optimization: Optimization level of the stencils

    Returns:
        Paths of the generated files and the code and data sizes
    """
    for n in (name, *inputs, *outputs):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', n):
            raise ValueError(f"'{n}' is no valid C identifier")
    if not outputs:
        raise ValueError("At least one output is required")
    for addr in (code_addr, data_addr):
        if not 0 <= addr < 2**32 or addr % MEM_ALIGNMENT:
            raise ValueError(f"Address {addr:#x} must be a 32 bit address aligned to {MEM_ALIGNMENT} bytes")

    sdb = stencil_db_from_package(arch, optimization)
    assert sdb.byteorder == sys.byteorder, "Byte order of the target must match the host"

    dw, variables = compile_to_dag(root_nodes(list(outputs.values())), sdb)

    # Last command of the program is ENTRY_POINT with the address relative to the code
    assert dw._data[-2][0] == binw.Command.ENTRY_POINT.name
    entry = int.from_bytes(dw._data[-1][1], sdb.byteorder)

    # Let the runner patch the code for the addresses on the target and read it back
    com = binw.data_writer(sdb.byteorder)
    com.write_com(binw.Command.LINK_ADDR)
    com.write_int(code_addr)
    com.write_int(data_addr)
    dw.write_com(binw.Command.END_COM)
    context = runner.create_target()
    try:
        if runner.coparun(context, com.get_data() + dw.get_data()) <= 0:
            raise RuntimeError("The runner failed to patch the program")
        code = runner.code_mem(context)
        try:
            data = bytes(memoryview(runner.data_mem(context)))
        except ValueError:
            data = b''  # Program without data
    finally:
        runner.clear_target(context)

    in_fields = [_field(n, v, variables, sdb, True) for n, v in inputs.items()]
    out_fields = [_field(n, v, variables, sdb, False) for n, v in outputs.items()]

    upper = name.upper()
    in_arg = f"const {name}_in_t *in, " if in_fields else ''
    step_decl = f"void {name}_step({in_arg}{name}_out_t *out)"

    header = f"""\
/* Generated by copapy for {arch} - do not edit
 *
 * The code is linked for fixed addresses, the firmware must place the section
 * .copapy_code at {upper}_CODE_ADDR and the section .copapy_data at {upper}_DATA_ADDR.
 */
#ifndef {upper}_H
#define {upper}_H

#include <stdint.h>

#define {upper}_CODE_ADDR {code_addr:#010x}u
#define {upper}_CODE_SIZE {len(code)}
#define {upper}_DATA_ADDR {data_addr:#010x}u
#define {upper}_DATA_SIZE {len(data)}

{_struct(name + '_in_t', in_fields) if in_fields else ''}
{_struct(name + '_out_t', out_fields)}
/* Initializes the data memory, to be called once before {name}_step.
   Returns 0 on success and -1 if code or data are not located at the
   addresses the program was generated for. */
int {name}_init(void);

/* Runs the program. Not reentrant: all calls share one data memory. */
{step_decl};

#endif
"""

    attr = '__attribute__((section("{}"), aligned({}), used))'
    addr_checks = [f"(uintptr_t){name}_code != {upper}_CODE_ADDR"]
    source = f"""\
/* Generated by copapy for {arch} - do not edit */
#include <string.h>
#include "{name}.h"

/* Machine code patched for {upper}_CODE_ADDR and {upper}_DATA_ADDR */
{attr.format('.copapy_code', MEM_ALIGNMENT)}
static const uint8_t {name}_code[{len(code)}] = {_byte_array(code)};
"""
    init_body = ''
    if data:
        addr_checks.append(f"(uintptr_t){name}_data != {upper}_DATA_ADDR")
        init_body = f"    memcpy({name}_data, {name}_data_init, sizeof({name}_data));\n"
        source += f"""
/* Data memory: constants and variables of the program */
{attr.format('.copapy_data', MEM_ALIGNMENT)}
static uint8_t {name}_data[{len(data)}];

static const uint8_t {name}_data_init[{len(data)}] = {_byte_array(data)};
"""
    step_lines = _copies(in_fields, sdb, 'in', f"{name}_data", True)
    step_lines.append(f"((int (*)(void))((uintptr_t){name}_code + {entry}))();")
    step_lines += _copies(out_fields, sdb, 'out', f"{name}_data", False)
    source += f"""
int {name}_init(void) {{
    if ({' || '.join(addr_checks)}) return -1;
{init_body}    return 0;
}}

{step_decl} {{
{''.join(f'    {line}{chr(10)}' for line in step_lines)}}}
"""

    os.makedirs(out_dir, exist_ok=True)
    c_file = os.path.join(out_dir, name + '.c')
    h_file = os.path.join(out_dir, name + '.h')
    for path, text in ((c_file, source), (h_file, header)):
        with open(path, 'w', newline='\n') as f:
            f.write(text)

    return StaticProgram(c_file, h_file, len(code), len(data))
