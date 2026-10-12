import importlib.util
import os
import re
from typing import Any

import pytest

import copapy as cp
import coparun_module
from copapy import _binwrite as binw
from copapy.backend import Store, compile_to_dag, stencil_db_from_package

pytestmark = pytest.mark.skipif(not hasattr(coparun_module, 'code_mem'),
                                reason="coparun_module without code_mem, rebuild it")

CODE_ADDR = 0x08010000
DATA_ADDR = 0x20001000


def program() -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {'a': cp.value(1.5),
              'n': cp.value(3),
              'unused': cp.value(0.0),
              'v': cp.vector(cp.value(float(i)) for i in range(3)),
              'arr': cp.array([[1.0, 2.0], [3.0, 4.0]])}
    a, n, v, arr = inputs['a'], inputs['n'], inputs['v'], inputs['arr']
    outputs = {'y': cp.sin(a) * n + v[0],
               'flag': a > 1.0,
               'scaled': v * a,
               'arr2': arr * 2.0,
               'fixed': 4.5}
    return inputs, outputs


def test_generated_interface(tmp_path: Any) -> None:
    inputs, outputs = program()
    prog = cp.export_static('prog', inputs, outputs, str(tmp_path), 'armv7mthumb', CODE_ADDR, DATA_ADDR)

    with open(prog.h_file) as f:
        header = f.read()
    with open(prog.c_file) as f:
        source = f.read()

    assert prog.code_size > 0 and prog.data_size > 0
    assert f"#define PROG_CODE_ADDR {CODE_ADDR:#010x}u" in header
    assert f"#define PROG_DATA_SIZE {prog.data_size}" in header
    for member in ('float a;', 'int32_t n;', 'float unused;', 'float v[3];', 'float arr[2][2];',
                   'float y;', 'int32_t flag;', 'float scaled[3];', 'float arr2[2][2];', 'float fixed;'):
        assert member in header, member
    assert 'void prog_step(const prog_in_t *in, prog_out_t *out);' in header

    assert '/* unused is not used by the program */' in source
    assert 'out->fixed = 4.5f;' in source
    # Arrays are copied at once
    assert re.search(r'memcpy\(prog_data \+ \d+, in->arr, 16\);', source)
    assert re.search(r'memcpy\(out->arr2, prog_data \+ \d+, 16\);', source)
    # All copies are inside the data memory
    for offs, size in re.findall(r'prog_data \+ (\d+), [^,;]+, (\d+)\);', source):
        assert int(offs) + int(size) <= prog.data_size
    assert len(re.findall(r'0x[0-9a-f]{2},', source)) == prog.code_size + prog.data_size


def test_invalid_arguments(tmp_path: Any) -> None:
    inputs, outputs = program()
    with pytest.raises(ValueError):
        cp.export_static('prog', inputs, outputs, str(tmp_path), 'armv7mthumb', CODE_ADDR + 2, DATA_ADDR)
    with pytest.raises(ValueError):
        cp.export_static('1prog', inputs, outputs, str(tmp_path), 'armv7mthumb', CODE_ADDR, DATA_ADDR)
    with pytest.raises(TypeError):
        # A computed value is no input
        cp.export_static('prog', {'x': inputs['a'] + 1}, outputs, str(tmp_path), 'armv7mthumb', CODE_ADDR, DATA_ADDR)


def load_extract_code() -> Any:
    path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'extract_code.py')
    spec = importlib.util.spec_from_file_location('extract_code', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arch', ['armv7mthumb', 'armv7', 'arm64', 'riscv32', 'x86_64', 'x86'])
def test_link_addr_patching(arch: str) -> None:
    """Code patched by the runner for link addresses must equal the code
    patched by the reference implementation in tools/extract_code.py"""
    extract_code = load_extract_code()
    sdb = stencil_db_from_package(arch)
    _, outputs = program()
    dw, _ = compile_to_dag([Store(outputs['y'].net), Store(outputs['flag'].net), outputs['arr2'].net.source], sdb)

    # Reference: code at address 0 (only the offset between code and data is used
    # by extract_code), for absolute addresses the code is located at 0 as well
    reference = bytearray()
    dr = binw.data_reader(dw.get_data(), sdb.byteorder)
    n_patches = 0
    while True:
        com = dr.read_com()
        if com == binw.Command.ALLOCATE_CODE:
            reference = bytearray(dr.read_int())
        elif com == binw.Command.ALLOCATE_DATA:
            dr.read_int()
        elif com in (binw.Command.COPY_CODE, binw.Command.COPY_DATA):
            offs, size = dr.read_int(), dr.read_int()
            chunk = dr.read_bytes(size)
            if com == binw.Command.COPY_CODE:
                reference[offs:offs + size] = chunk
        elif com == binw.Command.PATCH:
            offs, val, mask = dr.read_int(), dr.read_int(signed=True), dr.read_int()
            encoding, shift, flags = binw.PatchEncoding(dr.read_byte()), dr.read_byte(), binw.PatchFlag(dr.read_byte())
            dr.read_byte()
            extract_code.apply_patch(reference, offs, val, mask, encoding, shift, flags, DATA_ADDR, sdb.byteorder)
            n_patches += 1
        elif com == binw.Command.ENTRY_POINT:
            dr.read_int()
            break
        else:
            assert com == binw.Command.FREE_MEMORY, com
    assert n_patches

    com_writer = binw.data_writer(sdb.byteorder)
    com_writer.write_com(binw.Command.LINK_ADDR)
    com_writer.write_int(0)
    com_writer.write_int(DATA_ADDR)
    dw.write_com(binw.Command.END_COM)
    context = coparun_module.create_target()
    try:
        assert coparun_module.coparun(context, com_writer.get_data() + dw.get_data()) > 0
        assert coparun_module.code_mem(context) == bytes(reference)

        # Code patched for link addresses must not be run
        run = binw.data_writer(sdb.byteorder)
        run.write_com(binw.Command.RUN_PROG)
        run.write_com(binw.Command.END_COM)
        assert coparun_module.coparun(context, run.get_data()) < 0
    finally:
        coparun_module.clear_target(context)
