import time
from copapy import backend
from copapy.backend import Write, stencil_db_from_package
import copapy.backend as cpb
import copapy as cp
import copapy._binwrite as binw
from copapy._compiler import get_nets, get_section_layout, get_data_layout
from copapy._compiler import patch_entry, CPConstant, get_aux_func_layout


def test_timing_compiler():
    t1 = cp.vector([10, 11]*128) + cp.vector(cp.value(v) for v in range(256))
    #t2 = t1.sum()
    t3 = cp.vector(cp.value(1 / (v + 1)) for v in range(256))
    t5 = ((t3 * t1) * 2).magnitude()
    out = [Write(t5)]

    print(out)

    print('-- get_edges:')
    t0 = time.time()
    edges = list(cpb.get_all_dag_edges(out))
    t1 = time.time()
    print(f'   found {len(edges)} edges')
    #for p in edges:
    #    print('#', p)
    print(f"get_edges time: {t1-t0:.6f}s")

    print('-- get_ordered_ops:')
    t0 = time.time()
    ordered_ops = cpb.stable_toposort(edges)
    t1 = time.time()
    print(f'   found {len(ordered_ops)} ops')
    #for p in ordered_ops:
    #    print('#', p)
    print(f"get_ordered_ops time: {t1-t0:.6f}s")

    print('-- get_consts:')
    t0 = time.time()
    const_net_list = cpb.get_const_nets(ordered_ops)
    t1 = time.time()
    #for p in const_list:
    #    print('#', p)
    print(f"get_consts time: {t1-t0:.6f}s")

    print('-- add_read_ops:')
    t0 = time.time()
    output_ops = list(cpb.add_read_ops(ordered_ops))
    t1 = time.time()
    #for p in output_ops:
    #    print('#', p)
    print(f"add_read_ops time: {t1-t0:.6f}s")

    print('-- add_write_ops:')
    t0 = time.time()
    extended_output_ops = list(cpb.add_write_ops(output_ops, const_net_list))
    t1 = time.time()
    #for p in extended_output_ops:
    #    print('#', p)
    print(f"add_write_ops time: {t1-t0:.6f}s")
    print('--')

    print('-- load_stencil_db:')
    t0 = time.time()
    sdb = stencil_db_from_package()
    dw = binw.data_writer(sdb.byteorder)
    t1 = time.time()
    print(f"load_stencil_db time: {t1-t0:.6f}s")



    # Get all nets/variables associated with heap memory
    variable_list = get_nets([[const_net_list]], extended_output_ops)
    stencil_names = {node.name for _, node in extended_output_ops}

    print(f'-- get_sub_functions:  {len(stencil_names)}')
    t0 = time.time()
    aux_function_names = sdb.get_sub_functions(stencil_names)
    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")

    print('-- const_sections_from_functions:')
    t0 = time.time()
    used_sections = sdb.const_sections_from_functions(aux_function_names | stencil_names)
    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")

    print('-- get_section_layout:')
    t0 = time.time()
    section_mem_layout, sections_length = get_section_layout(used_sections, sdb)
    variable_mem_layout, _ = get_data_layout(variable_list, sdb, sections_length)
    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")


    variables: dict[backend.Net, tuple[int, int, str]] = {}
    data_list: list[bytes] = []
    patch_list: list[patch_entry] = []


    print('-- write_data:')
    t0 = time.time()
    # Heap constants
    for section_id, start, lengths in section_mem_layout:
        dw.write_com(binw.Command.COPY_DATA)
        dw.write_int(start)
        dw.write_int(lengths)
        dw.write_bytes(sdb.get_section_data(section_id))

    # Heap variables
    for net, start, lengths in variable_mem_layout:
        variables[net] = (start, lengths, net.dtype)
        if isinstance(net.source, CPConstant):
            dw.write_com(binw.Command.COPY_DATA)
            dw.write_int(start)
            dw.write_int(lengths)
            dw.write_value(net.source.value, lengths)
            #print(f'+ {net.dtype} {net.source.value}')

    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")



    # prep auxiliary_functions
    _, aux_func_addr_lookup, aux_function_lengths = get_aux_func_layout(aux_function_names, sdb)

    # Prepare program code and relocations
    object_addr_lookup = {net: offs for net, offs, _ in variable_mem_layout}
    section_addr_lookup = {id: offs for id, offs, _ in section_mem_layout}

    # assemble stencils to main program and patch stencils
    data = sdb.get_function_code('entry_function_shell', 'start')
    data_list.append(data)
    offset = aux_function_lengths + len(data)


    print('-- relocate stencils:')
    t0 = time.time()
    for associated_net, node in extended_output_ops:
        assert node.name in sdb.stencil_definitions, f"- Warning: {node.name} stencil not found"
        data = sdb.get_stencil_code(node.name)
        data_list.append(data)
        #print(f"* {node.name} ({offset}) " + ' '.join(f'{d:02X}' for d in data))

        for reloc in sdb.get_relocations(node.name, stencil=True):
            if reloc.target_symbol_info in ('STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'):
                #print('-- ' + reloc.target_symbol_name + ' // ' + node.name)
                if reloc.target_symbol_name.startswith('dummy_'):
                    # Patch for write and read addresses to/from heap variables
                    assert associated_net, f"Relocation found but no net defined for operation {node.name}"
                    #print(f"Patch for write and read addresses to/from heap variables: {node.name} {patch.target_symbol_info} {patch.target_symbol_name}")
                    obj_addr = object_addr_lookup[associated_net]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                elif reloc.target_symbol_name.startswith('result_'):
                    # Set return jump address to address of following stencil
                    patch = sdb.get_patch(reloc, offset + len(data), offset, binw.Command.PATCH_FUNC.value)
                else:
                    # Patch constants addresses on heap
                    assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {node.name} missing: {reloc.pelfy_reloc.symbol.name}"
                    obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                    patch = sdb.get_patch(reloc, obj_addr, offset, binw.Command.PATCH_OBJECT.value)
                    #print('* constants stancils', patch.type, patch.patch_address, binw.Command.PATCH_OBJECT, node.name)

            elif reloc.target_symbol_info == 'STT_FUNC':
                func_addr = aux_func_addr_lookup[reloc.target_symbol_name]
                patch = sdb.get_patch(reloc, func_addr, offset, binw.Command.PATCH_FUNC.value)
                #print(patch.type, patch.addr, binw.Command.PATCH_FUNC, node.name, '->', patch.target_symbol_name)
            else:
                raise ValueError(f"Unsupported: {node.name} {reloc.target_symbol_info} {reloc.target_symbol_name}")

            patch_list.append(patch)

        offset += len(data)
    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")

    print('-- relocate aux functions:')
    t0 = time.time()
    # Patch aux functions
    for name, start in aux_func_addr_lookup.items():
        for reloc in sdb.get_relocations(name):

            #assert reloc.target_symbol_info != 'STT_FUNC', "Not tested yet!"

            if reloc.target_symbol_info in {'STT_OBJECT', 'STT_NOTYPE', 'STT_SECTION'}:
                # Patch constants/variable addresses on heap
                #print('--> DATA ', name, reloc.pelfy_reloc.symbol.name, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                assert reloc.target_section_index in section_addr_lookup, f"- Function or object in {name} missing: {reloc.pelfy_reloc.symbol.name}"
                obj_addr = reloc.target_symbol_offset + section_addr_lookup[reloc.target_section_index]
                patch = sdb.get_patch(reloc, obj_addr, start, binw.Command.PATCH_OBJECT.value)

            elif reloc.target_symbol_info == 'STT_FUNC':
                #print('--> FUNC', name, reloc.pelfy_reloc.symbol.name, reloc.pelfy_reloc.symbol.info, reloc.pelfy_reloc.symbol.section.name)
                func_addr = aux_func_addr_lookup[reloc.target_symbol_name]
                patch = sdb.get_patch(reloc, func_addr, start, binw.Command.PATCH_FUNC.value)
                #print(f'    FUNC {func_addr=}     {start=}    {patch.address=}')

            else:
                raise ValueError(f"Unsupported: {name=} {reloc.target_symbol_info=} {reloc.target_symbol_name=} {reloc.target_section_index}")

            patch_list.append(patch)
    t1 = time.time()
    print(f"time: {t1-t0:.6f}s")


if __name__ == "__main__":
    test_timing_compiler()
