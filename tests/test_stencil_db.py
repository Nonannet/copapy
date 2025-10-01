from copapy import stencil_database
from copapy import stencil_db


def test_list_symbols():
    sdb = stencil_database('src/copapy/obj/stencils_x86_64_O3.o')
    print('----')
    #print(sdb.function_definitions)
    for sym_name in sdb.function_definitions.keys():
        print('\n-', sym_name)
        print(list(sdb.get_patch_positions(sym_name)))


def test_start_end_function():
    sdb = stencil_database('src/copapy/obj/stencils_x86_64_O3.o')
    for sym_name in sdb.function_definitions.keys():
        data = sdb.elf.symbols[sym_name].data
        print('-', sym_name, stencil_db.get_stencil_position(data, sdb.elf.byteorder), len(data))

        start, end = stencil_db.get_stencil_position(data, sdb.elf.byteorder)

        assert start >= 0 and end >= start and end <= len(data)


if __name__ == "__main__":
    test_list_symbols()
    test_start_end_function()
