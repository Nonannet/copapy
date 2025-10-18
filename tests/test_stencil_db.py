from copapy import stencil_database, stencil_db
import platform

arch = platform.machine()
sdb = stencil_database(f'src/copapy/obj/stencils_{arch}_O3.o')


def test_list_symbols():
    print('----')
    #print(sdb.function_definitions)
    for sym_name in sdb.stencil_definitions.keys():
        print('\n-', sym_name)
        #print(list(sdb.get_patch_positions(sym_name)))


def test_start_end_function():
    for sym_name in sdb.stencil_definitions.keys():
        symbol = sdb.elf.symbols[sym_name]

        if symbol.relocations and symbol.relocations[-1].symbol.info == 'STT_NOTYPE':

            print('-', sym_name, stencil_db.get_stencil_position(symbol), len(symbol.data))

            start, end = stencil_db.get_stencil_position(symbol)

            assert start >= 0 and end >= start and end <= len(symbol.data)


def test_aux_functions():
    for sym_name in sdb.stencil_definitions.keys():
        symbol = sdb.elf.symbols[sym_name]
        for reloc in symbol.relocations:
            if reloc.symbol.info != "STT_NOTYPE":
                print(reloc.symbol.name, reloc.symbol.info)


if __name__ == "__main__":
    test_aux_functions()
