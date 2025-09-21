from copapy import stencil_database

if __name__ == "__main__":
    sdb = stencil_database('src/copapy/obj/stencils_x86_64_O3.o')
    print('----')
    #print(sdb.function_definitions)
    for sym_name in sdb.function_definitions.keys():
        print('\n-', sym_name)
        print(list(sdb.get_patch_positions(sym_name)))