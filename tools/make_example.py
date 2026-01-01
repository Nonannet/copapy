from copapy import value
from copapy.backend import Store, compile_to_dag, stencil_db_from_package
from copapy._binwrite import Command

input = value(9.0)

result = input ** 2 / 3.3 + 5

arch = 'native'
sdb = stencil_db_from_package(arch)
dw, _ = compile_to_dag([Store(result)], sdb)

# Instruct runner to dump patched code to a file:
dw.write_com(Command.DUMP_CODE)

dw.to_file('build/runner/test.copapy')