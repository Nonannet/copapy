from copapy.binwrite import data_reader, Command, ByteOrder
from copapy.stencil_db import RelocationType

input_file = "bin/test.copapy"

output_file = "bin/test_code.bin"

data_section_offset = 0x2000

byteorder: ByteOrder = 'little'

with open(input_file, mode='rb') as f:
    dr = data_reader(f.read(), byteorder)

buffer_index: int = 0
end_flag: int = 0
program_data: bytearray = bytearray([])

while(end_flag == 0):
    com = dr.read_com()

    if com == Command.ALLOCATE_DATA:
        size = dr.read_int()
        print(f"ALLOCATE_DATA size={size}")
    elif com == Command.ALLOCATE_CODE:
        size = dr.read_int()
        program_data = bytearray(size)
        print(f"ALLOCATE_CODE size={size}")
    elif com == Command.COPY_DATA:
        offs = dr.read_int()
        size = dr.read_int()
        datab = dr.read_bytes(size)
        print(f"COPY_DATA offs={offs} size={size} data={' '.join(hex(d) for d in datab)}")
    elif com == Command.COPY_CODE:
        offs = dr.read_int()
        size = dr.read_int()
        datab = dr.read_bytes(size)
        program_data[offs:offs + size] = datab
        print(f"COPY_CODE offs={offs} size={size} data={' '.join(hex(d) for d in datab[:5])}...")
    elif com == Command.PATCH_FUNC:
        offs = dr.read_int()
        reloc_type = dr.read_int()
        value = dr.read_int(signed=True)
        print(f"PATCH_FUNC patch_offs={offs} reloc_type={reloc_type} value={value}")
    elif com == Command.PATCH_OBJECT:
        offs = dr.read_int()
        reloc_type = dr.read_int()
        value = dr.read_int(signed=True)
        assert reloc_type == RelocationType.RELOC_RELATIVE_32.value
        program_data[offs:offs + 4] = (value + data_section_offset).to_bytes(4, byteorder, signed=True)
        print(f"PATCH_OBJECT patch_offs={offs} reloc_type={reloc_type} value={value}")
    elif com == Command.RUN_PROG:
        rel_entr_point = dr.read_int()
        print(f"RUN_PROG rel_entr_point={rel_entr_point}")
    elif com == Command.READ_DATA:
        offs = dr.read_int()
        size = dr.read_int()
        print(f"READ_DATA offs={offs} size={size}")
    elif com == Command.FREE_MEMORY:
        print("READ_DATA")
    elif com == Command.END_PROG:
        print("END_PROG")
        end_flag = 1
    else:
        assert False, f"Unknown command: {com}"

with open(output_file, mode='wb') as f:
    f.write(program_data)

print('OK')