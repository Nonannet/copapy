from copapy._binwrite import data_reader, Command, ByteOrder
import argparse
from typing import Literal

def patch(data: bytearray, offset: int, patch_mask: int, value: int, byteorder: Literal['little', 'big']) -> None:
    # Read 4 bytes at the offset as a little-endian uint32
    original = int.from_bytes(data[offset:offset+4], byteorder)
    
    # Apply the patch
    new_value = (original & ~patch_mask) | (value & patch_mask)
    
    # Write the new value back to the bytearray
    data[offset:offset+4] = new_value.to_bytes(4, byteorder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path with copapy commands")
    parser.add_argument("output_file", type=str, help="Output file with patched code")
    parser.add_argument("--data_section_offset", type=int, default=0x3000, help="Offset for data relative to code section")
    parser.add_argument("--byteorder", type=str, choices=['little', 'big'], default='little', help="Select byteorder")
    args = parser.parse_args()

    input_file: str = args.input_file
    output_file: str = args.output_file
    data_section_offset: int = args.data_section_offset
    byteorder: ByteOrder = args.byteorder

    with open(input_file, mode='rb') as f_in:
        dr = data_reader(f_in.read(), byteorder)

    buffer_index: int = 0
    end_flag: int = 0
    program_data: bytearray = bytearray([])

    while (end_flag == 0):
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
            mask = dr.read_int()
            value = dr.read_int(signed=True)
            patch(program_data, offs, mask, value, byteorder)
            print(f"PATCH_FUNC patch_offs={offs} mask=0x{mask:x} value={value}")
        elif com == Command.PATCH_OBJECT:
            offs = dr.read_int()
            mask = dr.read_int()
            value = dr.read_int(signed=True)
            patch(program_data, offs, mask, value + data_section_offset, byteorder)
            print(f"PATCH_OBJECT patch_offs={offs} mask=ox{mask:x} value={value}")
        elif com == Command.ENTRY_POINT:
            rel_entr_point = dr.read_int()
            print(f"ENTRY_POINT rel_entr_point={rel_entr_point}")
        elif com == Command.RUN_PROG:
            print("RUN_PROG")
        elif com == Command.READ_DATA:
            offs = dr.read_int()
            size = dr.read_int()
            print(f"READ_DATA offs={offs} size={size}")
        elif com == Command.FREE_MEMORY:
            print("READ_DATA")
        elif com == Command.END_COM:
            print("END_COM")
            end_flag = 1
        else:
            assert False, f"Unknown command: {com}"

    with open(output_file, mode='wb') as f_out:
        f_out.write(program_data)

    print(f"Code written to {output_file}.")
