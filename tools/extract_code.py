from copapy._binwrite import data_reader, Command, ByteOrder, PatchEncoding, PatchFlag
import argparse


def read_u32(data: bytearray, offset: int, byteorder: ByteOrder) -> int:
    return int.from_bytes(data[offset:offset+4], byteorder)


def write_u32(data: bytearray, offset: int, value: int, byteorder: ByteOrder) -> None:
    data[offset:offset+4] = (value & 0xFFFFFFFF).to_bytes(4, byteorder)


def patch_bitfield(data: bytearray, offset: int, patch_mask: int, value: int, byteorder: ByteOrder) -> None:
    original = read_u32(data, offset, byteorder)
    shift_factor = patch_mask & -patch_mask
    write_u32(data, offset, (original & ~patch_mask) | ((value * shift_factor) & patch_mask), byteorder)


def patch_hi21(data: bytearray, offset: int, page_offset: int, byteorder: ByteOrder) -> None:
    # ADRP immediate is signed 21 bits: valid range [-2^20, 2^20 - 1]
    if not (-(1 << 20) <= page_offset < (1 << 20)):
        raise ValueError(f"page_offset {page_offset} out of 21-bit range")

    instr = read_u32(data, offset, byteorder)

    # Split the page offset into immhi (bits[20:2]) and immlo (bits[1:0])
    immlo = page_offset & 0x3
    immhi = (page_offset >> 2) & 0x7FFFF

    # Clear and insert immhi (bits 23:5) and immlo (bits 30:29)
    instr &= ~((0x7FFFF << 5) | (0x3 << 29))
    instr |= (immhi << 5) | (immlo << 29)

    write_u32(data, offset, instr, byteorder)


def patch_arm_movw_movt(data: bytearray, offset: int, imm16: int, byteorder: ByteOrder) -> None:
    instr = read_u32(data, offset, byteorder)
    instr &= ~((0xF << 16) | 0xFFF)
    instr |= (((imm16 >> 12) & 0xF) << 16) | (imm16 & 0xFFF)
    write_u32(data, offset, instr, byteorder)


def patch_thumb_movw_movt(data: bytearray, offset: int, imm16: int, byteorder: ByteOrder) -> None:
    first = int.from_bytes(data[offset:offset+2], byteorder)
    second = int.from_bytes(data[offset+2:offset+4], byteorder)
    first &= ~(0x000F | (1 << 10))
    second &= ~(0x00FF | (0x7 << 12))
    first |= ((imm16 >> 12) & 0xF) | (((imm16 >> 11) & 0x1) << 10)
    second |= (imm16 & 0xFF) | (((imm16 >> 8) & 0x7) << 12)
    data[offset:offset+2] = (first & 0xFFFF).to_bytes(2, byteorder)
    data[offset+2:offset+4] = (second & 0xFFFF).to_bytes(2, byteorder)


def patch_thumb_branch(data: bytearray, offset: int, value: int, byteorder: ByteOrder) -> None:
    first = int.from_bytes(data[offset:offset+2], byteorder)
    second = int.from_bytes(data[offset+2:offset+4], byteorder)
    s = (value >> 23) & 0x1
    j1 = (~(((value >> 22) & 0x1) ^ s)) & 0x1
    j2 = (~(((value >> 21) & 0x1) ^ s)) & 0x1
    first = (first & 0xF800) | (s << 10) | ((value >> 11) & 0x3FF)
    second = (second & 0xD000) | (j1 << 13) | (j2 << 11) | (value & 0x7FF)
    data[offset:offset+2] = first.to_bytes(2, byteorder)
    data[offset+2:offset+4] = second.to_bytes(2, byteorder)


def apply_patch(data: bytearray, offs: int, value: int, mask: int, encoding: PatchEncoding,
                shift: int, flags: PatchFlag, data_section_offset: int, byteorder: ByteOrder) -> int:
    """Same calculation as apply_patch in runmem.c with the code memory located at address 0"""
    target = value + (data_section_offset if flags & PatchFlag.DATA else 0)  # S + A
    pc = offs  # P
    if flags & PatchFlag.PAGE:
        target &= ~0xFFF
        pc &= ~0xFFF
    result = (target - pc if flags & PatchFlag.PC_REL else target) >> shift

    if encoding == PatchEncoding.BITFIELD:
        patch_bitfield(data, offs, mask, result, byteorder)
    elif encoding == PatchEncoding.AARCH64_ADRP:
        patch_hi21(data, offs, result, byteorder)
    elif encoding == PatchEncoding.ARM_MOVW_MOVT:
        patch_arm_movw_movt(data, offs, result & 0xFFFF, byteorder)
    elif encoding == PatchEncoding.THUMB_MOVW_MOVT:
        patch_thumb_movw_movt(data, offs, result & 0xFFFF, byteorder)
    elif encoding == PatchEncoding.THUMB_BRANCH:
        patch_thumb_branch(data, offs, result, byteorder)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file path with copapy commands")
    parser.add_argument("output_file", type=str, help="Output file with patched code")
    parser.add_argument("--data_section_offset", type=int, default=-0x1000, help="Offset for data relative to code section")
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
            print(f"COPY_DATA offs=0x{offs + data_section_offset:x} size={size} data={' '.join(hex(d) for d in datab)}")
        elif com == Command.COPY_CODE:
            offs = dr.read_int()
            size = dr.read_int()
            datab = dr.read_bytes(size)
            program_data[offs:offs + size] = datab
            print(f"COPY_CODE offs=0x{offs:x} size={size} data={' '.join(hex(d) for d in datab[:5])}...")
        elif com == Command.PATCH:
            offs = dr.read_int()
            value = dr.read_int(signed=True)
            mask = dr.read_int()
            encoding = PatchEncoding(dr.read_byte())
            shift = dr.read_byte()
            flags = PatchFlag(dr.read_byte())
            dr.read_byte()  # reserved
            result = apply_patch(program_data, offs, value, mask, encoding, shift, flags, data_section_offset, byteorder)
            print(f"PATCH patch_offs=0x{offs:x} value=0x{value:x} mask=0x{mask:x} encoding={encoding.name} shift={shift} flags={flags!r}")
            print(f" | calculated value: 0x{result & 0xFFFFFFFF:x}")
        elif com == Command.ENTRY_POINT:
            rel_entr_point = dr.read_int()
            print(f"ENTRY_POINT rel_entr_point=0x{rel_entr_point:x}")
        elif com == Command.RUN_PROG:
            print("RUN_PROG")
        elif com == Command.READ_DATA:
            offs = dr.read_int()
            size = dr.read_int()
            print(f"READ_DATA offs=0x{offs:x} size={size}")
        elif com == Command.FREE_MEMORY:
            print("READ_DATA")
        elif com == Command.DUMP_CODE:
            print("DUMP_CODE")
            end_flag = 2
        elif com == Command.END_COM:
            print("END_COM")
            end_flag = 1
        else:
            assert False, f"Unknown command: {com}"

    with open(output_file, mode='wb') as f_out:
        f_out.write(program_data)

    print(f"Code written to {output_file}.")
