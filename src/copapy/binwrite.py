from enum import Enum
from pelfy import elf_symbol

Command = Enum('Command', [('ALLOCATE_DATA', 1), ('COPY_DATA', 2),
                           ('ALLOCATE_CODE', 3), ('COPY_CODE', 4),
                           ('RELOCATE_FUNC', 5), ('RELOCATE_OBJECT', 6),
                           ('SET_ENTR_POINT', 64), ('END_PROG', 255)])

RelocationType = Enum('RelocationType', [('RELOC_RELATIVE_32', 0)])

def translate_relocation(new_sym_addr: int, new_patch_addr: int, reloc_type: str, r_addend: int) -> int:
    if reloc_type in ('R_AMD64_PLT32', 'R_AMD64_PC32'):
        # S + A - P
        value = new_sym_addr + r_addend - new_patch_addr
        return RelocationType.RELOC_RELATIVE_32, value
    else:
        raise Exception(f"Unknown: {reloc_type}")


def get_variable_data(symbols: list[elf_symbol]) -> tuple[list[tuple[elf_symbol, int, int]], int]:
    object_list = []
    out_offs = 0
    for sym in symbols:
        assert sym.info == 'STT_OBJECT'
        lengths = sym.fields['st_size']
        object_list.append((sym, out_offs, lengths))
        out_offs += (lengths + 3) // 4 * 4
    return object_list, out_offs


def get_function_data(symbols: list[elf_symbol]) -> tuple[list[tuple[elf_symbol, int, int, int]], int]:
    code_list = []
    out_offs = 0
    for sym in symbols:
        assert sym.info == 'STT_FUNC'
        lengths = sym.fields['st_size']

        #if strip_function:
        #    assert False, 'Not implemente'
        # TODO: Strip functions
        # Symbol, start out_offset in symbol, offset in output file, output lengths
        # Set in_sym_out_offs and lengths
        in_sym_offs = 0
        
        code_list.append((sym, in_sym_offs, out_offs, lengths))
        # out_offs += (lengths + 3) // 4 * 4
        out_offs += lengths  # should be aligned by default?
    return code_list, out_offs


def get_function_data_blob(symbols: list[elf_symbol]) -> tuple[list[tuple[elf_symbol, int, int, int]], int]:
    code_list = []
    out_offs = 0
    for sym in symbols:
        assert sym.info == 'STT_FUNC'
        lengths = sym.fields['st_size']

        #if strip_function:
        #    assert False, 'Not implemente'
        # TODO: Strip functions
        # Symbol, start out_offset in symbol, offset in output file, output lengths
        # Set in_sym_out_offs and lengths
        in_sym_offs = 0
        
        code_list.append((sym, in_sym_offs, out_offs, lengths))
        out_offs += (lengths + 3) // 4 * 4
    return code_list, out_offs


class data_writer():
    def __init__(self, byteorder: str):
        self._data: list[(str, bytes)] = list()
        self.byteorder = byteorder

    def write_int(self, value: int, num_bytes: int = 4, signed: bool = False):
        self._data.append((f"INT {value}", value.to_bytes(length=num_bytes, byteorder=self.byteorder, signed=signed), 0))

    def write_com(self, value: Enum, num_bytes: int = 4):
        self._data.append((value.name, value.value.to_bytes(length=num_bytes, byteorder=self.byteorder, signed=False), 1))
    
    def write_byte(self, value: int):
        self._data.append((f"BYTE {value}", bytes([value]), 0))

    def write_bytes(self, value: bytes):
        self._data.append((f"BYTES {len(value)}", value, 0))

    def print(self) -> str:
        for name, dat, flag in self._data:
            if flag:
                print('')
            print(f"{name:18}" + ' '.join(f'{b:02X}' for b in dat))

    def get_data(self) -> bytes:
        return b''.join(dat for _, dat, _ in self._data)

    def to_file(self, path: str):
        with open(path, 'wb') as f:
            f.write(self.get_data())