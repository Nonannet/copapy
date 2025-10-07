from dataclasses import dataclass
import pelfy
from typing import Generator, Literal
from enum import Enum

ByteOrder = Literal['little', 'big']

START_MARKER = 0xE1401F0F  # Nop on x86-64
END_MARKER = 0xE2401F0F  # Nop on x86-64
MARKER_LENGTH = 4

# on x86_64: call or jmp instruction when tail call optimized
LENGTH_CALL_INSTRUCTION = 5

RelocationType = Enum('RelocationType', [('RELOC_RELATIVE_32', 0)])


@dataclass
class patch_entry:
    """
    A class for representing a relocation entry

    Attributes:
        addr (int): address of first byte to patch relative to the start of the symbol
        type (RelocationType): relocation type"""

    type: RelocationType
    addr: int
    addend: int


def translate_relocation(relocation_addr: int, reloc_type: str, bits: int, r_addend: int) -> patch_entry:
    if reloc_type in ('R_AMD64_PLT32', 'R_AMD64_PC32'):
        # S + A - P
        patch_offset = relocation_addr
        return patch_entry(RelocationType.RELOC_RELATIVE_32, patch_offset, r_addend)
    else:
        print('relocation_addr: ', relocation_addr)
        print('reloc_type:      ', reloc_type)
        print('bits:            ', bits)
        print('r_addend:        ', r_addend)
        raise Exception(f"Unknown relocation type: {reloc_type}")


def get_ret_function_def(symbol: pelfy.elf_symbol) -> str:
    #print('*', symbol.name, len(symbol.relocations))
    if symbol.relocations:
        result_func = symbol.relocations[-1].symbol

        assert result_func.name.startswith('result_')
        return result_func.name[7:]
    else:
        return 'void'


def strip_symbol(data: bytes, byteorder: ByteOrder) -> bytes:
    """Return data between start and end marker and removing last instruction (call)"""
    start_index, end_index = get_stencil_position(data, byteorder)
    return data[start_index:end_index]


def get_stencil_position(data: bytes, byteorder: ByteOrder) -> tuple[int, int]:
    # Find first start marker
    marker_index = data.find(START_MARKER.to_bytes(MARKER_LENGTH, byteorder))
    start_index = 0 if marker_index < 0 else marker_index + MARKER_LENGTH

    # Find last end marker
    end_index = data.rfind(END_MARKER.to_bytes(MARKER_LENGTH, byteorder))
    end_index = len(data) if end_index < 0 else end_index - LENGTH_CALL_INSTRUCTION

    return start_index, end_index


class stencil_database():
    def __init__(self, obj_file: str | bytes):
        """Load the stencil database from an ELF object file
        """
        if isinstance(obj_file, str):
            self.elf = pelfy.open_elf_file(obj_file)
        else:
            self.elf = pelfy.elf_file(obj_file)

        #print(self.elf.symbols)

        self.function_definitions = {s.name: get_ret_function_def(s) for s in self.elf.symbols
                                     if s.info == 'STT_FUNC'}

        self.data = {s.name: strip_symbol(s.data, self.elf.byteorder) for s in self.elf.symbols
                     if s.info == 'STT_FUNC'}

        self.var_size = {s.name: s.fields['st_size'] for s in self.elf.symbols
                         if s.info == 'STT_OBJECT'}

        self.byteorder: Literal['little', 'big'] = self.elf.byteorder

        for name in self.function_definitions.keys():
            sym = self.elf.symbols[name]
            sym.relocations
            self.elf.symbols[name].data

    def get_patch_positions(self, symbol_name: str) -> Generator[patch_entry, None, None]:
        """Return patch positions
        """
        symbol = self.elf.symbols[symbol_name]
        start_index, end_index = get_stencil_position(symbol.data, symbol.file.byteorder)

        for reloc in symbol.relocations:

            # address to fist byte to patch relative to the start of the symbol
            patch = translate_relocation(
                reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index,
                reloc.type,
                reloc.bits,
                reloc.fields['r_addend'])

            # Exclude the call to the result_x function
            if patch.addr < end_index - start_index:
                yield patch

    def get_func_data(self, name: str) -> bytes:
        return strip_symbol(self.elf.symbols[name].data, self.elf.byteorder)
