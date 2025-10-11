from dataclasses import dataclass
from pelfy import open_elf_file, elf_file, elf_symbol
from typing import Generator, Literal
from enum import Enum

ByteOrder = Literal['little', 'big']

# on x86_64: call or jmp instruction when tail call optimized
LENGTH_CALL_INSTRUCTION = 5

RelocationType = Enum('RelocationType', [('RELOC_RELATIVE_32', 0)])


@dataclass
class patch_entry:
    """
    A dataclass for representing a relocation entry

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


def get_return_function_type(symbol: elf_symbol) -> str:
    if symbol.relocations:
        result_func = symbol.relocations[-1].symbol
        if result_func.name.startswith('result_'):
            return result_func.name[7:]
    return 'void'


def strip_function(func: elf_symbol) -> bytes:
    """Return stencil code by striped stancil function"""
    start_index, end_index = get_stencil_position(func)
    return func.data[start_index:end_index]


def get_stencil_position(func: elf_symbol) -> tuple[int, int]:
    start_index = 0  # For a "naked" function
    end_index = get_last_call_in_function(func)
    return start_index, end_index


def get_last_call_in_function(func: elf_symbol) -> int:
    # Find last relocation in function
    reloc = func.relocations[-1]
    assert reloc, f'No call function in stencil function {func.name}.'
    return reloc.fields['r_offset'] - func.fields['st_value'] - reloc.fields['r_addend'] - LENGTH_CALL_INSTRUCTION


class stencil_database():
    """A class for loading and querying a stencil database from an ELF object file

    Attributes:
        function_definitions (dict[str, str]): dictionary of function names and their return types
        data (dict[str, bytes]): dictionary of function names and their stripped code
        var_size (dict[str, int]): dictionary of object names and their sizes
        byteorder (ByteOrder): byte order of the ELF file
        elf (elf_file): the loaded ELF file
    """

    def __init__(self, obj_file: str | bytes):
        """Load the stencil database from an ELF object file

        Args:
            obj_file: path to the ELF object file or bytes of the ELF object file
        """
        if isinstance(obj_file, str):
            self.elf = open_elf_file(obj_file)
        else:
            self.elf = elf_file(obj_file)

        self.function_definitions = {s.name: get_return_function_type(s)
                                     for s in self.elf.symbols
                                     if s.info == 'STT_FUNC'}
        self.data = {s.name: strip_function(s)
                     for s in self.elf.symbols
                     if s.info == 'STT_FUNC'}
        self.var_size = {s.name: s.fields['st_size']
                         for s in self.elf.symbols
                         if s.info == 'STT_OBJECT'}
        self.byteorder: ByteOrder = self.elf.byteorder

        for name in self.function_definitions.keys():
            sym = self.elf.symbols[name]
            sym.relocations
            self.elf.symbols[name].data

    def get_patch_positions(self, symbol_name: str) -> Generator[patch_entry, None, None]:
        """Return patch positions for a provided symbol (function or object)

        Args:
            symbol_name: function or object name

        Yields:
            patch_entry: every relocation for the symbol
        """
        symbol = self.elf.symbols[symbol_name]
        start_index, end_index = get_stencil_position(symbol)

        for reloc in symbol.relocations:

            # address to fist byte to patch relative to the start of the symbol
            patch = translate_relocation(
                reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index,
                reloc.type,
                reloc.bits,
                reloc.fields['r_addend'])

            # Exclude the call to the result_* function
            if patch.addr < end_index - start_index:
                yield patch

    def get_stencil_code(self, name: str) -> bytes:
        """Return the striped function code for a provided function name
        
        Args:
            name: function name

        Returns:
            Striped function code
        """
        return strip_function(self.elf.symbols[name])

    def get_function_body(self, name: str, part: Literal['start', 'end']) -> bytes:
        func = self.elf.symbols[name]
        index = get_last_call_in_function(func)
        if part == 'start':
            return func.data[:index]
        else:
            return func.data[index + LENGTH_CALL_INSTRUCTION:]