from dataclasses import dataclass
from pelfy import open_elf_file, elf_file, elf_symbol
from typing import Generator, Literal, Iterable
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
    patch_address: int
    addend: int
    target_symbol_name: str
    target_symbol_info: str
    target_symbol_section_index: int
    target_symbol_address: int


def translate_relocation(relocation_addr: int, reloc_type: str, bits: int, r_addend: int) -> RelocationType:
    if reloc_type in ('R_AMD64_PLT32', 'R_AMD64_PC32'):
        # S + A - P
        return RelocationType.RELOC_RELATIVE_32
    else:
        raise Exception(f"Unknown relocation type: {reloc_type}")


def get_return_function_type(symbol: elf_symbol) -> str:
    if symbol.relocations:
        result_func = symbol.relocations[-1].symbol
        if result_func.name.startswith('result_'):
            return result_func.name[7:]
    return 'void'


def strip_function(func: elf_symbol) -> bytes:
    """Return stencil code by striped stancil function"""
    assert func.relocations and func.relocations[-1].symbol.info == 'STT_NOTYPE', f"{func.name} is not a stancil function"
    start_index, end_index = get_stencil_position(func)
    return func.data[start_index:end_index]


def get_stencil_position(func: elf_symbol) -> tuple[int, int]:
    start_index = 0  # For a "naked" function
    end_index = get_last_call_in_function(func)
    return start_index, end_index


def get_last_call_in_function(func: elf_symbol) -> int:
    # Find last relocation in function
    assert func.relocations, f'No call function in stencil function {func.name}.'
    reloc = func.relocations[-1]
    return reloc.fields['r_offset'] - func.fields['st_value'] - reloc.fields['r_addend'] - LENGTH_CALL_INSTRUCTION


def symbol_is_stencil(sym: elf_symbol) -> bool:
    return (sym.info == 'STT_FUNC' and len(sym.relocations) > 0 and
            sym.relocations[-1].symbol.info == 'STT_NOTYPE')


class stencil_database():
    """A class for loading and querying a stencil database from an ELF object file

    Attributes:
        stencil_definitions (dict[str, str]): dictionary of function names and their return types
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

        self.stencil_definitions = {s.name: get_return_function_type(s)
                                    for s in self.elf.symbols
                                    if s.info == 'STT_FUNC'}

        #self.data = {s.name: strip_function(s)
        #             for s in self.elf.symbols
        #             if s.info == 'STT_FUNC'}

        #self.var_size = {s.name: s.fields['st_size']
        #                 for s in self.elf.symbols
        #                 if s.info == 'STT_OBJECT'}
        self.byteorder: ByteOrder = self.elf.byteorder

        #for name in self.function_definitions.keys():
        #    sym = self.elf.symbols[name]
        #    sym.relocations
        #    self.elf.symbols[name].data

    def const_sections_from_functions(self, symbol_names: Iterable[str]) -> list[int]:
        ret: set[int] = set()

        for name in symbol_names:
            for reloc in self.elf.symbols[name].relocations:
                sym = reloc.symbol
                if sym.section and sym.section.type == 'SHT_PROGBITS' and \
                   sym.info != 'STT_FUNC' and not sym.name.startswith('dummy_'):
                    ret.add(sym.section.index)
        return list(ret)

    def get_patch_positions(self, symbol_name: str, stencil: bool = False) -> Generator[patch_entry, None, None]:
        """Return patch positions for a provided symbol (function or object)

        Args:
            symbol_name: function or object name

        Yields:
            patch_entry: every relocation for the symbol
        """
        symbol = self.elf.symbols[symbol_name]
        if stencil:
            start_index, end_index = get_stencil_position(symbol)
        else:
            start_index = 0
            end_index = symbol.fields['st_size']

        for reloc in symbol.relocations:

            patch_offset = reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index

            # address to fist byte to patch relative to the start of the symbol
            rtype = translate_relocation(
                patch_offset,
                reloc.type,
                reloc.bits,
                reloc.fields['r_addend'])

            patch = patch_entry(rtype, patch_offset,
                                reloc.fields['r_addend'],
                                reloc.symbol.name,
                                reloc.symbol.info,
                                reloc.symbol.fields['st_shndx'],
                                reloc.symbol.fields['st_value'])

            # Exclude the call to the result_* function
            if patch.patch_address < end_index - start_index:
                yield patch

    def get_stencil_code(self, name: str) -> bytes:
        """Return the striped function code for a provided function name

        Args:
            name: function name

        Returns:
            Striped function code
        """
        return strip_function(self.elf.symbols[name])

    def get_sub_functions(self, names: Iterable[str]) -> set[str]:
        name_set: set[str] = set()
        for name in names:
            if name not in name_set:
                # assert name in self.elf.symbols, f"Stencil {name} not found" <-- see: https://github.com/Nonannet/pelfy/issues/1
                func = self.elf.symbols[name]
                for r in func.relocations:
                    if r.symbol.info == 'STT_FUNC':
                        name_set.add(r.symbol.name)
                        name_set |= self.get_sub_functions([r.symbol.name])
        return name_set

    def get_symbol_size(self, name: str) -> int:
        return self.elf.symbols[name].fields['st_size']

    def get_section_size(self, id: int) -> int:
        return self.elf.sections[id].fields['sh_size']

    def get_section_data(self, id: int) -> bytes:
        return self.elf.sections[id].data

    def get_function_code(self, name: str, part: Literal['full', 'start', 'end'] = 'full') -> bytes:
        """Returns machine code for a specified function name."""
        func = self.elf.symbols[name]
        assert func.info == 'STT_FUNC', f"{name} is not a function"

        if part == 'start':
            index = get_last_call_in_function(func)
            return func.data[:index]
        elif part == 'end':
            index = get_last_call_in_function(func)
            return func.data[index + LENGTH_CALL_INSTRUCTION:]
        else:
            return func.data
