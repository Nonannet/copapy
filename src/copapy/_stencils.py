from dataclasses import dataclass
from pelfy import open_elf_file, elf_file, elf_symbol
from typing import Generator, Literal, Iterable
import pelfy

ByteOrder = Literal['little', 'big']


@dataclass
class relocation_entry:
    """
    A dataclass for representing a relocation entry
    """

    target_symbol_name: str
    target_symbol_info: str
    target_symbol_offset: int
    target_section_index: int
    function_offset: int
    start: int
    pelfy_reloc: pelfy.elf_relocation


@dataclass
class patch_entry:
    """
    A dataclass for representing a relocation entry

    Attributes:
        addr (int): address of first byte to patch relative to the start of the symbol
        type (RelocationType): relocation type"""

    mask: int
    address: int
    value: int
    patch_type: int


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

    instruction_lenghs = 4 if reloc.bits < 32 else 5
    return reloc.fields['r_offset'] - func.fields['st_value'] - reloc.fields['r_addend'] - instruction_lenghs


def get_op_after_last_call_in_function(func: elf_symbol) -> int:
    # Find last relocation in function
    assert func.relocations, f'No call function in stencil function {func.name}.'
    reloc = func.relocations[-1]
    return reloc.fields['r_offset'] - func.fields['st_value'] - reloc.fields['r_addend']


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

    def get_relocations(self, symbol_name: str, stencil: bool = False) -> Generator[relocation_entry, None, None]:
        symbol = self.elf.symbols[symbol_name]
        if stencil:
            start_index, end_index = get_stencil_position(symbol)
        else:
            start_index = 0
            end_index = symbol.fields['st_size']

        print('->', symbol_name)
        for reloc in symbol.relocations:
            print('  ', symbol_name, reloc.symbol.info, reloc.symbol.name, reloc.type)

            # address to fist byte to patch relative to the start of the symbol
            patch_offset = reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index

            if patch_offset < end_index - start_index:  # Exclude the call to the result_* function
                yield relocation_entry(reloc.symbol.name,
                                       reloc.symbol.info,
                                       reloc.symbol.fields['st_value'],
                                       reloc.symbol.fields['st_shndx'],
                                       symbol.fields['st_value'],
                                       start_index,
                                       reloc)

    def get_patch(self, relocation: relocation_entry, symbol_address: int, function_offset: int, symbol_type: int) -> patch_entry:
        """Return patch positions for a provided symbol (function or object)

        Args:
            relocation: relocation entry
            symbol_address: absolute address of the target symbol
            function_offset: absolute address of the first byte of the
                function the patch is applied to

        Yields:
            patch_entry: every relocation for the symbol
        """
        pr = relocation.pelfy_reloc

        # calculate absolut address to the first byte to patch
        # relative to the start of the (stripped stencil) function:
        patch_offset = pr.fields['r_offset'] - relocation.function_offset - relocation.start + function_offset
        
        if pr.type.endswith('_PLT32') or pr.type.endswith('_PC32'):
            # S + A - P
            mask = 0xFFFFFFFF  # 32 bit
            patch_value = symbol_address + pr.fields['r_addend'] - patch_offset

            print(f'** {patch_offset=} {relocation.target_symbol_name=} {pr.fields['r_offset']=} {relocation.function_offset=} {relocation.start=} {function_offset=}')
            print(f'   {patch_value=} {symbol_address=} {pr.fields['r_addend']=}, {function_offset=}')

        #elif reloc.type.endswith('_JUMP26') or reloc.type.endswith('_CALL26'):
        #    # S + A - P
        #    assert reloc.file.byteorder == 'little', "Big endian not supported for ARM64"
        #    mask = 0x3ffffff  # 26 bit
        #    imm = offset >> 2
        #    assert imm < mask, "Relocation immediate value too large"

        else:
            raise NotImplementedError(f"Relocation type {pr.type} not implemented")

        return patch_entry(mask, patch_offset, patch_value, symbol_type)


    def get_stencil_code(self, name: str) -> bytes:
        """Return the striped function code for a provided function name

        Args:
            name: function name

        Returns:
            Striped function code
        """
        return strip_function(self.elf.symbols[name])

    def get_sub_functions(self, names: Iterable[str]) -> set[str]:
        """Return recursively all functions called by stencils or by other functions
        Args:
            names: function or stencil names
        
        Returns:
            set of all sub function names
        """
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

    def get_type_size(self, type_name: str) -> int:
        """Returns the size of a variable type in bytes."""
        return {'int': 4, 'float': 4}[type_name]

    def get_symbol_size(self, name: str) -> int:
        """Returns the size of a specified symbol name."""
        return self.elf.symbols[name].fields['st_size']

    def get_section_size(self, index: int) -> int:
        """Returns the size of a section specified by index."""
        return self.elf.sections[index].fields['sh_size']

    def get_section_data(self, index: int) -> bytes:
        """Returns the data of a section specified by index."""
        return self.elf.sections[index].data

    def get_function_code(self, name: str, part: Literal['full', 'start', 'end'] = 'full') -> bytes:
        """Returns machine code for a specified function name.
        
        Args:
            name: function name
            part: part of the function to return ('full', 'start', 'end')

        Returns:
            Machine code bytes of the specified part of the function
        """
        func = self.elf.symbols[name]
        assert func.info == 'STT_FUNC', f"{name} is not a function"

        if part == 'start':
            index = get_last_call_in_function(func)
            return func.data[:index]
        elif part == 'end':
            index = get_op_after_last_call_in_function(func)
            return func.data[index:]
        else:
            return func.data
