from dataclasses import dataclass
from pelfy import open_elf_file, elf_file, elf_symbol
from typing import Generator, Literal, Iterable
import pelfy
import struct
import platform

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
        type (RelocationType): relocation type
    """
    mask: int
    address: int
    value: int
    scale: int
    patch_type: int


def detect_process_arch() -> str:
    bits = struct.calcsize("P") * 8
    arch = platform.machine().lower()

    if arch in ('amd64', 'x86_64'):
        arch_family = 'x86_64' if bits == 64 else 'x86'
    elif arch in ('i386', 'i686', 'x86'):
        arch_family = 'x86'
    elif arch in ('arm64', 'aarch64'):
        arch_family = 'arm64'
    elif 'armv7' in arch or 'armv8' in arch:
        arch_family = 'armv7'  # Treat armv8 (64 bit CPU) as armv7 for 32 bit
    elif 'armv6' in arch:
        arch_family = 'armv6'
    elif 'mips' in arch:
        arch_family = 'mips64' if bits == 64 else 'mips'
    elif 'riscv' in arch:
        arch_family = 'riscv64' if bits == 64 else 'riscv'
    else:
        raise NotImplementedError(f"Platform {arch} with {bits} bits is not supported.")

    return arch_family


def get_return_function_type(symbol: elf_symbol) -> str:
    if symbol.relocations:
        for reloc in reversed(symbol.relocations):
            func_name = reloc.symbol.name
            if func_name.startswith('result_'):
                return func_name[7:]
    return 'void'


def get_stencil_position(func: elf_symbol) -> tuple[int, int]:
    start_index = 0  # There must be no prolog
    # Find last relocation in function
    last_instr = get_last_call_in_function(func)
    function_size = func.fields['st_size']
    if last_instr + 5 >= function_size:  # Check if jump is last instruction
        end_index = last_instr  # Jump can be striped
    else:
        end_index = function_size

    return start_index, end_index


def get_last_call_in_function(func: elf_symbol) -> int:
    # Find last relocation in function
    assert func.relocations, f'No call function in stencil function {func.name}.'
    reloc = func.relocations[-1]
    # Assume the call instruction is 4 bytes long for relocations with less than 32 bit and 5 bytes otherwise
    instruction_lengths = 4 if reloc.bits < 32 else 5
    address_field_length = 4
    #print(f"-> {[r.fields['r_offset'] - func.fields['st_value'] for r in func.relocations]}")
    return reloc.fields['r_offset'] - func.fields['st_value'] + address_field_length - instruction_lengths


def get_op_after_last_call_in_function(func: elf_symbol) -> int:
    # Find last relocation in function
    assert func.relocations, f'No call function in stencil function {func.name}.'
    reloc = func.relocations[-1]
    assert reloc.bits <= 32, "Relocation segment might be larger then 32 bit"
    return reloc.fields['r_offset'] - func.fields['st_value'] + 4


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

        self._relocation_cache: dict[tuple[str, bool], list[relocation_entry]] = {}
        self._stencil_cache: dict[str, tuple[int, int]] = {}

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
        cache_key = (symbol_name, stencil)
        if cache_key in self._relocation_cache:
            # cache hit:
            for reloc_entry in self._relocation_cache[cache_key]:
                yield reloc_entry
            return

        # cache miss:
        cache: list[relocation_entry] = []
        self._relocation_cache[cache_key] = cache

        symbol = self.elf.symbols[symbol_name]
        if stencil:
            start_index, end_index = get_stencil_position(symbol)
        else:
            start_index = 0
            end_index = symbol.fields['st_size']

        for reloc in symbol.relocations:

            # address to fist byte to patch relative to the start of the symbol
            patch_offset = reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index

            if patch_offset < end_index - start_index:  # Exclude the call to the result_* function
                reloc_entry = relocation_entry(reloc.symbol.name,
                                       reloc.symbol.info,
                                       reloc.symbol.fields['st_value'],
                                       reloc.symbol.fields['st_shndx'],
                                       symbol.fields['st_value'],
                                       start_index,
                                       reloc)
                cache.append(reloc_entry)
                yield reloc_entry

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
        #print(f"xx {pr.fields['r_offset'] - relocation.function_offset} {relocation.target_symbol_name=} {pr.fields['r_offset']=} {relocation.function_offset=} {relocation.start=} {function_offset=}")
        scale = 1
        mask = 0xFFFFFFFF  # 32 bit

        #print("------- reloc ", pr.type, pr.target_section.name, pr.symbol.name)

        if pr.type.endswith('64_PC32') or pr.type.endswith('64_PLT32'):
            # S + A - P
            patch_value = symbol_address + pr.fields['r_addend'] - patch_offset
            #print(f" *> {pr.type} {patch_value=} {symbol_address=} {pr.fields['r_addend']=} {pr.bits=}, {function_offset=} {patch_offset=}")

        elif pr.type == 'R_386_PC32':
            # S + A - P
            patch_value = symbol_address + pr.fields['r_addend'] - patch_offset
            #print(f" *> {pr.type}     {pr.symbol.name} {patch_value=} {symbol_address=} {pr.fields['r_addend']=} {bin(pr.fields['r_addend'])} {pr.bits=}, {function_offset=} {patch_offset=}")

        elif pr.type == 'R_386_32':
            # R_386_32
            # S + A
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x03  # Relative to data section
            #print(f" *> {pr.type} {patch_value=} {symbol_address=} {pr.fields['r_addend']=} {pr.bits=}, {function_offset=} {patch_offset=}")

        elif pr.type.endswith('_ARM_JUMP24') or pr.type.endswith('_ARM_CALL'):
            # R_ARM_JUMP24 & R_ARM_CALL
            # ((S + A) - P) >> 2
            mask = 0xffffff  # 24 bit
            patch_value = symbol_address + pr.fields['r_addend'] - patch_offset
            scale = 4

        elif pr.type.endswith('_CALL26') or pr.type.endswith('_JUMP26'):
            # R_AARCH64_CALL26
            # ((S + A) - P) >> 2
            assert pr.file.byteorder == 'little', "Big endian not supported for ARM64"
            mask = 0x3ffffff  # 26 bit (1<<26)-1
            patch_value = symbol_address + pr.fields['r_addend'] - patch_offset
            scale = 4

        elif pr.type.endswith('_ADR_PREL_PG_HI21'):
            # R_AARCH64_ADR_PREL_PG_HI21
            assert pr.file.byteorder == 'little', "Big endian not supported for ARM64"
            mask = 0  # Handled by runner
            patch_value = symbol_address + pr.fields['r_addend']
            scale = 4096
            symbol_type = symbol_type + 0x01  # HI21
            #print(f" *> {patch_value=} {symbol_address=} {pr.fields['r_addend']=}, {function_offset=}")

        elif pr.type.endswith('_LDST32_ABS_LO12_NC'):
            # R_AARCH64_LDST32_ABS_LO12_NC
            # (S + A) & 0xFFF
            mask = 0b00_1111_1111_1100_0000_0000
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x02  # Absolut value
            scale = 4
            #print(f" *> {patch_value=} {symbol_address=} {pr.fields['r_addend']=}, {function_offset=}")

        elif pr.type.endswith('_ADD_ABS_LO12_NC'):
            # R_AARCH64_ADD_ABS_LO12_NC
            # (S + A) & 0xFFF
            mask = 0b11_1111_1111_1100_0000_0000
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x02  # Absolut value
            scale = 1
            #print(f" *> {patch_value=} {symbol_address=} {pr.fields['r_addend']=}, {function_offset=}")

        elif pr.type.endswith('_LDST64_ABS_LO12_NC'):
            # R_AARCH64_LDST64_ABS_LO12_NC
            # (S + A) & 0xFFF
            mask = 0b00_0111_1111_1100_0000_0000
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x02  # Absolut value
            scale = 8
            #print(f" *> {patch_value=} {symbol_address=} {pr.fields['r_addend']=}, {function_offset=}")

        elif pr.type.endswith('_MOVW_ABS_NC'):
            # R_ARM_MOVW_ABS_NC
            # (S + A) & 0xFFFF
            mask = 0xFFFF
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x04  # Absolut value
            #print(f" *> {pr.type} {patch_value=} {symbol_address=}, {function_offset=}")

        elif pr.type.endswith('_MOVT_ABS'):
            # R_ARM_MOVT_ABS
            # (S + A) & 0xFFFF0000
            mask = 0xFFFF0000
            patch_value = symbol_address + pr.fields['r_addend']
            symbol_type = symbol_type + 0x04  # Absolut value
            scale = 0x10000

        else:
            raise NotImplementedError(f"Relocation type {pr.type} in {relocation.pelfy_reloc.target_section.name} pointing to {relocation.pelfy_reloc.symbol.name} not implemented")

        return patch_entry(mask, patch_offset, patch_value, scale, symbol_type)

    def get_stencil_code(self, name: str) -> bytes:
        """Return the striped function code for a provided function name

        Args:
            name: function name

        Returns:
            Striped function code
        """
        if name in self._stencil_cache:
            start_index, lengths = self._stencil_cache[name]
        else:
            func = self.elf.symbols[name]
            start_stencil, end_stencil = get_stencil_position(func)
            assert func.section
            start_index = func.section['sh_offset'] + func['st_value'] + start_stencil
            lengths = end_stencil - start_stencil
            self._stencil_cache[name] = (start_index, lengths)

        return self.elf.read_bytes(start_index, lengths)

    def get_sub_functions(self, names: Iterable[str]) -> set[str]:
        """Return recursively all functions called by stencils or by other functions
        Args:
            names: function or stencil names

        Returns:
            set of all sub function names
        """
        name_set: set[str] = set()
        for name in names:
            #print('- get_sub_functions: ', name)
            if name not in name_set:
                #print('||||', name)
                # assert name in self.elf.symbols, f"Stencil {name} not found" <-- see: https://github.com/Nonannet/pelfy/issues/1
                func = self.elf.symbols[name]
                for r in func.relocations:
                    if r.symbol.info == 'STT_FUNC':
                        #print('    ', r.symbol.name, r.symbol.section.type)
                        name_set.add(r.symbol.name)
                        name_set |= self.get_sub_functions([r.symbol.name])
        return name_set

    def get_type_size(self, type_name: str) -> int:
        """Returns the size of a variable type in bytes."""
        return {'int': 4, 'float': 4}[type_name]

    def get_symbol_size(self, name: str) -> int:
        """Returns the size of a specified symbol name."""
        return self.elf.symbols[name].fields['st_size']

    def get_symbol_offset(self, name: str) -> int:
        """Returns the offset of a specified symbol in the section."""
        return self.elf.symbols[name].fields['st_value']

    def get_symbol_section_index(self, name: str) -> int:
        """Returns the section index for a specified symbol name."""
        return self.elf.symbols[name].fields['st_shndx']

    def get_section_size(self, index: int) -> int:
        """Returns the size of a section specified by index."""
        return self.elf.sections[index].fields['sh_size']

    def get_section_alignment(self, index: int) -> int:
        """Returns the required alignment of a section specified by index."""
        return self.elf.sections[index].fields['sh_addralign']

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
