from dataclasses import dataclass
import pelfy
from typing import Generator, Literal
from enum import Enum


start_marker = 0xF17ECAFE
end_marker = 0xF27ECAFE

LENGTH_CALL_INSTRUCTION = 4  # x86_64

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


def translate_relocation(relocation_addr: int, reloc_type: str, bits: int, r_addend: int) -> patch_entry:
    if reloc_type in ('R_AMD64_PLT32', 'R_AMD64_PC32'):
        # S + A - P
        patch_offset = relocation_addr - r_addend
        return patch_entry(RelocationType.RELOC_RELATIVE_32, patch_offset)
    else:
        raise Exception(f"Unknown: {reloc_type}")


def get_ret_function_def(symbol: pelfy.elf_symbol):
    #print('*', symbol.name, symbol.section)
    result_func = symbol.relocations[-1].symbol

    assert result_func.name.startswith('result_')
    return result_func.name[7:]


def strip_symbol(data: bytes, byteorder: Literal['little', 'big']) -> bytes:
    """Return data between start and end marker and removing last instruction (call)"""

    # Find first start marker
    start_index = data.find(start_marker.to_bytes(4, byteorder))

    # Find last end marker
    end_index = data.rfind(end_marker.to_bytes(4, byteorder), start_index)

    assert start_index > -1 and end_index > -1, f"Marker not found"
    return data[start_index + 4:end_index - LENGTH_CALL_INSTRUCTION]



class stencil_database():
    def __init__(self, obj_file: str):
        self.elf = pelfy.open_elf_file(obj_file)

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
        start_index = symbol.data.find(start_marker.to_bytes(4, symbol.file.byteorder))
        end_index = symbol.data.rfind(end_marker.to_bytes(4, symbol.file.byteorder), start_index)
        
        for reloc in symbol.relocations:
            
            # address to fist byte to patch relative to the start of the symbol
            patch = translate_relocation(
                reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index,
                reloc.type,
                reloc.bits,
                reloc.fields['r_addend'])

            # Exclude the call to the  result_x function
            if patch.addr < end_index - start_index - LENGTH_CALL_INSTRUCTION:
                yield patch


    def get_func_data(self, name: str) -> bytes:
        return strip_symbol(self.elf.symbols[name].data, self.elf.byteorder)


