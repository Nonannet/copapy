from ast import Tuple
from os import name
from tkinter import NO
import pelfy
from typing import Generator, Literal


start_marker = 0xF17ECAFE
end_marker = 0xF27ECAFE

LENGTH_CALL_INSTRUCTION = 4  # x86_64

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


    def get_relocs(self, symbol_name: str) -> Generator[tuple[int, int, str], None, None]:
        """Return relocation offset relative to striped symbol start.
        Yields tuples of (reloc_offset, symbol_lenght, bits, reloc_type)
        1. reloc_offset: Offset of the relocation relative to the start of the stripped symbol data.
        2. Length of the striped symbol.
        3. Bits to patch
        4. reloc_type: Type of the relocation as a string.
        """
        symbol = self.elf.symbols[symbol_name]
        start_index = symbol.data.find(start_marker.to_bytes(4, symbol.file.byteorder))
        end_index = symbol.data.rfind(end_marker.to_bytes(4, symbol.file.byteorder), start_index)
        
        for reloc in symbol.relocations:
            
            reloc_offset = reloc.fields['r_offset'] - symbol.fields['st_value'] - start_index

            if reloc_offset < end_index - start_index - LENGTH_CALL_INSTRUCTION:
                yield (reloc_offset, reloc.bits, reloc.type)


    def get_func_data(self, name: str) -> bytes:
        return strip_symbol(self.elf.symbols[name].data, self.elf.byteorder)


