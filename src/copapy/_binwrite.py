from enum import Enum, IntEnum, IntFlag
from typing import Literal
import struct

ByteOrder = Literal['little', 'big']


def array_format(dtype: str, length: int, byteorder: ByteOrder) -> str:
    """struct format string for an array of 32 bit int or float values"""
    return {'little': '<', 'big': '>'}[byteorder] + str(length) + ('f' if dtype == 'float' else 'i')


def pack_array(values: 'tuple[int | float, ...] | list[int | float]', dtype: str, byteorder: ByteOrder) -> bytes:
    return struct.pack(array_format(dtype, len(values), byteorder), *values)

Command = Enum('Command', [('ALLOCATE_DATA', 1), ('COPY_DATA', 2),
                           ('ALLOCATE_CODE', 3), ('COPY_CODE', 4),
                           ('PATCH', 0x1000),
                           ('ENTRY_POINT', 7),
                           ('RUN_PROG', 64), ('READ_DATA', 65),
                           ('END_COM', 256), ('FREE_MEMORY', 257), ('DUMP_CODE', 258)])
COMMAND_SIZE = 4


class PatchEncoding(IntEnum):
    """How the calculated patch value is inserted into the instruction
    (must match PATCH_ENC_* in runmem.h)"""
    BITFIELD = 0         # 32 bit word, value placed at the lowest set bit of mask
    AARCH64_ADRP = 1     # AArch64 ADRP immhi:immlo (21 bit)
    ARM_MOVW_MOVT = 2    # ARM MOVW/MOVT (A1) imm4:imm12 (16 bit)
    THUMB_MOVW_MOVT = 3  # Thumb MOVW/MOVT (T3/T1) imm4:i:imm3:imm8 (16 bit)
    THUMB_BRANCH = 4     # Thumb B.W/BL (T4/T1) S:J1:J2:imm10:imm11 (24 bit)
    RISCV_S_TYPE = 5     # RISC-V store imm[11:5|4:0] (12 bit)
    RISCV_B_TYPE = 6     # RISC-V branch imm[12|10:5|4:1|11] (13 bit)
    RISCV_J_TYPE = 7     # RISC-V jal imm[20|10:1|11|19:12] (21 bit)
    RISCV_CB_TYPE = 8    # RISC-V compressed branch imm[8|4:3|7:6|2:1|5] (9 bit, 16 bit instruction)
    RISCV_CJ_TYPE = 9    # RISC-V compressed jump imm[11|4|9:8|10|6|7|3:1|5] (12 bit, 16 bit instruction)


class PatchFlag(IntFlag):
    """How the patch value is calculated (must match PATCH_FLAG_* in runmem.h)"""
    CODE = 0    # Value is relative to the start of the code memory
    DATA = 1    # Value is relative to the start of the data memory
    PC_REL = 2  # Subtract the address of the patched instruction
    PAGE = 4    # Round target and instruction address down to 4 KiB pages


class data_writer():
    def __init__(self, byteorder: ByteOrder):
        self._data: list[tuple[str, bytes, int]] = []
        self.byteorder: ByteOrder = byteorder

    def copy(self) -> 'data_writer':
        cp = data_writer(self.byteorder)
        cp._data = self._data.copy()
        return cp

    def write_int(self, value: int, num_bytes: int = 4, signed: bool = False) -> None:
        self._data.append((f"INT {value}", value.to_bytes(length=num_bytes, byteorder=self.byteorder, signed=signed), 0))

    def write_com(self, value: Command) -> None:
        self._data.append((value.name, value.value.to_bytes(length=COMMAND_SIZE, byteorder=self.byteorder, signed=False), 1))

    def write_byte(self, value: int) -> None:
        self._data.append((f"BYTE {value}", bytes([value]), 0))

    def write_bytes(self, value: bytes) -> None:
        self._data.append((f"BYTES {len(value)}", value, 0))

    def write_value(self, value: int | float, num_bytes: int = 4) -> None:
        if isinstance(value, int):
            self.write_int(value, num_bytes, True)
        else:
            # 32 bit or 64 bit float
            en = {'little': '<', 'big': '>'}[self.byteorder]
            if num_bytes == 4:
                data = struct.pack(en + 'f', value)
            else:
                data = struct.pack(en + 'd', value)
            assert len(data) == num_bytes, (len(data), num_bytes)
            self.write_bytes(data)

    def print(self) -> None:
        for name, dat, flag in self._data:
            if flag:
                print('')
            print(f"{name:18}" + ' '.join(f'{b:02X}' for b in dat))

    def get_data(self) -> bytes:
        return b''.join(dat for _, dat, _ in self._data)

    def to_file(self, path: str) -> None:
        with open(path, 'wb') as f:
            f.write(self.get_data())


class data_reader():
    def __init__(self, data: bytes | bytearray, byteorder: ByteOrder):
        self._data = data
        self._index: int = 0
        self.byteorder: ByteOrder = byteorder

    def read_int(self, num_bytes: int = 4, signed: bool = False) -> int:
        ret = int.from_bytes(self._data[self._index:self._index + num_bytes], byteorder=self.byteorder, signed=signed)
        self._index += num_bytes
        return ret

    def read_com(self) -> Command:
        com_value = int.from_bytes(self._data[self._index:self._index + COMMAND_SIZE], byteorder=self.byteorder)
        ret = Command(com_value)
        self._index += COMMAND_SIZE
        return ret

    def read_byte(self) -> int:
        ret = self._data[self._index]
        self._index += 1
        return ret

    def read_bytes(self, num_bytes: int) -> bytes | bytearray:
        ret = self._data[self._index:self._index + num_bytes]
        self._index += num_bytes
        return ret
