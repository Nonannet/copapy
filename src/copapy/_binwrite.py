from enum import Enum
from typing import Literal
import struct

ByteOrder = Literal['little', 'big']

Command = Enum('Command', [('ALLOCATE_DATA', 1), ('COPY_DATA', 2),
                           ('ALLOCATE_CODE', 3), ('COPY_CODE', 4),
                           ('PATCH_FUNC', 0x1000), ('PATCH_OBJECT', 0x2000),
                           ('PATCH_OBJECT_HI21', 0x2001),
                           ('PATCH_OBJECT_ABS', 0x2002),
                           ('PATCH_OBJECT_REL', 0x2003),
                           ('PATCH_OBJECT_ARM32_ABS', 0x2004),
                           ('ENTRY_POINT', 7),
                           ('RUN_PROG', 64), ('READ_DATA', 65),
                           ('END_COM', 256), ('FREE_MEMORY', 257), ('DUMP_CODE', 258)])
COMMAND_SIZE = 4


class data_writer():
    def __init__(self, byteorder: ByteOrder):
        self._data: list[tuple[str, bytes, int]] = []
        self.byteorder: ByteOrder = byteorder

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
