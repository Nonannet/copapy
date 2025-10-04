from enum import Enum
from typing import Literal
import struct


Command = Enum('Command', [('ALLOCATE_DATA', 1), ('COPY_DATA', 2),
                           ('ALLOCATE_CODE', 3), ('COPY_CODE', 4),
                           ('PATCH_FUNC', 5), ('PATCH_OBJECT', 6),
                           ('RUN_PROG', 64), ('READ_DATA', 65),
                           ('END_PROG', 256), ('FREE_MEMORY', 257)])


class data_writer():
    def __init__(self, byteorder: Literal['little', 'big']):
        self._data: list[tuple[str, bytes, int]] = list()
        self.byteorder: Literal['little', 'big'] = byteorder

    def write_int(self, value: int, num_bytes: int = 4, signed: bool = False) -> None:
        self._data.append((f"INT {value}", value.to_bytes(length=num_bytes, byteorder=self.byteorder, signed=signed), 0))

    def write_com(self, value: Enum, num_bytes: int = 4) -> None:
        self._data.append((value.name, value.value.to_bytes(length=num_bytes, byteorder=self.byteorder, signed=False), 1))

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
            assert len(data) == num_bytes
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
