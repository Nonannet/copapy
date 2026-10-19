"""Tests for the decoder of the messages of the Zephyr runner (backchannel,
see docs/source/runner_transfer.md), without a runner"""
import binascii

import pytest

from copapy._serial import (MSG_DATA, MSG_DONE, MSG_IO, MSG_READY, MSG_SYNC, MSG_TEXT, MessageReader,
                            message)


class BytesConnection():
    """Connection that returns the given bytes in parts of at most chunk bytes"""
    def __init__(self, data: bytes, chunk: int = 3) -> None:
        self.data = data
        self.chunk = chunk

    def write(self, data: bytes) -> None:
        pass

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        part = self.data[:min(size, self.chunk)]
        self.data = self.data[len(part):]
        return part


def test_crc() -> None:
    # CRC-16/XMODEM check value, as crc16_itu_t(0, ...) of Zephyr
    assert binascii.crc_hqx(b'123456789', 0) == 0x31C3


def test_message_layout() -> None:
    check = MSG_DONE ^ 1 ^ 0xFF
    assert message(MSG_DONE, b'\x01') == bytes((MSG_SYNC, MSG_DONE, 1, check, 1)) + \
        binascii.crc_hqx(bytes((MSG_DONE, 1, check, 1)), 0).to_bytes(2, 'little')
    assert len(message(MSG_DATA, bytes(255))) == 261
    with pytest.raises(AssertionError):
        message(MSG_DATA, bytes(256))


def test_reader() -> None:
    stream = (b'*** Booting Zephyr OS ***\r\n' +  # console output outside of messages
              message(MSG_READY, b'\x01') +
              message(MSG_TEXT, b'Return ') + message(MSG_TEXT, b'value: 1\nREAD_DATA') +
              message(MSG_DATA, b'\xCA\x01\x02') +
              message(MSG_TEXT, b' offs=8\n') +
              message(MSG_IO, b'\x02\x01\x01led') +
              message(MSG_DONE, b'\xFF'))
    rx = MessageReader(BytesConnection(stream))
    assert rx.read() == (MSG_READY, b'\x01')
    assert rx.read() == (MSG_DATA, b'\xCA\x01\x02')
    assert rx.read() == (MSG_IO, b'\x02\x01\x01led')
    assert rx.read() == (MSG_DONE, b'\xFF')
    assert list(rx.output) == ['*** Booting Zephyr OS ***', 'Return value: 1', 'READ_DATA offs=8']
    with pytest.raises(TimeoutError):
        rx.read()


def test_reader_resync() -> None:
    # Corrupted message and a sync byte in other output are skipped without
    # waiting for the payload of a false header
    corrupted = bytearray(message(MSG_DATA, b'abc'))
    corrupted[5] ^= 0x10
    stream = bytes(corrupted) + b'x\xCA\x05y\n' + message(MSG_DONE, b'\x01')
    output: list[str] = []
    rx = MessageReader(BytesConnection(stream, chunk=100), output)
    assert rx.read() == (MSG_DONE, b'\x01')
    assert output[-1].endswith('y')
