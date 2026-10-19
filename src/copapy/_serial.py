"""Target on a device running the Zephyr runner (targets/zephyr),
connected over a serial port. The command stream is sent in frames with
checksum, the runner answers with binary messages, see
docs/source/runner_transfer.md for the protocol."""
import binascii
import struct
from collections import deque
import zlib
from typing import Any, Protocol
from . import _binwrite as binw
from ._basic_types import stencil_db_from_package
from ._tasks import Task, _TaskTarget
from .devices import Device, teensy41
from ._io import IoChannel, IoChannels

CONTROL_ABORT = 0
CONTROL_RESET = 1
CONTROL_BOOTLOADER = 2

# Messages of the runner (targets/zephyr/src/protocol.h)
MSG_SYNC = 0xCA
MSG_READY = 0x01    # uint8 protocol version
MSG_DONE = 0x02     # int8 state flag of the parser
MSG_ERROR = 0x03    # uint8 error code (ERRORS)
MSG_CONTROL = 0x04  # uint8 control code
MSG_DATA = 0x05     # part of the data of READ_MEM
MSG_TEXT = 0x06     # part of the text output
MSG_IO = 0x07       # uint16 channel, uint8 kind (IO_KINDS), name

ERRORS = {1: 'timeout', 2: 'checksum', 3: 'frame too large', 4: 'unknown control code'}
IO_KINDS = ('gpio_in', 'gpio_out', 'pwm_out', 'encoder_in', 'adc_in')

# Frames are sent again after these errors of the runner (timeout, checksum)
_RETRY_ERRORS = (1, 2)
_RETRIES = 3


class Connection(Protocol):
    """Byte stream to the runner, e.g. serial.Serial of pyserial"""
    def write(self, data: bytes, /) -> Any: ...
    def flush(self) -> None: ...
    def read(self, size: int, /) -> bytes: ...


def frame(data: bytes) -> bytes:
    """Stream frame with a part of a command stream"""
    return b'CPY1' + struct.pack('<I', len(data)) + data + struct.pack('<I', zlib.crc32(data))


def control_frame(code: int) -> bytes:
    """Control frame (CONTROL_ABORT, CONTROL_RESET, CONTROL_BOOTLOADER)"""
    return b'CPYC' + struct.pack('<I', code)


def message(msg_type: int, payload: bytes = b'') -> bytes:
    """Message of the runner to the host (encoder of the backchannel)"""
    assert len(payload) <= 255
    header = bytes((MSG_SYNC, msg_type, len(payload), msg_type ^ len(payload) ^ 0xFF))
    return header + payload + struct.pack('<H', binascii.crc_hqx(header[1:] + payload, 0))


class MessageReader():
    """Decodes the messages of the runner from a byte stream. Text output of
    the runner and bytes outside of messages (e.g. Zephyr console output) are
    collected as lines in output."""
    def __init__(self, con: Connection, output: 'deque[str] | list[str] | None' = None) -> None:
        self._con = con
        self._buffer = bytearray()  # Bytes received but not decoded
        self._line = bytearray()  # Incomplete line of text output
        self.output: deque[str] | list[str] = deque(maxlen=1000) if output is None else output

    def _read(self, size: int) -> bytes:
        while len(self._buffer) < size:
            part = self._con.read(size - len(self._buffer))
            if not part:
                raise TimeoutError(f"No answer from the runner, output: {list(self.output)[-10:]}")
            self._buffer += part
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def _add_text(self, text: bytes) -> None:
        self._line += text
        *lines, rest = self._line.split(b'\n')
        for line in lines:
            self.output.append(line.decode('utf8', errors='replace').rstrip('\r'))
        self._line = bytearray(rest)

    def read(self) -> tuple[int, bytes]:
        """Reads the next message that is not text output

        Returns:
            Message type and payload
        """
        while True:
            sync = self._read(1)
            if sync[0] != MSG_SYNC:
                self._add_text(sync)
                continue
            header = self._read(3)
            if header[0] ^ header[1] ^ header[2] != 0xFF:
                # No message: search the next sync byte after this one
                self._buffer[:0] = header
                self._add_text(sync)
                continue
            rest = self._read(header[1] + 2)
            payload = rest[:-2]
            if binascii.crc_hqx(header + payload, 0) != struct.unpack('<H', rest[-2:])[0]:
                # Corrupted message: search the next sync byte after this one
                self._buffer[:0] = header + rest
                self._add_text(sync)
                continue
            if header[0] == MSG_TEXT:
                self._add_text(payload)
                continue
            return header[0], payload


def find_port(device: Device) -> str:
    """Serial port of a device by its USB id (requires pyserial)

    Arguments:
        device: Device description with usb_id

    Returns:
        Name of the serial port, e.g. 'COM11' or '/dev/ttyACM0'
    """
    from serial.tools import list_ports  # pyserial

    if not device.usb_id:
        raise ValueError(f"No USB id known for {device}, the port must be specified")
    vid, pid = device.usb_id
    ports = [p.device for p in list_ports.comports() if p.vid == vid and p.pid == pid]
    if not ports:
        raise RuntimeError(f"No serial port with USB id {vid:04X}:{pid:04X} of {device} found")
    if len(ports) > 1:
        raise RuntimeError(f"Several serial ports for {device} found ({', '.join(ports)}), the port must be specified")
    return ports[0]


class SerialTarget(_TaskTarget):
    """Target on a device running the Zephyr runner, connected over a serial
    port. The periodic tasks run in real time on the device after run()."""
    def __init__(self, port: str | None = None, device: Device = teensy41, optimization: str = 'O3',
                 timeout: float = 10.0, connection: Connection | None = None) -> None:
        """Connect to a device.

        Arguments:
            port: Serial port (e.g. 'COM11' or '/dev/ttyACM0'), by default the port is
                searched by the USB id of the device. Requires pyserial.
            device: Device description, e.g. copapy.devices.teensy41
            optimization: Optimization level of the stencils
            timeout: Timeout for answers of the runner in seconds
            connection: Already opened byte stream to the runner instead of a port
        """
        self.device = device
        self.sdb = stencil_db_from_package(device.arch, optimization)
        self._values = {}
        self._init_tasks(device.tick)
        self.running = False
        self._data = bytearray()
        self._io_channels: list[IoChannel] = []
        self._owns_connection = connection is None
        if connection is None:
            import serial  # pyserial
            connection = serial.Serial(port or find_port(device), 115200, timeout=timeout)
        self._con: Connection = connection
        self._rx = MessageReader(connection)
        self.output = self._rx.output  # Last lines of output of the runner

        # The runner may be in the middle of a command stream of an aborted session
        self._control(CONTROL_ABORT)

        self.io = self._list_io()

    def _list_io(self) -> IoChannels:
        """I/O channels of the device, empty if not supported by the runner"""
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.LIST_IO)
        self._io_channels = []
        try:
            self._send(dw)
        except RuntimeError:
            return IoChannels([])  # Runner without LIST_IO
        return IoChannels(self._io_channels)

    def close(self) -> None:
        """Closes the serial port. The tasks keep running on the device."""
        if self._owns_connection:
            self._con.close()  # type: ignore[attr-defined]

    def __enter__(self) -> 'SerialTarget':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _control(self, code: int) -> None:
        """Sends a control frame and waits for its acknowledgement, other
        messages (e.g. of an aborted session) are skipped"""
        self._con.write(control_frame(code))
        self._con.flush()
        while self._rx.read() != (MSG_CONTROL, bytes((code,))):
            pass

    def _receive(self) -> tuple[int, int]:
        """Reads the answer of the runner to a stream frame

        Returns:
            MSG_DONE and the state flag or MSG_ERROR and the error code
        """
        while True:
            msg_type, payload = self._rx.read()
            if msg_type == MSG_DONE:
                return msg_type, struct.unpack('<b', payload)[0]
            if msg_type == MSG_ERROR:
                return msg_type, payload[0]
            if msg_type == MSG_DATA:
                self._data += payload
            elif msg_type == MSG_IO:
                index, kind = struct.unpack_from('<HB', payload)
                name = payload[3:].decode('utf8', errors='replace')
                self._io_channels.append(IoChannel(index, IO_KINDS[kind] if kind < len(IO_KINDS) else f'kind{kind}', name))

    def _send(self, dw: binw.data_writer) -> None:
        dw.write_com(binw.Command.END_COM)
        data = dw.get_data()
        size = self.device.frame_size
        for i in range(0, len(data), size):
            last = i + size >= len(data)
            answer = (MSG_ERROR, 0)
            for _ in range(_RETRIES):
                self._con.write(frame(data[i:i + size]))
                self._con.flush()
                answer = self._receive()
                if answer[0] != MSG_ERROR or answer[1] not in _RETRY_ERRORS:
                    break
            if answer != (MSG_DONE, 1 if last else 0):
                msg_type, value = answer
                reason = f"state flag {value}" if msg_type == MSG_DONE else ERRORS.get(value, f"error {value}")
                raise RuntimeError(f"Runner error: {reason}, output: {list(self.output)[-10:]}")

    def _read_mem(self, addr: int, lengths: int) -> bytes:
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.READ_MEM)
        dw.write_int(addr)
        dw.write_int(lengths)
        self._data = bytearray()
        self._send(dw)
        assert len(self._data) == lengths
        return bytes(self._data)

    def deploy(self, *tasks: Any) -> None:
        """Compiles the tasks for the device and loads them. Previously
        deployed tasks are removed, the device must be stopped.

        Arguments:
            tasks: Tasks or sequences of tasks. All other arguments (values,
                vectors, tensors, arrays and sequences of them) are the outputs
                of one additional task without period.
        """
        if self.running:
            raise RuntimeError("The tasks are running, stop() must be called before deploy")
        super().deploy(*tasks)

    def run(self) -> None:
        """Starts the real-time execution of the periodic tasks, all tasks are
        released at the start. Inputs can be written and outputs read
        while the tasks are running."""
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.START_TASKS)
        dw.write_int(round(self.tick * 1e6))
        self._send(dw)
        self.running = True

    def stop(self) -> None:
        """Stops the real-time execution after the running jobs"""
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.STOP_TASKS)
        self._send(dw)
        self.running = False

    def exec(self, task: Task | None = None) -> None:
        """Runs one job of a deployed task: reading inputs, computing and
        publishing the outputs. While the tasks are running, only tasks
        without period can be run.

        Arguments:
            task: Task to run, can be omitted if only one task is deployed
        """
        if self.running and self._tasks[self._task_index(task)].period is not None:
            raise RuntimeError("Periodic tasks can only be run by exec if the tasks are stopped")
        super().exec(task)

    def step(self, until: float) -> None:
        """Runs the periodic tasks in simulated time (not in real time) on the
        device, see LocalTarget.step. The tasks must be stopped."""
        if self.running:
            raise RuntimeError("Simulated time is not available while the tasks are running")
        super().step(until)

    def reset(self) -> None:
        """Resets the device, the connection must be opened again afterwards"""
        self._control(CONTROL_RESET)
        self.running = False
