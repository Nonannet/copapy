"""Target on a device running the Zephyr runner (targets/zephyr),
connected over a serial port. The command stream is sent in frames with
checksum, see targets/zephyr/src/main.c for the protocol."""
import struct
from collections import deque
import zlib
from typing import Any, Protocol
from . import _binwrite as binw
from ._basic_types import stencil_db_from_package
from ._tasks import Task, _TaskTarget
from .devices import Device, teensy41

CONTROL_ABORT = 0
CONTROL_RESET = 1
CONTROL_BOOTLOADER = 2

# Frames are sent again after these errors of the runner
_RETRY_ERRORS = ('checksum', 'timeout')
_RETRIES = 3


class Connection(Protocol):
    """Byte stream to the runner, e.g. serial.Serial of pyserial"""
    def write(self, data: bytes, /) -> Any: ...
    def flush(self) -> None: ...
    def read(self, size: int, /) -> bytes: ...
    def readline(self) -> bytes: ...


def frame(data: bytes) -> bytes:
    """Stream frame with a part of a command stream"""
    return b'CPY1' + struct.pack('<I', len(data)) + data + struct.pack('<I', zlib.crc32(data))


def control_frame(code: int) -> bytes:
    """Control frame (CONTROL_ABORT, CONTROL_RESET, CONTROL_BOOTLOADER)"""
    return b'CPYC' + struct.pack('<I', code)


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
        self.output: deque[str] = deque(maxlen=1000)  # Last lines of output of the runner
        self._data: list[bytes] = []
        self._owns_connection = connection is None
        if connection is None:
            import serial  # pyserial
            connection = serial.Serial(port or find_port(device), 115200, timeout=timeout)
        self._con: Connection = connection

        # The runner may be in the middle of a command stream of an aborted session
        self._con.write(control_frame(CONTROL_ABORT))
        self._con.flush()
        self._read_until('COPARUN_ABORT')

    def close(self) -> None:
        """Closes the serial port. The tasks keep running on the device."""
        if self._owns_connection:
            self._con.close()  # type: ignore[attr-defined]

    def __enter__(self) -> 'SerialTarget':
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _readline(self) -> str:
        raw = self._con.readline()
        if not raw.endswith(b'\n'):
            raise TimeoutError(f"No answer from the runner, output: {list(self.output)[-10:]}")
        return raw.decode('utf8', errors='replace').rstrip('\r\n')

    def _read_until(self, marker: str) -> str:
        """Reads the answer of the runner up to the line starting with marker or
        COPARUN_ERROR, returns this line"""
        while True:
            line = self._readline()
            if line.startswith(marker) or line.startswith('COPARUN_ERROR'):
                return line
            if line.startswith('COPARUN_DATA '):
                size = int(line.split()[1])
                data = b''
                while len(data) < size:
                    part = self._con.read(size - len(data))
                    if not part:
                        raise TimeoutError("Incomplete data from the runner")
                    data += part
                self._data.append(data)
            else:
                self.output.append(line)

    def _send(self, dw: binw.data_writer) -> None:
        dw.write_com(binw.Command.END_COM)
        data = dw.get_data()
        size = self.device.frame_size
        for i in range(0, len(data), size):
            last = i + size >= len(data)
            for _ in range(_RETRIES):
                self._con.write(frame(data[i:i + size]))
                self._con.flush()
                line = self._read_until('COPARUN_DONE')
                if not any(line == 'COPARUN_ERROR ' + e for e in _RETRY_ERRORS):
                    break
            if line != f"COPARUN_DONE ret={1 if last else 0}":
                raise RuntimeError(f"Runner error: {line}, output: {list(self.output)[-10:]}")

    def _read_mem(self, addr: int, lengths: int) -> bytes:
        dw = binw.data_writer(self.sdb.byteorder)
        dw.write_com(binw.Command.READ_MEM)
        dw.write_int(addr)
        dw.write_int(lengths)
        self._data = []
        self._send(dw)
        assert len(self._data) == 1 and len(self._data[0]) == lengths
        return self._data[0]

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
        self._con.write(control_frame(CONTROL_RESET))
        self._con.flush()
        self._read_until('COPARUN_RESET')
        self.running = False
