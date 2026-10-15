"""Descriptions of the devices copapy programs can be deployed to with
SerialTarget"""
from types import SimpleNamespace


class MemRegion():
    """Memory region of a device

    Attributes:
        name: Name of the region
        start: Start address
        size: Size in bytes
    """
    def __init__(self, name: str, start: int, size: int) -> None:
        self.name = name
        self.start = start
        self.size = size

    def __repr__(self) -> str:
        return f"MemRegion({self.name}, 0x{self.start:08X}, {self.size})"


class Device():
    """Device running the Zephyr runner (targets/zephyr)

    Attributes:
        name: Name of the device (board name of Zephyr)
        arch: Architecture of the stencils
        tick: Tick of the real-time execution in seconds, the periods of the
            tasks must be multiples of it (CONFIG_SYS_CLOCK_TICKS_PER_SEC)
        frame_size: Maximum size of the command stream in a frame
            (CONFIG_COPARUN_MAX_FRAME_SIZE)
        usb_id: USB vendor and product id for finding the serial port
        mem: Memory regions by name as attributes
    """
    def __init__(self, name: str, arch: str, tick: float = 1e-4, frame_size: int = 4096,
                 usb_id: tuple[int, int] | None = None, mem: tuple[MemRegion, ...] = ()) -> None:
        self.name = name
        self.arch = arch
        self.tick = tick
        self.frame_size = frame_size
        self.usb_id = usb_id
        self.mem = SimpleNamespace(**{m.name: m for m in mem})

    def __repr__(self) -> str:
        return f"Device({self.name})"


# Teensy 4.1 (NXP i.MX RT1062, Cortex-M7), runner on its USB serial port
teensy41 = Device('teensy41', 'armv7mthumb_fpv5', usb_id=(0x2FE3, 0x0004), mem=(
    MemRegion('ITCM', 0x00000000, 0x20000),
    MemRegion('DTCM', 0x20000000, 0x20000),
    MemRegion('OCRAM2', 0x20200000, 0x80000),
    MemRegion('OCRAM', 0x20280000, 0x40000)))

# Arm MPS2 board with Cortex-M7, used in qemu (qemu-system-arm -M mps2-an500)
mps2_an500 = Device('mps2_an500', 'armv7mthumb_fpv5')
