from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serial import Serial
else:
    try:
        from serial import Serial
    except ImportError:
        Serial = None


class TransferChannel():
    """Class for handling data transfer between copapy and a target device.
    """

    def __init__(self, channel: str):
        self.channel = Serial(channel, baudrate=115200, timeout=0.01)

    def send(self, data: bytes) -> None:
        """Sends data to the target device in chunks."""
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            print(f"> Sending chunk {i//chunk_size + 1} of {(len(data) + chunk_size - 1) // chunk_size}")
            self.channel.write(data[i:i+chunk_size])
            self.channel.flush()
            ret_data = self.channel.read_all()
            print(f"> Received response: {ret_data}")

    def recv(self, size: int) -> bytes:
        """Receives data from the target device."""
        return self.channel.read(size)