"""I/O channels of a device (e.g. GPIO pins), defined by the runner"""
from typing import Iterator


class IoChannel():
    """I/O channel of a device

    Attributes:
        index: Number of the channel on the runner
        kind: Kind of the channel, e.g. 'gpio_in' or 'gpio_out'
        name: Name of the channel (devicetree node name on Zephyr)
    """
    def __init__(self, index: int, kind: str, name: str) -> None:
        self.index = index
        self.kind = kind
        self.name = name

    @property
    def input(self) -> bool:
        """The channel is read by the device"""
        return self.kind.endswith('_in')

    def __repr__(self) -> str:
        return f"IoChannel({self.name!r}, {self.kind})"


class IoChannels():
    """I/O channels of a device by name"""
    def __init__(self, channels: list[IoChannel]) -> None:
        self._channels = {c.name: c for c in channels}

    def _get(self, name: str, kinds: tuple[str, ...]) -> IoChannel:
        channel = self._channels.get(name)
        if channel is None or channel.kind not in kinds:
            available = [c.name for c in self._channels.values() if c.kind in kinds]
            raise KeyError(f"No {'/'.join(kinds)} channel {name!r}, available: {', '.join(available) or 'none'}")
        return channel

    def gpio(self, name: str) -> IoChannel:
        """GPIO channel (input or output) by name"""
        return self._get(name, ('gpio_in', 'gpio_out'))

    def __getitem__(self, name: str) -> IoChannel:
        return self._get(name, tuple({c.kind for c in self._channels.values()}))

    def __contains__(self, channel: object) -> bool:
        return isinstance(channel, IoChannel) and self._channels.get(channel.name) is channel

    def __iter__(self) -> Iterator[IoChannel]:
        return iter(self._channels.values())

    def __len__(self) -> int:
        return len(self._channels)

    def __repr__(self) -> str:
        return '[' + ', '.join(repr(c) for c in self._channels.values()) + ']'
