"""I/O channels of a device (e.g. GPIO pins), defined by the runner"""
from typing import Iterator


# Type of the values of each kind of channel
_DTYPES = {'gpio_in': 'int', 'gpio_out': 'int', 'pwm_out': 'float', 'encoder_in': 'int', 'adc_in': 'int'}


class IoChannel():
    """I/O channel of a device

    Attributes:
        index: Number of the channel on the runner
        kind: Kind of the channel: 'gpio_in', 'gpio_out' (bool), 'pwm_out' (float
            duty cycle 0..1), 'encoder_in' (int position count), 'adc_in' (int raw
            conversion result)
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

    @property
    def readable(self) -> bool:
        """The channel can be read: inputs and the level of GPIO outputs"""
        return self.input or self.kind == 'gpio_out'

    @property
    def dtype(self) -> str:
        """Type of the values of the channel"""
        return _DTYPES.get(self.kind, 'int')

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
        """GPIO channel (input or output, bool values) by name"""
        return self._get(name, ('gpio_in', 'gpio_out'))

    def pwm(self, name: str) -> IoChannel:
        """PWM output (float duty cycle 0..1) by name"""
        return self._get(name, ('pwm_out',))

    def encoder(self, name: str) -> IoChannel:
        """Quadrature encoder input (int position count) by name"""
        return self._get(name, ('encoder_in',))

    def adc(self, name: str) -> IoChannel:
        """ADC input (int raw conversion result) by name"""
        return self._get(name, ('adc_in',))

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
