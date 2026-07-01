"""
Byte transports for talking to the module.

The library is deliberately decoupled from any particular I/O mechanism: the
:class:`HWT901B` sensor only needs an object that can ``read`` and ``write``
bytes (the :class:`Transport` protocol). This keeps the core dependency-free and
lets you drive the sensor from a serial port, a TCP bridge, a replayed capture
file, or a unit-test fake.

``pyserial`` is an *optional* dependency: it is imported lazily inside
:class:`SerialTransport` so ``import hwt901b`` works with zero third-party
packages installed. Install it with ``pip install hwt901b[serial]`` (or just
``pip install pyserial``) when you actually need a live link. ``pyserial``
itself is cross-platform, so :class:`SerialTransport` works identically on
Windows (``COM3``) and Linux (``/dev/ttyUSB0``).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Minimal byte-stream interface the sensor depends on."""

    def read(self, size: int) -> bytes:
        """Read up to *size* bytes. May return fewer (or none) on timeout."""

    def write(self, data: bytes) -> int:
        """Write *data*, returning the number of bytes written."""

    def close(self) -> None:
        ...


class SerialTransport:
    """A :class:`Transport` backed by ``pyserial``.

    Parameters mirror the sensor defaults: 9600 baud, 8N1. The HWT901B ships at
    9600 baud but is commonly reconfigured to 115200 for high output rates.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 0.1,
        write_timeout: float = 1.0,
    ) -> None:
        try:
            import serial  # noqa: PLC0415 -- lazy so the core stays dep-free
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "SerialTransport requires pyserial. Install it with "
                "'pip install hwt901b[serial]' or 'pip install pyserial'."
            ) from exc

        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=write_timeout,
        )

    def read(self, size: int) -> bytes:
        return self._serial.read(size)

    def write(self, data: bytes) -> int:
        n = self._serial.write(data)
        self._serial.flush()
        return n or 0

    def reset_input(self) -> None:
        """Drop any buffered inbound bytes (useful after a config change)."""
        self._serial.reset_input_buffer()

    @property
    def baudrate(self) -> int:
        return self._serial.baudrate

    @baudrate.setter
    def baudrate(self, value: int) -> None:
        # Changing the host side to match a module baud-rate switch, without
        # reopening the port.
        self._serial.baudrate = value

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> "SerialTransport":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BytesTransport:
    """An in-memory :class:`Transport` over a fixed buffer.

    Handy for tests and for replaying a recorded capture through the same code
    path as a live sensor. Writes are captured in :attr:`written`.
    """

    def __init__(self, data: bytes = b"") -> None:
        self._data = bytes(data)
        self._pos = 0
        self.written = bytearray()

    def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def close(self) -> None:  # noqa: D401 - nothing to release
        pass
