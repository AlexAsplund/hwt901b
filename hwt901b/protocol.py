"""
Pure-stdlib implementation of the WitMotion "WIT standard" binary protocol as
used by the HWT901B-TTL 9/10-axis AHRS module.

Nothing in this module imports anything outside the Python standard library, so
it can be used to parse a capture file, a socket, an MQTT payload, or anything
else that yields bytes -- the physical serial link lives in :mod:`hwt901b.transport`.

Wire format
-----------
The module streams fixed 11-byte little-endian frames:

    byte 0        0x55                    frame header
    byte 1        0x5?                    packet type (see :class:`PacketType`)
    bytes 2..9    8 payload bytes         four little-endian int16 values
    byte 10       checksum                (sum of bytes 0..9) & 0xFF

Every int16 is signed, low byte first: ``value = int16(high << 8 | low)``.

Configuration is written back with 5-byte commands of the form::

    0xFF 0xAA <reg> <data_low> <data_high>

References
----------
* WIT Standard Communication Protocol
  https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/wit-standard-communication-protocol
* HWT901B-TTL datasheet v20-0707
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

# --------------------------------------------------------------------------- #
# Framing constants
# --------------------------------------------------------------------------- #

FRAME_HEADER = 0x55
FRAME_LEN = 11
PAYLOAD_LEN = 8  # four int16 values

WRITE_HEADER = (0xFF, 0xAA)

# Full-scale ranges the module reports in (fixed in firmware for the HWT901B).
ACC_RANGE_G = 16.0          # +/- 16 g
GYRO_RANGE_DPS = 2000.0     # +/- 2000 deg/s
ANGLE_RANGE_DEG = 180.0     # +/- 180 deg
SHORT_SCALE = 32768.0       # 2**15


class PacketType(IntEnum):
    """Second byte of a 0x55 frame."""

    TIME = 0x50
    ACCELERATION = 0x51
    ANGULAR_VELOCITY = 0x52
    ANGLE = 0x53
    MAGNETIC = 0x54
    PORT = 0x55
    PRESSURE = 0x56
    LAT_LON = 0x57
    GROUND_SPEED = 0x58
    QUATERNION = 0x59
    GPS_ACCURACY = 0x5A
    READ_REGISTER = 0x5F


class Register(IntEnum):
    """Configuration register addresses (subset relevant to the HWT901B)."""

    SAVE = 0x00        # 0x0000 save, 0x0001 factory reset, 0x00FF reboot
    CALSW = 0x01       # calibration mode selector (see :class:`CalibrationMode`)
    RSW = 0x02         # bitmask of which packet types to emit
    RRATE = 0x03       # output rate (see :class:`OutputRate`)
    BAUD = 0x04        # baud rate (see :class:`BaudRate`)
    AXOFFSET = 0x05
    AYOFFSET = 0x06
    AZOFFSET = 0x07
    GXOFFSET = 0x08
    GYOFFSET = 0x09
    GZOFFSET = 0x0A
    HXOFFSET = 0x0B
    HYOFFSET = 0x0C
    HZOFFSET = 0x0D
    IICADDR = 0x1A
    LEDOFF = 0x1B
    BANDWIDTH = 0x1F   # low-pass filter cutoff (see :class:`Bandwidth`)
    GYRORANGE = 0x20
    ACCRANGE = 0x21
    SLEEP = 0x22       # 0x0001 -> sleep, wakes on serial traffic
    ORIENT = 0x23      # 0 horizontal, 1 vertical mounting
    AXIS6 = 0x24       # 0 = 9-axis (absolute yaw), 1 = 6-axis (relative yaw)
    FILTK = 0x25       # dynamic filter K value
    READADDR = 0x27    # register to read back
    VERSION = 0x2E
    KEY = 0x69         # unlock key register


class CalibrationMode(IntEnum):
    """Values written to :attr:`Register.CALSW`."""

    NONE = 0x00              # normal operation / end calibration
    ACCELERATION = 0x01      # automatic accelerometer (gravity) calibration
    HEIGHT_RESET = 0x03      # zero the barometric altitude
    HEADING_RESET = 0x04     # set current yaw as zero (6-axis mode)
    MAGNETIC = 0x07          # magnetic-field (spherical) calibration
    ANGLE_REFERENCE = 0x08   # set current attitude as reference
    MAGNETIC_DUAL = 0x09     # magnetic dual-plane calibration


class OutputRate(IntEnum):
    """Values for :attr:`Register.RRATE` and their frequency in Hz."""

    HZ_0_2 = 0x01
    HZ_0_5 = 0x02
    HZ_1 = 0x03
    HZ_2 = 0x04
    HZ_5 = 0x05
    HZ_10 = 0x06
    HZ_20 = 0x07
    HZ_50 = 0x08
    HZ_100 = 0x09
    HZ_125 = 0x0A
    HZ_200 = 0x0B

    @property
    def hz(self) -> float:
        return {
            0x01: 0.2, 0x02: 0.5, 0x03: 1, 0x04: 2, 0x05: 5, 0x06: 10,
            0x07: 20, 0x08: 50, 0x09: 100, 0x0A: 125, 0x0B: 200,
        }[int(self)]


class BaudRate(IntEnum):
    """Values for :attr:`Register.BAUD` and their bits/second."""

    BPS_4800 = 0x01
    BPS_9600 = 0x02
    BPS_19200 = 0x03
    BPS_38400 = 0x04
    BPS_57600 = 0x05
    BPS_115200 = 0x06
    BPS_230400 = 0x07

    @property
    def bps(self) -> int:
        return {
            0x01: 4800, 0x02: 9600, 0x03: 19200, 0x04: 38400,
            0x05: 57600, 0x06: 115200, 0x07: 230400,
        }[int(self)]

    @classmethod
    def from_bps(cls, bps: int) -> "BaudRate":
        for member in cls:
            if member.bps == bps:
                return member
        raise ValueError(f"unsupported baud rate: {bps}")


class Bandwidth(IntEnum):
    """Low-pass filter cutoff for :attr:`Register.BANDWIDTH`."""

    HZ_256 = 0x00
    HZ_188 = 0x01
    HZ_98 = 0x02
    HZ_42 = 0x03
    HZ_20 = 0x04
    HZ_10 = 0x05
    HZ_5 = 0x06


class RswBit(IntEnum):
    """Bit positions in :attr:`Register.RSW` selecting which frames are emitted."""

    TIME = 0
    ACCELERATION = 1
    ANGULAR_VELOCITY = 2
    ANGLE = 3
    MAGNETIC = 4
    PORT = 5
    PRESSURE = 6
    LAT_LON = 7
    GROUND_SPEED = 8
    QUATERNION = 9
    GPS_ACCURACY = 10


def rsw_mask(*bits: RswBit) -> int:
    """Build an :attr:`Register.RSW` bitmask from selected outputs."""
    mask = 0
    for bit in bits:
        mask |= 1 << int(bit)
    return mask


# --------------------------------------------------------------------------- #
# Decoded packet payloads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Acceleration:
    """0x51 -- acceleration in g plus die temperature in deg C."""

    x: float
    y: float
    z: float
    temperature: float


@dataclass(frozen=True)
class AngularVelocity:
    """0x52 -- angular velocity in deg/s (voltage field is chip-specific)."""

    x: float
    y: float
    z: float
    voltage: float


@dataclass(frozen=True)
class Angle:
    """0x53 -- roll/pitch/yaw in degrees plus firmware version."""

    roll: float
    pitch: float
    yaw: float
    version: int


@dataclass(frozen=True)
class Magnetic:
    """0x54 -- raw magnetometer counts plus die temperature in deg C."""

    x: float
    y: float
    z: float
    temperature: float


@dataclass(frozen=True)
class Quaternion:
    """0x59 -- normalized orientation quaternion (w, x, y, z)."""

    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Pressure:
    """0x56 -- barometric pressure in Pa and altitude in cm."""

    pressure: int
    altitude: int


@dataclass(frozen=True)
class Time:
    """0x50 -- on-board real-time clock."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    millisecond: int


@dataclass(frozen=True)
class RegisterRead:
    """0x5F -- four consecutive register words starting at :attr:`start`."""

    start: int
    values: tuple  # four signed int16 words


@dataclass
class RawFrame:
    """A verified 11-byte frame prior to type-specific decoding."""

    type: int
    payload: bytes  # 8 bytes

    def int16(self) -> tuple:
        """Unpack the payload as four signed little-endian int16 values."""
        return struct.unpack("<4h", self.payload)

    def uint16(self) -> tuple:
        return struct.unpack("<4H", self.payload)


def checksum(frame: bytes) -> int:
    """Return the WitMotion checksum for the first 10 bytes of *frame*."""
    return sum(frame[:10]) & 0xFF


def verify(frame: bytes) -> bool:
    """True if *frame* is a well-formed, correctly check-summed 0x55 frame."""
    return (
        len(frame) == FRAME_LEN
        and frame[0] == FRAME_HEADER
        and frame[10] == checksum(frame)
    )


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def _scaled(values, factor):
    return tuple(v / SHORT_SCALE * factor for v in values)


def decode(frame: RawFrame):
    """Decode a verified :class:`RawFrame` into a typed payload object.

    Returns ``None`` for frame types this library does not model, so callers
    can simply skip unknown packets.
    """
    v = frame.int16()
    t = frame.type

    if t == PacketType.ACCELERATION:
        ax, ay, az = _scaled(v[:3], ACC_RANGE_G)
        return Acceleration(ax, ay, az, v[3] / 100.0)
    if t == PacketType.ANGULAR_VELOCITY:
        wx, wy, wz = _scaled(v[:3], GYRO_RANGE_DPS)
        return AngularVelocity(wx, wy, wz, v[3] / 100.0)
    if t == PacketType.ANGLE:
        roll, pitch, yaw = _scaled(v[:3], ANGLE_RANGE_DEG)
        return Angle(roll, pitch, yaw, v[3] & 0xFFFF)
    if t == PacketType.MAGNETIC:
        return Magnetic(float(v[0]), float(v[1]), float(v[2]), v[3] / 100.0)
    if t == PacketType.QUATERNION:
        w, x, y, z = _scaled(v, 1.0)
        return Quaternion(w, x, y, z)
    if t == PacketType.PRESSURE:
        pressure, altitude = struct.unpack("<2i", frame.payload)
        return Pressure(pressure, altitude)
    if t == PacketType.TIME:
        p = frame.payload
        ms = p[6] | (p[7] << 8)
        return Time(2000 + p[0], p[1], p[2], p[3], p[4], p[5], ms)
    if t == PacketType.READ_REGISTER:
        # The 0x5F payload echoes four register words; the caller knows which
        # start address it asked for via READADDR, so we leave start unset.
        return RegisterRead(start=-1, values=v)
    return None


# --------------------------------------------------------------------------- #
# Streaming frame parser (byte-at-a-time, resynchronising)
# --------------------------------------------------------------------------- #


class FrameParser:
    """Incremental parser turning an arbitrary byte stream into frames.

    Feed it whatever chunks arrive from the transport; it buffers partial
    frames, resynchronises after corruption, and yields verified
    :class:`RawFrame` objects. It never blocks and holds no I/O.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> Iterator[RawFrame]:
        """Append *data* and yield every complete, valid frame it produces."""
        self._buf.extend(data)
        buf = self._buf
        while True:
            # Discard bytes until a header candidate is at index 0.
            start = buf.find(FRAME_HEADER)
            if start == -1:
                buf.clear()
                return
            if start:
                del buf[:start]
            if len(buf) < FRAME_LEN:
                return
            frame = bytes(buf[:FRAME_LEN])
            if frame[10] == checksum(frame):
                del buf[:FRAME_LEN]
                yield RawFrame(type=frame[1], payload=frame[2:10])
            else:
                # False header (checksum mismatch): drop one byte and rescan.
                del buf[:1]


# --------------------------------------------------------------------------- #
# Command builders (host -> module)
# --------------------------------------------------------------------------- #


def write_command(register: int, value: int) -> bytes:
    """Build a 5-byte register write: ``FF AA reg lo hi``."""
    value &= 0xFFFF
    return bytes((*WRITE_HEADER, register & 0xFF, value & 0xFF, value >> 8))


def unlock_command() -> bytes:
    """The mandatory unlock preamble (``FF AA 69 88 B5``).

    Config registers relock automatically ~10 s after the last write.
    """
    return bytes((*WRITE_HEADER, Register.KEY, 0x88, 0xB5))


def save_command() -> bytes:
    """Persist the current register set to flash (``FF AA 00 00 00``)."""
    return write_command(Register.SAVE, 0x0000)


def read_command(register: int) -> bytes:
    """Request the four register words starting at *register* (``FF AA 27 reg 00``)."""
    return bytes((*WRITE_HEADER, Register.READADDR, register & 0xFF, 0x00))
