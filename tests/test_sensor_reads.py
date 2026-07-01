"""Regression tests for HWT901B read paths (heading deadline loop, stale 0x5F)."""

import struct

import pytest

from hwt901b import HWT901B
from hwt901b import protocol as P


def make_frame(ptype: int, *int16_values: int) -> bytes:
    """Build a valid 11-byte frame from up to four int16 values."""
    payload = struct.pack("<4h", *int16_values)
    body = bytes((P.FRAME_HEADER, ptype)) + payload
    return body + bytes((P.checksum(body + b"\x00"),))


class ChunkedTransport:
    """Fake transport returning one predefined chunk per read() call."""

    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)
        self.written = bytearray()

    def read(self, size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def close(self) -> None:
        pass


class RespondAfterWriteTransport:
    """Fake transport that only serves *response* after something was written."""

    def __init__(self, response: bytes) -> None:
        self._response = response
        self._sent = False
        self.written = bytearray()

    def read(self, size: int) -> bytes:
        if self.written and not self._sent:
            self._sent = True
            return self._response
        return b""

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# read_true_heading keeps polling until an Angle frame arrives
# --------------------------------------------------------------------------- #


def test_read_true_heading_waits_past_non_angle_frames():
    # First read yields only an acceleration frame; the angle frame arrives on
    # the second read. read_true_heading must keep polling instead of raising.
    accel = make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
    angle = make_frame(P.PacketType.ANGLE, 0, 0, 16384, 1)  # yaw = +90 deg
    imu = HWT901B(ChunkedTransport([accel, angle]))
    heading = imu.read_true_heading(timeout=1.0)
    # Module yaw +90 (CCW-positive) -> compass heading 270.
    assert heading == pytest.approx(270.0)


def test_read_true_heading_applies_declination():
    accel = make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
    angle = make_frame(P.PacketType.ANGLE, 0, 0, 16384, 1)  # yaw = +90 deg
    imu = HWT901B(ChunkedTransport([accel, angle]))
    heading = imu.read_true_heading(declination_deg=5.0, timeout=1.0)
    assert heading == pytest.approx(275.0)


def test_read_true_heading_times_out_without_angle():
    accel = make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
    imu = HWT901B(ChunkedTransport([accel]))
    with pytest.raises(TimeoutError):
        imu.read_true_heading(timeout=0.05)


# --------------------------------------------------------------------------- #
# read_register discards a stale, unconsumed 0x5F response
# --------------------------------------------------------------------------- #


def test_read_register_returns_fresh_response_not_stale():
    fresh = make_frame(P.PacketType.READ_REGISTER, 11, 22, 33, 44)
    imu = HWT901B(RespondAfterWriteTransport(fresh))
    # A leftover response from an earlier read is still sitting in state.
    imu.state.last_register_read = P.RegisterRead(start=-1, values=(9, 9, 9, 9))
    values = imu.read_register(P.Register.VERSION, timeout=1.0)
    assert values == (11, 22, 33, 44)


def test_read_register_times_out_instead_of_returning_stale():
    imu = HWT901B(ChunkedTransport([]))  # transport never answers
    imu.state.last_register_read = P.RegisterRead(start=-1, values=(9, 9, 9, 9))
    with pytest.raises(TimeoutError):
        imu.read_register(P.Register.VERSION, timeout=0.05)
