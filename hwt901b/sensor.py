"""
High-level driver for the HWT901B-TTL.

:class:`HWT901B` wraps a :class:`~hwt901b.transport.Transport`, decodes the
stream into a rolling :class:`State`, and exposes the configuration and
calibration commands from the WIT protocol behind readable method names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator, Optional

from . import protocol as P
from .transport import SerialTransport, Transport

# How long the module leaves its registers unlocked after the unlock command.
_UNLOCK_WINDOW_S = 10.0
# Recommended settling time between consecutive register writes.
_WRITE_GAP_S = 0.05


@dataclass
class State:
    """The latest decoded value of every sample type, updated in place.

    Fields are ``None`` until the corresponding packet type has been seen. The
    HWT901B emits acceleration, angular velocity and angle by default; magnetic,
    quaternion and pressure require enabling the matching :class:`RswBit`.
    """

    acceleration: Optional[P.Acceleration] = None
    angular_velocity: Optional[P.AngularVelocity] = None
    angle: Optional[P.Angle] = None
    magnetic: Optional[P.Magnetic] = None
    quaternion: Optional[P.Quaternion] = None
    pressure: Optional[P.Pressure] = None
    time: Optional[P.Time] = None
    last_register_read: Optional[P.RegisterRead] = None
    updated_at: float = 0.0

    def apply(self, decoded) -> None:
        if isinstance(decoded, P.Acceleration):
            self.acceleration = decoded
        elif isinstance(decoded, P.AngularVelocity):
            self.angular_velocity = decoded
        elif isinstance(decoded, P.Angle):
            self.angle = decoded
        elif isinstance(decoded, P.Magnetic):
            self.magnetic = decoded
        elif isinstance(decoded, P.Quaternion):
            self.quaternion = decoded
        elif isinstance(decoded, P.Pressure):
            self.pressure = decoded
        elif isinstance(decoded, P.Time):
            self.time = decoded
        elif isinstance(decoded, P.RegisterRead):
            self.last_register_read = decoded

    def copy(self) -> "State":
        return replace(self)


class HWT901B:
    """Driver for a single HWT901B-TTL module.

    Example
    -------
    >>> with HWT901B.open("COM3", baudrate=9600) as imu:      # doctest: +SKIP
    ...     for state in imu.stream():
    ...         print(state.angle)
    """

    def __init__(self, transport: Transport, mount=None) -> None:
        self._t = transport
        self._parser = P.FrameParser()
        self.state = State()
        self._monotonic = time.monotonic
        # Optional software mounting remap (see hwt901b.mount.Mount). Applied to
        # every decoded payload so state is always in the body frame. None or an
        # identity mount means "sensor lies flat" and adds no work.
        self.mount = mount if (mount is not None and not mount.is_identity) else None

    # ---- construction ---------------------------------------------------- #

    @classmethod
    def open(
        cls,
        port: str,
        baudrate: int = 9600,
        timeout: float = 0.1,
        mount=None,
    ) -> "HWT901B":
        """Open a serial port and return a driver bound to it.

        Works with Windows names (``"COM3"``) and POSIX device paths
        (``"/dev/ttyUSB0"``) alike. Pass a :class:`~hwt901b.mount.Mount` to
        remap the axes in software when the module is not mounted flat.
        """
        return cls(SerialTransport(port, baudrate=baudrate, timeout=timeout),
                   mount=mount)

    def __enter__(self) -> "HWT901B":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._t.close()

    # ---- reading --------------------------------------------------------- #

    def poll(self, size: int = 256) -> int:
        """Read whatever bytes are available and fold them into :attr:`state`.

        Returns the number of complete frames decoded in this call.
        """
        data = self._t.read(size)
        if not data:
            return 0
        count = 0
        for raw in self._parser.feed(data):
            decoded = P.decode(raw)
            if decoded is not None:
                if self.mount is not None:
                    decoded = self.mount.apply(decoded)
                self.state.apply(decoded)
                count += 1
        if count:
            self.state.updated_at = self._monotonic()
        return count

    def read_state(self, timeout: float = 1.0) -> State:
        """Block until at least one frame arrives, then return a state snapshot.

        Raises :class:`TimeoutError` if nothing decodes within *timeout*.
        """
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            if self.poll():
                return self.state.copy()
        raise TimeoutError("no valid frame received within timeout")

    def read_true_heading(self, declination_deg: float = 0.0,
                          timeout: float = 1.0) -> float:
        """Return the gyro-fused compass heading in ``[0, 360)`` degrees.

        Uses the module's onboard sensor fusion (:class:`Angle` yaw), which is
        the right choice on **dynamic platforms like boats** -- the gyro carries
        the heading through wave and turn accelerations that would corrupt an
        accelerometer-only :func:`~hwt901b.calibration.tilt_compensated_heading`.
        Add your local declination for true (vs magnetic) north.
        """
        from .calibration import yaw_to_heading
        state = self.read_state(timeout)
        if state.angle is None:
            raise TimeoutError("no fused angle packet received")
        return yaw_to_heading(state.angle.yaw, declination_deg)

    def frames(self, size: int = 256) -> Iterator[P.RawFrame]:
        """Yield raw verified frames as they arrive (low-level access)."""
        while True:
            data = self._t.read(size)
            if not data:
                yield from ()
                continue
            yield from self._parser.feed(data)

    def stream(
        self,
        min_interval: float = 0.0,
        size: int = 256,
    ) -> Iterator[State]:
        """Yield a fresh :class:`State` snapshot after each batch of frames.

        *min_interval* throttles emission to at most one snapshot per that many
        seconds, which is convenient for console displays at high output rates.
        """
        last = 0.0
        while True:
            if self.poll(size):
                now = self._monotonic()
                if now - last >= min_interval:
                    last = now
                    yield self.state.copy()

    # ---- raw command plumbing ------------------------------------------- #

    def _send(self, data: bytes) -> None:
        self._t.write(data)

    def unlock(self) -> None:
        """Send the unlock preamble required before any register write."""
        self._send(P.unlock_command())
        time.sleep(_WRITE_GAP_S)

    def save(self) -> None:
        """Persist the current configuration to flash."""
        self._send(P.save_command())
        time.sleep(_WRITE_GAP_S)

    def write_register(
        self,
        register: int,
        value: int,
        unlock: bool = True,
        save: bool = False,
    ) -> None:
        """Write a single configuration register.

        By default each call is wrapped in the unlock preamble; set
        ``unlock=False`` when batching writes you have already unlocked for. Set
        ``save=True`` to persist immediately.
        """
        if unlock:
            self.unlock()
        self._send(P.write_command(register, value))
        time.sleep(_WRITE_GAP_S)
        if save:
            self.save()

    def read_register(self, register: int, timeout: float = 1.0) -> tuple:
        """Read four consecutive register words starting at *register*.

        Returns the four signed int16 values from the 0x5F response.
        """
        self._send(P.read_command(register))
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            self.poll()
            rr = self.state.last_register_read
            if rr is not None:
                self.state.last_register_read = None
                return rr.values
        raise TimeoutError(f"register 0x{register:02X} read timed out")

    # ---- configuration --------------------------------------------------- #

    def set_output_rate(self, rate: P.OutputRate, save: bool = True) -> None:
        """Set the packet output rate (see :class:`~hwt901b.protocol.OutputRate`)."""
        self.write_register(P.Register.RRATE, int(rate), save=save)

    def set_outputs(self, *bits: P.RswBit, save: bool = True) -> None:
        """Choose which packet types the module emits.

        >>> imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGLE, RswBit.MAGNETIC)
        """
        self.write_register(P.Register.RSW, P.rsw_mask(*bits), save=save)

    def set_bandwidth(self, bw: P.Bandwidth, save: bool = True) -> None:
        """Set the sensor low-pass filter cutoff."""
        self.write_register(P.Register.BANDWIDTH, int(bw), save=save)

    def set_algorithm(self, six_axis: bool, save: bool = True) -> None:
        """Select the fusion algorithm.

        ``six_axis=False`` uses the magnetometer for an absolute (compass) yaw;
        ``True`` uses a 6-axis relative yaw that will not be pulled by nearby
        magnetic disturbances but drifts slowly over time.
        """
        self.write_register(P.Register.AXIS6, 1 if six_axis else 0, save=save)

    def set_orientation_vertical(self, vertical: bool, save: bool = True) -> None:
        """Tell the *on-chip* fusion whether the module is mounted vertically.

        This is WitMotion's own install-direction toggle (register
        :attr:`~hwt901b.protocol.Register.ORIENT`). It has exactly one non-flat
        setting, so it only fits that single fixed vertical orientation, but its
        advantage is that the module fuses the heading around true vertical --
        keeping the gyro-backed yaw meaningful when mounted on edge.

        For any *other* mounting attitude, or to choose exactly which axis maps
        where (e.g. Z+ -> Y+), use the software :class:`~hwt901b.mount.Mount`
        instead: ``HWT901B.open(port, mount=Mount.z_up_to_y())``.
        """
        self.write_register(P.Register.ORIENT, 1 if vertical else 0, save=save)

    def set_baudrate(self, baud: P.BaudRate, save: bool = True) -> None:
        """Change the module's UART baud rate.

        The module answers the write at the *old* rate, then switches. If the
        underlying transport is a :class:`SerialTransport`, the host side is
        reconfigured to match automatically.
        """
        self.write_register(P.Register.BAUD, int(baud), save=save)
        time.sleep(_WRITE_GAP_S)
        if isinstance(self._t, SerialTransport):
            self._t.baudrate = baud.bps
            self._t.reset_input()

    def sleep(self, enabled: bool) -> None:
        """Put the module to sleep (wakes on the next serial byte)."""
        self.write_register(P.Register.SLEEP, 1 if enabled else 0, save=False)

    def reboot(self) -> None:
        """Reboot the module (does not erase saved settings)."""
        self.write_register(P.Register.SAVE, 0x00FF, save=False)

    def factory_reset(self) -> None:
        """Restore factory defaults, then save."""
        self.unlock()
        self.write_register(P.Register.SAVE, 0x0001, unlock=False)
        self.save()

    # ---- on-board calibration ------------------------------------------- #

    def calibrate_acceleration(self, settle: float = 5.5) -> None:
        """Run the built-in accelerometer (gravity) zero calibration.

        Keep the module **still and level** for the whole call. This nulls the
        accelerometer bias and the resting roll/pitch. Takes >5 s per WitMotion.
        """
        self.write_register(P.Register.CALSW, P.CalibrationMode.ACCELERATION)
        time.sleep(settle)
        self.write_register(P.Register.CALSW, P.CalibrationMode.NONE, save=True)

    def begin_magnetic_calibration(self) -> None:
        """Enter magnetic-field calibration mode.

        After calling this, rotate the module slowly through all orientations
        (ideally a full turn about each axis) for ~10-20 s, then call
        :meth:`end_magnetic_calibration`.
        """
        self.write_register(P.Register.CALSW, P.CalibrationMode.MAGNETIC)

    def end_magnetic_calibration(self) -> None:
        """Leave magnetic calibration mode and save the fitted offsets."""
        self.write_register(P.Register.CALSW, P.CalibrationMode.NONE, save=True)

    def calibrate_magnetic(self, rotate_seconds: float = 15.0,
                           progress: Optional[Callable[[float], None]] = None) -> None:
        """Convenience wrapper: begin, wait while you rotate, end.

        *progress* (if given) is called with a 0..1 fraction during the wait.
        """
        self.begin_magnetic_calibration()
        start = self._monotonic()
        while True:
            elapsed = self._monotonic() - start
            if elapsed >= rotate_seconds:
                break
            if progress:
                progress(min(elapsed / rotate_seconds, 1.0))
            self.poll()  # keep draining the stream so buffers don't overflow
            time.sleep(0.05)
        if progress:
            progress(1.0)
        self.end_magnetic_calibration()

    def compass_swing(self, seconds: float = 90.0,
                      progress: Optional[Callable[[float], None]] = None) -> None:
        """Marine 'compass swing' -- magnetometer calibration in the boat.

        Run this with the sensor **already mounted in its final position**, then
        slowly turn the *whole boat* through at least one full 360 deg circle
        (two slow turns is better) before the timer ends. This captures the
        boat's own hard/soft-iron (deviation) from the engine, hull, batteries
        and wiring -- a bench calibration cannot, because that field only exists
        in the installation. Uses the module's on-board fit and saves it.

        *seconds* should comfortably exceed the time for your turns; *progress*
        (if given) receives a 0..1 fraction.
        """
        self.calibrate_magnetic(rotate_seconds=seconds, progress=progress)

    def configure_marine(
        self,
        rate: "P.OutputRate" = P.OutputRate.HZ_10,
        bandwidth: "P.Bandwidth" = P.Bandwidth.HZ_20,
    ) -> None:
        """Apply a sensible baseline configuration for a boat and save it.

        - **9-axis** algorithm: absolute (magnetic) heading, so yaw does not
          drift -- essential for a compass.
        - a moderate low-pass **bandwidth** to reject engine vibration.
        - a modest **output rate** suitable for steering/logging.

        This does *not* calibrate the magnetometer -- run :meth:`compass_swing`
        after mounting. Enables accel/gyro/angle/magnetic output.
        """
        self.set_algorithm(six_axis=False, save=False)     # absolute heading
        self.set_bandwidth(bandwidth, save=False)
        self.set_output_rate(rate, save=False)
        self.set_outputs(P.RswBit.ACCELERATION, P.RswBit.ANGULAR_VELOCITY,
                        P.RswBit.ANGLE, P.RswBit.MAGNETIC, save=True)

    def reset_heading(self) -> None:
        """Zero the yaw at the current heading (6-axis / relative mode)."""
        self.write_register(P.Register.CALSW, P.CalibrationMode.HEADING_RESET,
                            save=True)

    def reset_height(self) -> None:
        """Zero the barometric altitude at the current pressure."""
        self.write_register(P.Register.CALSW, P.CalibrationMode.HEIGHT_RESET,
                            save=True)

    def set_angle_reference(self) -> None:
        """Store the current attitude as the zero reference."""
        self.write_register(P.Register.CALSW, P.CalibrationMode.ANGLE_REFERENCE,
                            save=True)

    # ---- manual offset registers ---------------------------------------- #

    def set_magnetic_offsets(self, x: int, y: int, z: int,
                             save: bool = True) -> None:
        """Write hard-iron offsets directly (e.g. from an offline fit).

        Values are the raw integer biases to subtract, matching the units the
        module reports magnetic field in.
        """
        self.unlock()
        self.write_register(P.Register.HXOFFSET, x, unlock=False)
        self.write_register(P.Register.HYOFFSET, y, unlock=False)
        self.write_register(P.Register.HZOFFSET, z, unlock=False)
        if save:
            self.save()
