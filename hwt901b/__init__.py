"""
hwt901b -- a dependency-light Python driver for the WitMotion HWT901B-TTL
9/10-axis AHRS (accelerometer, gyroscope, magnetometer, barometer).

The protocol/parser/calibration core is pure standard library. ``pyserial`` is
an optional extra pulled in only when you open a live serial link
(``pip install hwt901b[serial]``). Works identically on Windows (``COM3``) and
Linux (``/dev/ttyUSB0``).

Quick start
-----------
>>> from hwt901b import HWT901B                       # doctest: +SKIP
>>> with HWT901B.open("COM3", baudrate=9600) as imu:  # doctest: +SKIP
...     s = imu.read_state()
...     print(s.angle, s.acceleration, s.magnetic)
"""

from __future__ import annotations

from . import calibration, mount, nmea, profiler, protocol, stabilizer, transport
from .mount import Mount
from .profiler import WaveProfile, WaveProfiler
from .stabilizer import (
    HeadingSmoother,
    HeadingStabilizer,
    StabilizedHeading,
    magnetic_dip_deg,
)
from .calibration import (
    MagCalibration,
    fit_ellipsoid,
    fit_hard_iron,
    heading_from_magnetic,
    tilt_compensated_heading,
    yaw_to_heading,
)
from .protocol import (
    Acceleration,
    Angle,
    AngularVelocity,
    BaudRate,
    Bandwidth,
    CalibrationMode,
    Magnetic,
    OutputRate,
    Pressure,
    Quaternion,
    Register,
    RswBit,
    Time,
)
from .sensor import HWT901B, State
from .gps import GpsFix, parse_nmea_gps
from .transport import BytesTransport, SerialTransport, Transport

__version__ = "0.1.0"

__all__ = [
    "HWT901B",
    "State",
    # transports
    "Transport",
    "SerialTransport",
    "BytesTransport",
    # calibration
    "MagCalibration",
    "fit_ellipsoid",
    "fit_hard_iron",
    "heading_from_magnetic",
    "tilt_compensated_heading",
    "yaw_to_heading",
    # protocol payloads & enums
    "Acceleration",
    "AngularVelocity",
    "Angle",
    "Magnetic",
    "Quaternion",
    "Pressure",
    "Time",
    "Register",
    "RswBit",
    "OutputRate",
    "BaudRate",
    "Bandwidth",
    "CalibrationMode",
    # submodules
    "protocol",
    "transport",
    "calibration",
    "nmea",
    "stabilizer",
    "gps",
    "mount",
    # mounting orientation
    "Mount",
    # stabilization
    "HeadingStabilizer",
    "StabilizedHeading",
    "HeadingSmoother",
    "magnetic_dip_deg",
    "GpsFix",
    "parse_nmea_gps",
    "profiler",
    "WaveProfiler",
    "WaveProfile",
]
