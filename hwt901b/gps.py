"""
Optional NMEA GPS parsing -- pure standard library.

The :class:`~hwt901b.stabilizer.HeadingStabilizer` is deliberately GPS-source
agnostic: it just wants a course (deg) and a speed. This module is a *convenience
only*, for callers whose GPS happens to speak NMEA 0183. If your GPS comes from
gpsd, a CAN/NMEA2000 bus, a UDP feed, a phone, or anything else, ignore this file
and pass your own ``cog``/``speed`` straight into ``stabilizer.update()``.

Parses the two sentences that carry course + speed:

* ``RMC`` -- course over ground (true) and speed over ground (knots)
* ``VTG`` -- course (true) and speed (knots and km/h)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

KNOTS_TO_MS = 0.514444


@dataclass
class GpsFix:
    """A minimal course/speed fix. ``course_deg`` is true north."""

    course_deg: Optional[float]
    speed_knots: Optional[float]
    valid: bool = True

    @property
    def speed_ms(self) -> Optional[float]:
        return None if self.speed_knots is None else self.speed_knots * KNOTS_TO_MS


def _checksum_ok(sentence: str) -> bool:
    if "*" not in sentence:
        return True  # tolerate feeds without checksums
    body, _, cs = sentence.strip().lstrip("$").partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cs[:2], 16)
    except ValueError:
        return False


def parse_nmea_gps(sentence: str) -> Optional[GpsFix]:
    """Parse an ``RMC`` or ``VTG`` sentence into a :class:`GpsFix`.

    Returns ``None`` for other sentence types or malformed input. Accepts the
    optional ``$`` prefix and any talker id (``GP``, ``GN``, ``GL`` ...).
    """
    s = sentence.strip()
    if not s or not _checksum_ok(s):
        return None
    body = s.lstrip("$").split("*")[0]
    parts = body.split(",")
    if len(parts) < 2:
        return None
    kind = parts[0][2:] if len(parts[0]) >= 5 else parts[0]

    if kind == "RMC":
        # RMC: time, status, lat, N/S, lon, E/W, speed(kn), course, date, ...
        if len(parts) < 9:
            return None
        valid = parts[2] == "A"
        speed = _f(parts[7])
        course = _f(parts[8])
        return GpsFix(course, speed, valid)

    if kind == "VTG":
        # VTG: course(true), T, course(mag), M, speed(kn), N, speed(km/h), K, ...
        if len(parts) < 6:
            return None
        course = _f(parts[1])
        speed = _f(parts[5])
        return GpsFix(course, speed, valid=True)

    return None


def _f(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None
