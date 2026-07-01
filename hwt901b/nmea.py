"""
NMEA 0183 sentence output -- pure standard library.

Turns the module's fused attitude into the heading/attitude sentences a
chartplotter, autopilot or navigation app (OpenCPN, SignalK, ...) understands:

    HDT  heading, true          $--HDT,x.x,T*hh
    HDM  heading, magnetic      $--HDM,x.x,M*hh
    HDG  heading + dev + var     $--HDG,x.x,d,a,v,a*hh
    ROT  rate of turn            $--ROT,x.x,A*hh
    XDR  transducer (roll/pitch) $YXXDR,A,p,D,PTCH,A,r,D,ROLL*hh

All headings come from the module's **gyro-fused yaw** (the motion-robust
source, see :func:`hwt901b.calibration.yaw_to_heading`), so this is suitable for
a boat. The magnetometer measures magnetic north; supply the local **variation**
(declination, East positive) to also emit a true heading.

Checksum is the XOR of every character between ``$`` and ``*``, as two uppercase
hex digits; each sentence ends with CR LF.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def checksum(body: str) -> str:
    """Return the two-hex-digit NMEA checksum for *body* (text between $ and *)."""
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def sentence(body: str) -> str:
    """Wrap *body* (without the leading ``$``) into a full ``$...*hh\\r\\n``."""
    return f"${body}*{checksum(body)}\r\n"


def _fmt(value: Optional[float], prec: int = 1) -> str:
    return "" if value is None else f"{value:.{prec}f}"


def _dir(value: Optional[float], pos: str, neg: str) -> str:
    if value is None:
        return ""
    return pos if value >= 0 else neg


def hdt(heading_true_deg: float, talker: str = "HC") -> str:
    """Heading, True."""
    return sentence(f"{talker}HDT,{_fmt(heading_true_deg)},T")


def hdm(heading_mag_deg: float, talker: str = "HC") -> str:
    """Heading, Magnetic."""
    return sentence(f"{talker}HDM,{_fmt(heading_mag_deg)},M")


def hdg(heading_mag_deg: float, deviation_deg: Optional[float] = None,
        variation_deg: Optional[float] = None, talker: str = "HC") -> str:
    """Heading, Deviation & Variation (East positive for both)."""
    body = (
        f"{talker}HDG,{_fmt(heading_mag_deg)},"
        f"{_fmt(abs(deviation_deg)) if deviation_deg is not None else ''},"
        f"{_dir(deviation_deg, 'E', 'W')},"
        f"{_fmt(abs(variation_deg)) if variation_deg is not None else ''},"
        f"{_dir(variation_deg, 'E', 'W')}"
    )
    return sentence(body)


def rot(rate_deg_per_min: float, valid: bool = True, talker: str = "TI") -> str:
    """Rate Of Turn (deg/min; negative = bow to port)."""
    return sentence(f"{talker}ROT,{_fmt(rate_deg_per_min)},{'A' if valid else 'V'}")


def xdr_attitude(roll_deg: Optional[float] = None,
                 pitch_deg: Optional[float] = None, talker: str = "YX") -> str:
    """Transducer sentence carrying pitch and/or roll as angular degrees.

    Sign follows the module's axes; negate at the source if your mounting needs
    the marine convention (pitch + = bow up, roll + = starboard down).
    """
    parts: List[str] = [f"{talker}XDR"]
    if pitch_deg is not None:
        parts.append(f"A,{pitch_deg:.1f},D,PTCH")
    if roll_deg is not None:
        parts.append(f"A,{roll_deg:.1f},D,ROLL")
    return sentence(",".join(parts))


# Which sentence families to emit by default.
DEFAULT_INCLUDE = ("hdt", "hdm", "rot", "xdr")


def sentences_from_heading(
    heading_true_deg: float,
    *,
    variation_deg: float = 0.0,
    roll_deg: Optional[float] = None,
    pitch_deg: Optional[float] = None,
    rate_of_turn_dpm: Optional[float] = None,
    include: Sequence[str] = DEFAULT_INCLUDE,
    heading_talker: str = "HC",
) -> List[str]:
    """Build sentences from an already-computed **true** heading.

    Use this to emit a *stabilized* heading (e.g. from
    :class:`hwt901b.stabilizer.HeadingStabilizer`, which already applies
    declination): pass its ``heading`` as *heading_true_deg*. The magnetic
    ``HDM`` value is derived by subtracting *variation_deg*. ``rate_of_turn_dpm``
    is in degrees per minute; roll/pitch feed the ``XDR`` sentence.
    """
    out: List[str] = []
    true = heading_true_deg % 360.0
    mag = (heading_true_deg - variation_deg) % 360.0
    if "hdt" in include:
        out.append(hdt(true, heading_talker))
    if "hdm" in include:
        out.append(hdm(mag, heading_talker))
    if "hdg" in include:
        out.append(hdg(mag, None, variation_deg, heading_talker))
    if "xdr" in include and (roll_deg is not None or pitch_deg is not None):
        out.append(xdr_attitude(roll_deg, pitch_deg))
    if "rot" in include and rate_of_turn_dpm is not None:
        out.append(rot(rate_of_turn_dpm))
    return out


def sentences_from_state(
    state,
    variation_deg: float = 0.0,
    include: Sequence[str] = DEFAULT_INCLUDE,
    heading_talker: str = "HC",
) -> List[str]:
    """Build NMEA sentences from a :class:`hwt901b.sensor.State` snapshot.

    Heading uses the fused yaw. *variation_deg* (declination, East positive)
    converts magnetic to true. *include* selects families from
    ``{"hdt", "hdm", "hdg", "rot", "xdr"}``. Missing data is simply skipped.
    """
    out: List[str] = []
    if state.angle is not None:
        # Module yaw is CCW-positive; compass heading is CW-positive -> negate
        # (matches yaw_to_heading).
        mag = (-state.angle.yaw) % 360.0
        true = (variation_deg - state.angle.yaw) % 360.0
        if "hdt" in include:
            out.append(hdt(true, heading_talker))
        if "hdm" in include:
            out.append(hdm(mag, heading_talker))
        if "hdg" in include:
            out.append(hdg(mag, None, variation_deg, heading_talker))
        if "xdr" in include:
            out.append(xdr_attitude(state.angle.roll, state.angle.pitch))
    if "rot" in include and state.angular_velocity is not None:
        # Gyro Z (deg/s) -> deg/min. Yaw rate is the turn rate.
        out.append(rot(state.angular_velocity.z * 60.0))
    return out
