"""
Mounting-orientation remap -- pure standard library.

The HWT901B assumes it lies **flat**: sensor X forward, Y left, Z up. If you
bolt it in some other attitude (on its edge, upside down, rotated 90 deg) every
reading -- acceleration, angular velocity, magnetic field, and the fused
roll/pitch/yaw -- comes out in the *sensor's* frame, not the frame your boat
cares about.

WitMotion's own SDK exposes a single on-chip toggle for this
(:meth:`~hwt901b.sensor.HWT901B.set_orientation_vertical`, register
:attr:`~hwt901b.protocol.Register.ORIENT`) but it only offers one fixed
"vertical" remap. :class:`Mount` here is a software equivalent that handles *any*
axis-aligned mounting and lets you say exactly which sensor axis should become
which body axis -- e.g. "the axis that used to point up (Z+) is now my left
(Y+)".

How it works
------------
A mount is a signed axis permutation -- a proper rotation matrix ``R`` with
entries in ``{-1, 0, 1}`` such that::

    v_body = R @ v_sensor

Vectors (accel/gyro/mag) are just multiplied through. The fused Euler angles are
handled by rebuilding the sensor's earth<-sensor orientation matrix from
roll/pitch/yaw, composing with the mount, and re-extracting the angles, so the
on-chip gyro fusion is preserved rather than thrown away.

Convention: right-handed frames, aerospace Z-Y-X Euler order (yaw about Z, pitch
about Y, roll about X), yaw positive counter-clockwise -- matching how the module
itself reports angles. With ``Mount.identity()`` everything is a no-op, so
existing flat installations are unaffected.

Example
-------
>>> from hwt901b import Mount
>>> m = Mount.z_up_to_y()            # stood on edge: former up (Z+) -> left (Y+)
>>> m.apply_vector((0.0, -1.0, 0.0)) # a vertically-mounted sensor at rest ...
(0.0, 0.0, 1.0)                      # ... reads level once remapped
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence, Tuple

from . import protocol as P

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]

_AXES = {
    "x": (1.0, 0.0, 0.0), "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def _matvec(m: Mat3, v: Sequence[float]) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _matmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _transpose(m: Mat3) -> Mat3:
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _det(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> Mat3:
    """earth<-sensor rotation from Z-Y-X Euler angles (degrees)."""
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    # Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp,     cp * sr,                cp * cr),
    )


def _matrix_to_euler(m: Mat3) -> Vec3:
    """Extract (roll, pitch, yaw) in degrees from a Z-Y-X rotation matrix."""
    sp = -m[2][0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(sp) < 0.999999:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:  # gimbal lock: roll and yaw are coupled; pin roll to 0
        roll = 0.0
        yaw = math.atan2(-m[0][1], m[1][1])
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


class Mount:
    """A fixed mounting orientation expressed as a signed axis remap.

    Build one with a preset (:meth:`identity`, :meth:`z_up_to_y`,
    :meth:`vertical`, ...) or the general :meth:`from_axes`, then apply it to
    decoded payloads or a whole :class:`~hwt901b.sensor.State`. Pass it to
    :meth:`HWT901B.open(..., mount=...) <hwt901b.sensor.HWT901B.open>` to have
    every reading transformed automatically.
    """

    def __init__(self, matrix: Mat3, name: str = "custom") -> None:
        d = _det(matrix)
        if abs(d - 1.0) > 1e-6:
            raise ValueError(
                f"mount matrix must be a proper rotation (det=+1), got det={d:.3f};"
                " check the axis assignment is right-handed")
        self.matrix: Mat3 = tuple(tuple(float(x) for x in row) for row in matrix)  # type: ignore[assignment]
        self.name = name

    # ---- constructors ---------------------------------------------------- #

    @classmethod
    def identity(cls) -> "Mount":
        """Sensor lies flat as designed (X fwd, Y left, Z up) -- a no-op."""
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)), name="level")

    @classmethod
    def parse(cls, spec: str):
        """Parse a ``--mount`` style string into a Mount, or ``None`` if flat.

        Accepts the presets (``level``/``flat``/``none`` -> ``None``,
        ``vertical``, ``z-up-to-y``, ``upside-down``, ``yaw-90-ccw``) or a custom
        ``"X:Y:Z"`` axis map such as ``"x:z:-y"``. Raises :class:`ValueError` on
        anything else. Returning ``None`` for the flat default lets callers skip
        the remap entirely.
        """
        s = (spec or "level").strip().lower().replace("_", "-")
        presets = {
            "level": None, "flat": None, "none": None,
            "vertical": cls.vertical, "z-up-to-y": cls.z_up_to_y,
            "upside-down": cls.upside_down, "yaw-90-ccw": cls.yaw_90_ccw,
        }
        if s in presets:
            maker = presets[s]
            return maker() if maker else None
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 3:
                raise ValueError("custom mount must be 'X:Y:Z', e.g. x:z:-y")
            return cls.from_axes(*parts)
        raise ValueError(f"unknown mount: {spec!r}")

    @classmethod
    def from_axes(cls, x: str, y: str, z: str, name: str = "custom") -> "Mount":
        """Define the mount by which sensor axis feeds each **body** axis.

        Each argument names the sensor axis (optionally signed) that should
        become that body axis. For example the identity is
        ``from_axes(x="x", y="y", z="z")``; a sensor stood on edge so its old
        up-axis is now the body's left is ``from_axes(x="x", y="z", z="-y")``.
        """
        try:
            rows = (_AXES[x.lower()], _AXES[y.lower()], _AXES[z.lower()])
        except KeyError as exc:
            raise ValueError(
                f"axis must be one of +/-x, +/-y, +/-z; got {exc.args[0]!r}") from exc
        return cls(rows, name=name)  # rows ARE R: body row i = sensor axis

    @classmethod
    def z_up_to_y(cls) -> "Mount":
        """Stand the module on edge so its Z+ (was up) points to body Y+ (left).

        Keeps the X (forward/bow) axis unchanged; the former up-axis becomes
        left, and the old -Y becomes up. This is the "mount it upright instead
        of laying down" case.
        """
        return cls.from_axes(x="x", y="z", z="-y", name="z_up_to_y")

    @classmethod
    def vertical(cls) -> "Mount":
        """Alias for :meth:`z_up_to_y` -- the common vertical (on-edge) mount."""
        m = cls.z_up_to_y()
        m.name = "vertical"
        return m

    @classmethod
    def upside_down(cls) -> "Mount":
        """Module mounted inverted (roll 180): X fwd kept, Y and Z flipped."""
        return cls.from_axes(x="x", y="-y", z="-z", name="upside_down")

    @classmethod
    def yaw_90_ccw(cls) -> "Mount":
        """Rotated 90 deg CCW about the vertical (old +X now points to +Y)."""
        return cls.from_axes(x="-y", y="x", z="z", name="yaw_90_ccw")

    # ---- properties ------------------------------------------------------ #

    @property
    def is_identity(self) -> bool:
        ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        return all(self.matrix[i][j] == ident[i][j]
                   for i in range(3) for j in range(3))

    def __repr__(self) -> str:
        return f"Mount({self.name!r})"

    # ---- application ----------------------------------------------------- #

    def apply_vector(self, v: Sequence[float]) -> Vec3:
        """Remap a raw 3-vector from the sensor frame to the body frame."""
        return _matvec(self.matrix, v)

    def apply_acceleration(self, a: "P.Acceleration") -> "P.Acceleration":
        x, y, z = _matvec(self.matrix, (a.x, a.y, a.z))
        return P.Acceleration(x, y, z, a.temperature)

    def apply_angular_velocity(self, g: "P.AngularVelocity") -> "P.AngularVelocity":
        x, y, z = _matvec(self.matrix, (g.x, g.y, g.z))
        return P.AngularVelocity(x, y, z, g.voltage)

    def apply_magnetic(self, m: "P.Magnetic") -> "P.Magnetic":
        x, y, z = _matvec(self.matrix, (m.x, m.y, m.z))
        return P.Magnetic(x, y, z, m.temperature)

    def apply_angle(self, ang: "P.Angle") -> "P.Angle":
        """Re-express fused roll/pitch/yaw in the body frame.

        Rebuilds the earth<-sensor rotation from the reported Euler angles,
        composes it with the mount (``R_eb = R_es @ R.T``), and re-extracts the
        angles -- so the module's gyro-backed fusion is preserved. Yaw stays in
        the module's counter-clockwise convention; convert to a compass heading
        with :func:`~hwt901b.calibration.yaw_to_heading` exactly as before.
        """
        r_es = _euler_to_matrix(ang.roll, ang.pitch, ang.yaw)
        r_eb = _matmul(r_es, _transpose(self.matrix))
        roll, pitch, yaw = _matrix_to_euler(r_eb)
        return P.Angle(roll, pitch, yaw, ang.version)

    def apply(self, decoded):
        """Transform any decoded payload; unknown types pass through unchanged."""
        if isinstance(decoded, P.Acceleration):
            return self.apply_acceleration(decoded)
        if isinstance(decoded, P.AngularVelocity):
            return self.apply_angular_velocity(decoded)
        if isinstance(decoded, P.Magnetic):
            return self.apply_magnetic(decoded)
        if isinstance(decoded, P.Angle):
            return self.apply_angle(decoded)
        return decoded

    def apply_state(self, state):
        """Return a copy of *state* with every framed quantity in the body frame."""
        out = state.copy()
        if out.acceleration is not None:
            out.acceleration = self.apply_acceleration(out.acceleration)
        if out.angular_velocity is not None:
            out.angular_velocity = self.apply_angular_velocity(out.angular_velocity)
        if out.magnetic is not None:
            out.magnetic = self.apply_magnetic(out.magnetic)
        if out.angle is not None:
            out.angle = self.apply_angle(out.angle)
        return out
