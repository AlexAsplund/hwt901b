"""
Offline magnetometer (and accelerometer) calibration -- pure Python, no numpy.

The HWT901B has a decent *on-board* magnetic calibration (see
:meth:`hwt901b.sensor.HWT901B.calibrate_magnetic`), but it only corrects
hard-iron offsets and applies a coarse scale. When you need a proper
**hard-iron + soft-iron** correction -- e.g. the sensor is mounted near motors,
steel, or current-carrying wires -- you fit an ellipsoid to a cloud of raw
samples and derive a transform that maps that ellipsoid back onto a sphere.

Model
-----
For a raw reading ``h`` the calibrated reading is::

    h_cal = soft_iron @ (h - hard_iron)

``hard_iron`` is a 3-vector (the ellipsoid centre) and ``soft_iron`` is a 3x3
matrix that de-skews and re-scales the axes so ``|h_cal|`` is constant in a
uniform field.

The fit follows the standard algebraic ellipsoid method (Petrov / Li): solve a
linear least-squares system for the 9 quadratic coefficients, recover the centre
and shape matrix, then take the symmetric matrix square root (via a Jacobi
eigen-decomposition) to build the soft-iron matrix. All linear algebra is
implemented here in plain lists so the module has no third-party dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]


# --------------------------------------------------------------------------- #
# Tiny linear-algebra kernel (3x3 focused, plus a general LSQ solver)
# --------------------------------------------------------------------------- #


def _mat_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(mi[j] * v[j] for j in range(len(v))) for mi in m]


def _mat_mul(a: Sequence[Sequence[float]],
             b: Sequence[Sequence[float]]) -> List[List[float]]:
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)]
            for i in range(n)]


def _transpose(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*a)]


def _solve(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    """Solve ``a x = b`` by Gaussian elimination with partial pivoting."""
    n = len(a)
    # Build an augmented, mutable copy.
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-18:
            raise ValueError("singular matrix in least-squares solve")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / piv
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _jacobi_eigen(a: Sequence[Sequence[float]],
                  iterations: int = 100) -> Tuple[List[float], List[List[float]]]:
    """Symmetric eigen-decomposition via cyclic Jacobi rotations.

    Returns ``(eigenvalues, eigenvectors)`` where eigenvectors are columns.
    Accurate and dependency-free for the small 3x3 matrices we need.
    """
    n = len(a)
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    m = [list(row) for row in a]
    for _ in range(iterations):
        # Find the largest off-diagonal magnitude.
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(m[i][j]) > off:
                    off, p, q = abs(m[i][j]), i, j
        if off < 1e-15:
            break
        app, aqq, apq = m[p][p], m[q][q], m[p][q]
        phi = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            mkp, mkq = m[k][p], m[k][q]
            m[k][p] = c * mkp - s * mkq
            m[k][q] = s * mkp + c * mkq
        for k in range(n):
            mpk, mqk = m[p][k], m[q][k]
            m[p][k] = c * mpk - s * mqk
            m[q][k] = s * mpk + c * mqk
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    eigenvalues = [m[i][i] for i in range(n)]
    return eigenvalues, v


def _sqrtm_sym(a: Mat3) -> List[List[float]]:
    """Symmetric positive-definite matrix square root of a 3x3 matrix."""
    vals, vecs = _jacobi_eigen(a)
    d = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        d[i][i] = math.sqrt(max(vals[i], 0.0))
    vt = _transpose(vecs)
    return _mat_mul(_mat_mul(vecs, d), vt)


# --------------------------------------------------------------------------- #
# Calibration result
# --------------------------------------------------------------------------- #


@dataclass
class MagCalibration:
    """A fitted hard-iron/soft-iron correction.

    Apply it with :meth:`apply`. :attr:`field_strength` is the radius of the
    fitted sphere in raw sensor units -- calibrated vectors have this magnitude
    in a uniform field.
    """

    hard_iron: Vec3
    soft_iron: Mat3
    field_strength: float

    def apply(self, sample: Sequence[float]) -> Vec3:
        """Return the corrected (x, y, z) for one raw sample."""
        centred = [sample[i] - self.hard_iron[i] for i in range(3)]
        out = _mat_vec(self.soft_iron, centred)
        return (out[0], out[1], out[2])

    def residual(self, samples: Sequence[Sequence[float]]) -> float:
        """RMS deviation of ``|h_cal|`` from :attr:`field_strength`.

        A well-fitted calibration over well-distributed samples gives a small
        value relative to :attr:`field_strength`; a large value means the sample
        cloud did not cover enough orientations.
        """
        if not samples:
            return float("nan")
        acc = 0.0
        for s in samples:
            v = self.apply(s)
            mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            acc += (mag - self.field_strength) ** 2
        return math.sqrt(acc / len(samples))

    def hard_iron_int(self) -> Tuple[int, int, int]:
        """Hard-iron offset rounded to ints, for writing to HxOFFSET registers."""
        return (round(self.hard_iron[0]), round(self.hard_iron[1]),
                round(self.hard_iron[2]))

    def rotated(self, mount) -> "MagCalibration":
        """Express this calibration in a mount's **body** frame.

        The fit is done in the sensor's own axes, so if you remap the axes with a
        :class:`~hwt901b.mount.Mount` the magnetometer vector is rotated *before*
        the correction would otherwise be applied. Pass the same mount here to
        get an equivalent calibration in the body frame: with ``R`` the mount's
        ``body<-sensor`` matrix, ``hard' = R @ hard`` and
        ``soft' = R @ soft @ R.T``, so ``soft' @ (R@raw - hard') == R @ (soft @
        (raw - hard))``. A ``None`` or identity mount returns ``self``.
        """
        if mount is None or getattr(mount, "is_identity", False):
            return self
        r = mount.matrix
        rt = _transpose(r)
        hard = tuple(_mat_vec(r, self.hard_iron))
        soft = _mat_mul(_mat_mul(r, self.soft_iron), rt)
        return MagCalibration(
            hard_iron=hard,
            soft_iron=tuple(tuple(row) for row in soft),
            field_strength=self.field_strength)


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #


def fit_ellipsoid(samples: Sequence[Sequence[float]]) -> MagCalibration:
    """Fit a full hard-iron + soft-iron correction to raw magnetometer samples.

    *samples* is a sequence of ``(x, y, z)`` raw readings collected while
    rotating the sensor through as many orientations as possible (tumble it
    slowly for 20-60 s). At least ~50 well-distributed points are recommended;
    500+ gives a stable fit.

    The general quadric ``a x^2 + b y^2 + c z^2 + 2d xy + 2e xz + 2f yz
    + 2g x + 2h y + 2i z = 1`` is fitted by ordinary least squares, then
    converted to centre + shape matrix + soft-iron transform.
    """
    n = len(samples)
    if n < 10:
        raise ValueError("need at least 10 samples to fit an ellipsoid")

    # Design matrix rows and normal-equation accumulation (D^T D) v = D^T 1.
    # 9 unknowns: [a, b, c, 2d, 2e, 2f, 2g, 2h, 2i] with the '2' folded in.
    dtd = [[0.0] * 9 for _ in range(9)]
    dt1 = [0.0] * 9
    for x, y, z in ((s[0], s[1], s[2]) for s in samples):
        row = [x * x, y * y, z * z, x * y, x * z, y * z, x, y, z]
        for i in range(9):
            dt1[i] += row[i]
            ri = row[i]
            di = dtd[i]
            for j in range(9):
                di[j] += ri * row[j]

    v = _solve(dtd, dt1)
    a, b, c, dd, ee, ff, gg, hh, ii = v

    # Shape matrix A and linear term. Off-diagonals carry a factor 1/2 because
    # the fitted coefficients above already fold the '2' from the model.
    A = [
        [a, dd / 2, ee / 2],
        [dd / 2, b, ff / 2],
        [ee / 2, ff / 2, c],
    ]
    lin = [gg / 2, hh / 2, ii / 2]

    # Centre solves A * centre = -lin.
    centre = _solve(A, [-lin[0], -lin[1], -lin[2]])

    # Evaluate the constant so the quadric becomes (h-centre)^T A' (h-centre)=1.
    #   original: h^T A h + 2 lin . h = 1
    #   => (h-c)^T A (h-c) = 1 + c^T A c
    ac = _mat_vec(A, centre)
    k = 1.0 + sum(centre[i] * ac[i] for i in range(3))
    if k <= 0:
        raise ValueError("degenerate ellipsoid fit (non positive-definite)")
    A_scaled = [[A[i][j] / k for j in range(3)] for i in range(3)]

    # Semi-axes come from the eigenvalues of A_scaled (axis length = 1/sqrt(l)).
    vals, _ = _jacobi_eigen(A_scaled)
    if any(l <= 0 for l in vals):
        raise ValueError("degenerate ellipsoid fit (non positive-definite)")
    axes = [1.0 / math.sqrt(l) for l in vals]
    # Target sphere radius: geometric mean of the semi-axes keeps the corrected
    # magnitude close to the sensor's natural scale.
    radius = (axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0)

    # Soft-iron W = radius * sqrtm(A_scaled): maps the ellipsoid to |h_cal|=radius.
    root = _sqrtm_sym(A_scaled)
    soft = tuple(tuple(radius * root[i][j] for j in range(3)) for i in range(3))

    return MagCalibration(
        hard_iron=(centre[0], centre[1], centre[2]),
        soft_iron=soft,  # type: ignore[arg-type]
        field_strength=radius,
    )


def fit_hard_iron(samples: Sequence[Sequence[float]]) -> MagCalibration:
    """Cheap hard-iron-only fit from the min/max of each axis.

    Fast and robust with few samples, but does not correct soft-iron skew. Use
    :func:`fit_ellipsoid` when accuracy matters. The soft-iron matrix returned
    here is the identity.
    """
    if not samples:
        raise ValueError("no samples")
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    zs = [s[2] for s in samples]
    centre = (
        (max(xs) + min(xs)) / 2,
        (max(ys) + min(ys)) / 2,
        (max(zs) + min(zs)) / 2,
    )
    radius = (
        (max(xs) - min(xs)) + (max(ys) - min(ys)) + (max(zs) - min(zs))
    ) / 6.0
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return MagCalibration(hard_iron=centre, soft_iron=identity,
                          field_strength=radius)


def heading_from_magnetic(mag: Vec3,
                          declination_deg: float = 0.0) -> float:
    """Tilt-*un*compensated compass heading in degrees from a calibrated vector.

    Uses the **Y+ axis as the forward reference** and standard compass bearings:
    North = 0 deg, East = 90, South = 180, West = 270 (clockwise). So when the
    module's Y+ axis points north the heading reads 0; pointing south reads 180.

    Assumes the sensor is roughly level. When the sensor may be tilted, use
    :func:`tilt_compensated_heading` instead (or the module's fused yaw,
    :class:`hwt901b.protocol.Angle`).
    """
    heading = math.degrees(math.atan2(-mag[0], mag[1])) + declination_deg
    return heading % 360.0


def yaw_to_heading(yaw_deg: float, declination_deg: float = 0.0) -> float:
    """Convert the module's fused yaw (-180..180) to a compass heading [0, 360).

    The module's yaw is **counter-clockwise positive** (right-hand rule about
    +Z), whereas a compass bearing is **clockwise positive** (turning right
    increases it). So the yaw is negated here, which also makes this agree with
    :func:`heading_from_magnetic` for the same physical orientation.

    **Prefer this on dynamic platforms (boats, vehicles).** The fused yaw comes
    from the onboard gyro + accel + mag Kalman filter, which rides through wave-
    and motion-induced accelerations that would corrupt a purely accelerometer-
    based :func:`tilt_compensated_heading`. Add your local magnetic declination
    to convert a magnetic heading to a true heading.

    Note: the module's yaw zero reference is its own internal frame; verify it
    against a known bearing once installed and fold any constant offset into
    ``declination_deg``.
    """
    return (declination_deg - yaw_deg) % 360.0


def tilt_compensated_heading(accel: Vec3, mag: Vec3,
                             declination_deg: float = 0.0) -> float:
    """Tilt-compensated compass heading in degrees.

    Uses the accelerometer (gravity) to recover roll and pitch, then rotates the
    magnetometer vector back into the horizontal plane before computing the
    heading. This keeps the heading accurate even when the sensor is tilted --
    which a plain :func:`heading_from_magnetic` cannot do.

    Parameters
    ----------
    accel:
        Accelerometer reading ``(ax, ay, az)`` in any consistent unit (g or
        m/s^2); only the direction matters. Use a *static* reading -- the method
        assumes gravity is the only acceleration.
    mag:
        A **calibrated** magnetometer vector (apply your
        :class:`MagCalibration` first for best results).
    declination_deg:
        Local magnetic declination to add, converting magnetic north to true
        north.

    Returns
    -------
    float
        Heading in ``[0, 360)`` degrees using the **Y+ axis as forward** and
        standard compass bearings: North = 0, East = 90, South = 180, West = 270
        (clockwise). Matches :func:`heading_from_magnetic` when the sensor is
        level.

    Notes
    -----
    Follows the standard strapdown formulation (Freescale AN4248): roll about X,
    pitch about Y, then de-rotate the field and take ``atan2`` in the level
    frame.

    **Warning -- static assumption.** This uses the accelerometer as the gravity
    reference, so it is only correct when gravity is the *only* acceleration. On
    a boat, vehicle, or anything subject to wave/vibration/turn accelerations the
    "down" estimate is wrong and the heading swings. For those platforms use
    :func:`yaw_to_heading` on the module's gyro-fused yaw instead.
    """
    ax, ay, az = accel
    mx, my, mz = mag

    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, ay * math.sin(roll) + az * math.cos(roll))

    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)

    # Magnetometer projected onto the horizontal plane.
    mx_h = mx * cp + my * sp * sr + mz * sp * cr
    my_h = my * cr - mz * sr

    # atan2(-mx_h, my_h): Y+ forward, compass bearings (N=0, E=90, S=180,
    # W=270). Matches heading_from_magnetic when level.
    heading = math.degrees(math.atan2(-mx_h, my_h)) + declination_deg
    return heading % 360.0
