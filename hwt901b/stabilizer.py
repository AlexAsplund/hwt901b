"""
Downstream heading stabilizer -- pure standard library.

The HWT901B already runs a gyro/accel/mag Kalman fusion internally, so its
``Angle.yaw`` is a solid heading. This module adds the *downstream* tricks that
scientific AHRS work and marine autopilots layer on top, which the sensor cannot
do on its own because it has no notion of a magnetic disturbance or of the
vehicle's velocity:

1. **Magnetic disturbance gating (soft).** Watch the calibrated field magnitude
   and the magnetic dip (inclination) angle. When either strays from its
   reference the magnetometer is being distorted (switched DC loads, passing
   steel, etc.), so we *down-weight* the mag-based yaw and coast on the gyro
   rate-of-turn instead -- graduated down-weighting, not a hard cut-off, as the
   recent literature recommends (e.g. AMO-HEAD, arXiv:2510.10979).

2. **GPS course-over-ground (COG) aiding.** When the vehicle is moving, its
   velocity vector is a magnetically-independent heading reference. We slew the
   heading slowly toward COG (long time constant), which removes slow compass
   bias the way an autopilot blends compass with GPS course. The GPS data is
   supplied by the caller from *any* source (NMEA, gpsd, a UDP feed, another
   sensor) -- see :meth:`HeadingStabilizer.update`.

3. **Smoothing / rate limiting.** Optional low-pass and slew-rate cap to stop
   needle jitter.

Everything is a complementary filter over the fused yaw: gyro carries the fast
dynamics, magnetometer and COG make slow corrections that are gated by how much
they can be trusted. No third-party dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

Vec3 = Sequence[float]


# --------------------------------------------------------------------------- #
# Angle helpers (compass degrees, clockwise, wrap-safe)
# --------------------------------------------------------------------------- #


def wrap360(angle: float) -> float:
    """Wrap an angle to ``[0, 360)`` degrees."""
    return angle % 360.0


def angle_diff(target: float, current: float) -> float:
    """Shortest signed difference ``target - current`` in ``(-180, 180]``."""
    return (target - current + 180.0) % 360.0 - 180.0


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def magnetic_dip_deg(mag: Vec3, gravity: Vec3) -> float:
    """Magnetic dip (inclination) angle in degrees from field and gravity.

    *gravity* is the accelerometer vector (any scale); at rest it points "up".
    Positive dip means the field tilts downward (northern hemisphere). This is a
    powerful disturbance detector: it flags distortions that *rotate* the field
    even when they barely change its magnitude.
    """
    g = _norm(gravity)
    if g == 0:
        return float("nan")
    up = (gravity[0] / g, gravity[1] / g, gravity[2] / g)
    m_up = mag[0] * up[0] + mag[1] * up[1] + mag[2] * up[2]
    horiz = (mag[0] - m_up * up[0], mag[1] - m_up * up[1], mag[2] - m_up * up[2])
    h = _norm(horiz)
    return math.degrees(math.atan2(-m_up, h))


def _soft_gate(deviation: float, tol: float, hard: Optional[float] = None) -> float:
    """Trust weight in ``[0, 1]``: 1 within *tol*, ramping to 0 at *hard*."""
    if hard is None:
        hard = 2.0 * tol
    if deviation <= tol:
        return 1.0
    if deviation >= hard:
        return 0.0
    return 1.0 - (deviation - tol) / (hard - tol)


# --------------------------------------------------------------------------- #
# Output record
# --------------------------------------------------------------------------- #


@dataclass
class StabilizedHeading:
    """Result of one :meth:`HeadingStabilizer.update` step."""

    heading: float          # stabilized heading, [0, 360) -- damped if enabled
    mag_trust: float        # 0..1 weight the magnetometer got this step
    cog_weight: float       # 0..1 weight COG got this step
    field_ratio: float      # |m| / reference (nan if no mag supplied)
    dip_deg: float          # measured dip (nan if no gravity supplied)
    coasting: bool          # True when the magnetometer was effectively rejected
    raw_heading: float = 0.0  # the un-damped fused heading behind `heading`
    wave_level: float = 0.0   # detected wave intensity (g of dynamic accel EMA)
    damping_scale: float = 1.0  # current adaptive-damping multiplier (>=1)


class HeadingSmoother:
    """Display damping for a heading: steadies the readout, tracks real turns.

    This is a **display/output** layer, deliberately separate from the fusion in
    :class:`HeadingStabilizer`. Feed it a heading each tick and it returns a
    damped heading that:

    * **holds steady through jitter** -- a signal bouncing 180<->182 shows a
      rock-steady ~181, via a deadband that ignores sub-threshold wander;
    * **removes last-degree noise** with a light low-pass that averages the
      jitter toward its centre;
    * **snaps on real turns** -- a change beyond ``snap_threshold_deg`` bypasses
      the damping so fast turns are shown with no added lag.

    Because it is separate, you can damp what a human/plotter sees while feeding
    the *un-damped* :attr:`StabilizedHeading.raw_heading` to a control loop.

    Parameters
    ----------
    time_constant:
        Low-pass time constant (s). Larger = smoother/steadier, more lag.
    deadband_deg:
        The displayed value is held until the smoothed heading moves more than
        this from it. This is what freezes the readout during small jitter.
    snap_threshold_deg:
        A change larger than this jumps immediately (fast-turn bypass).
    """

    def __init__(self, time_constant: float = 1.0, deadband_deg: float = 1.0,
                 snap_threshold_deg: float = 30.0) -> None:
        self.tau = time_constant
        self.deadband = deadband_deg
        self.snap = snap_threshold_deg
        self._smoothed: Optional[float] = None
        self._display: Optional[float] = None

    def reset(self) -> None:
        self._smoothed = None
        self._display = None

    def update(self, heading_deg: float, dt: float) -> float:
        h = wrap360(heading_deg)
        if self._smoothed is None:
            self._smoothed = h
            self._display = h
            return h
        # Fast-turn bypass: jump straight there, no lag.
        if abs(angle_diff(h, self._display)) >= self.snap:
            self._smoothed = h
            self._display = h
            return h
        # Light low-pass: averages jitter toward its centre (e.g. 180<->182 ->
        # ~181).
        self._smoothed = wrap360(
            self._smoothed + _time_gain(dt, self.tau) * angle_diff(h, self._smoothed))
        err = angle_diff(self._smoothed, self._display)
        if abs(err) > self.deadband:
            # Left the band (a real move): track the smoothed value directly.
            self._display = self._smoothed
        else:
            # Inside the band: creep gently to the centre so the readout settles
            # on the middle of the jitter (181), then holds there -- steady, not
            # frozen off to one edge.
            self._display = wrap360(self._display + _time_gain(dt, self.tau) * err)
        return self._display


# --------------------------------------------------------------------------- #
# The stabilizer
# --------------------------------------------------------------------------- #


class HeadingStabilizer:
    """Complementary heading filter with magnetic gating and COG aiding.

    Construct one, then call :meth:`update` every time you have a new sensor
    sample (and, whenever available, a fresh GPS course/speed from *any* source).

    Parameters
    ----------
    expected_field:
        Reference calibrated field magnitude (raw sensor units). Pass the
        ``field_strength`` from your magnetometer calibration for best results.
        If ``None`` it is learned from the first clean non-zero samples.
    field_tolerance:
        Fractional band before the magnitude gate starts down-weighting the
        magnetometer (0.15 = 15%).
    expected_dip_deg:
        Reference dip angle. If ``None`` it is learned. Requires gravity to be
        supplied to :meth:`update` for the dip gate to act.
    dip_tolerance_deg:
        Degrees of dip deviation before the dip gate down-weights.
    dip_accel_tol:
        The dip gate is only applied when ``| |accel| - 1g | <= dip_accel_tol``
        (accel in g), i.e. when the accelerometer is measuring gravity rather
        than dynamic motion. Under waves/vibration the accel is corrupted and
        the computed dip is meaningless, so the gate is skipped and the robust
        magnitude gate carries the disturbance detection. Prevents wave
        acceleration from spuriously making the filter coast.
    mag_time_constant:
        Seconds. How fast the heading tracks the (trusted) magnetometer yaw
        *while turning*. Larger = smoother/slower. A few seconds lets the gyro
        handle dynamics while the magnetometer removes slow drift.
    adaptive_gain:
        When ``True`` (default), speed up the magnetometer correction while the
        vehicle is **not turning** (gyro near zero) so a post-turn offset settles
        quickly, and keep it slow while turning so the gyro leads the dynamics.
        This removes the "slow to settle after a turn" lag without adding jitter
        underway. See ``mag_time_constant_static`` and ``turn_rate_threshold``.
    mag_time_constant_static:
        Seconds. The (short) time constant used when stationary, if
        ``adaptive_gain`` is on. The effective time constant blends between this
        and ``mag_time_constant`` based on the current turn rate.
    turn_rate_threshold:
        deg/s. At/above this yaw rate the filter is fully in "turning" mode
        (uses ``mag_time_constant``); at zero it uses ``mag_time_constant_static``.
    cog_time_constant:
        Seconds. How fast the heading is pulled toward GPS course. Should be
        long (10-30 s) because COG differs from heading by leeway/current.
    cog_min_speed:
        Minimum speed for COG to be trusted at all. **Use the same unit as the
        ``speed`` you pass to** :meth:`update` (knots, m/s -- your choice).
    cog_full_speed:
        Speed at which COG gets full weight; trust ramps linearly from
        ``cog_min_speed`` to here.
    rate_of_turn_sign:
        +1 or -1, mapping the gyro Z rate to compass-heading rate. Only matters
        while coasting through a magnetic disturbance. Verify once: during a
        disturbance, turn to starboard; if the heading falls instead of rising,
        flip this sign.
    max_rate_dps:
        Optional slew-rate limit on the output heading (deg/s).
    output_smoothing, output_deadband:
        Enable a display-damping layer on the **returned** heading (see
        :class:`HeadingSmoother`): ``output_smoothing`` is its low-pass time
        constant (s) and ``output_deadband`` the hold band (deg). The internal
        estimate and :attr:`StabilizedHeading.raw_heading` stay un-damped, so
        control loops can still use the raw value. Both default 0 (off).
    wave_adaptive:
        When ``True`` (default) and display damping is enabled, the damping is
        **scaled up automatically as waves are detected** -- so the heading gets
        steadier in a seaway and stays responsive when calm. Waves are detected
        from dynamic acceleration (``| |accel| - 1g |``), which a slow turn does
        not trigger, so real turns are not mistaken for waves.
    wave_full_accel:
        Dynamic-acceleration level (g) at which the adaptive damping reaches its
        maximum; scaling ramps linearly from calm (0 g) to here.
    wave_damping_max:
        Maximum damping multiplier applied to the smoother's time constant and
        deadband in the roughest detected seas.
    wave_level_tau:
        Time constant (s) of the wave-level estimator -- long enough to track
        sea state rather than individual waves.
    """

    def __init__(
        self,
        *,
        expected_field: Optional[float] = None,
        field_tolerance: float = 0.15,
        expected_dip_deg: Optional[float] = None,
        dip_tolerance_deg: float = 10.0,
        dip_accel_tol: float = 0.25,
        mag_time_constant: float = 3.0,
        adaptive_gain: bool = True,
        mag_time_constant_static: float = 0.5,
        turn_rate_threshold: float = 15.0,
        cog_time_constant: float = 20.0,
        cog_min_speed: float = 0.5,
        cog_full_speed: float = 2.0,
        rate_of_turn_sign: float = -1.0,
        auto_rate_sign: bool = True,
        max_rate_dps: Optional[float] = None,
        adaptive_reference: bool = True,
        output_smoothing: float = 0.0,
        output_deadband: float = 0.0,
        wave_adaptive: bool = True,
        wave_full_accel: float = 0.25,
        wave_damping_max: float = 4.0,
        wave_level_tau: float = 4.0,
    ) -> None:
        self.expected_field = expected_field
        self.field_tolerance = field_tolerance
        self.expected_dip = expected_dip_deg
        self.dip_tolerance = dip_tolerance_deg
        self.dip_accel_tol = dip_accel_tol
        self.mag_tau = mag_time_constant
        self.adaptive_gain = adaptive_gain
        self.mag_tau_static = mag_time_constant_static
        self.turn_rate_threshold = turn_rate_threshold
        self.cog_tau = cog_time_constant
        self.cog_min_speed = cog_min_speed
        self.cog_full_speed = cog_full_speed
        self.rate_sign = rate_of_turn_sign
        self.auto_rate_sign = auto_rate_sign
        self.max_rate = max_rate_dps
        self.adaptive_reference = adaptive_reference

        # Optional display damping layer, applied only to the returned heading
        # (never fed back into the fusion). Off unless smoothing/deadband > 0.
        self._smoother: Optional[HeadingSmoother] = None
        self._base_smooth_tau = 0.0
        self._base_deadband = 0.0
        if output_smoothing > 0 or output_deadband > 0:
            self._smoother = HeadingSmoother(
                time_constant=output_smoothing if output_smoothing > 0 else 0.5,
                deadband_deg=output_deadband)
            self._base_smooth_tau = self._smoother.tau
            self._base_deadband = self._smoother.deadband

        # Wave-adaptive damping state.
        self.wave_adaptive = wave_adaptive
        self.wave_full_accel = wave_full_accel
        self.wave_damping_max = wave_damping_max
        self.wave_level_tau = wave_level_tau
        self._wave_level = 0.0

        self.heading: Optional[float] = None  # internal estimate, None until init
        self._prev_meas: Optional[float] = None
        self._sign_evidence = 0.0

    # -- reference learning --------------------------------------------------
    def _mag_trust(self, mag: Optional[Vec3],
                   gravity: Optional[Vec3]) -> tuple:
        """Return (trust 0..1, field_ratio, dip_deg)."""
        if mag is None:
            return 0.0, float("nan"), float("nan")
        fmag = _norm(mag)
        if self.expected_field is None and fmag > 0.0:
            self.expected_field = fmag
        if not self.expected_field:
            # No usable field reference (zero-magnitude sample at startup, or a
            # zero expected_field): the magnetometer cannot be trusted this
            # step. Keep expected_field unlearned (None) so a later good sample
            # can set it. Dip does not depend on the reference, so still report
            # it -- but do not learn expected_dip from an untrusted sample.
            dip = magnetic_dip_deg(mag, gravity) if gravity is not None else float("nan")
            return 0.0, float("nan"), dip
        ratio = fmag / self.expected_field
        mag_dev = abs(fmag - self.expected_field) / self.expected_field
        t = _soft_gate(mag_dev, self.field_tolerance)

        dip = float("nan")
        # The dip gate is only meaningful when the accelerometer is measuring
        # gravity. Under wave/vibration acceleration |accel| departs from 1g and
        # the computed dip is garbage, so skip it then (the magnitude gate above
        # is acceleration-invariant and carries the detection).
        quasi_static = False
        if gravity is not None:
            quasi_static = abs(_norm(gravity) - 1.0) <= self.dip_accel_tol
            dip = magnetic_dip_deg(mag, gravity)
            if quasi_static and not math.isnan(dip):
                if self.expected_dip is None:
                    self.expected_dip = dip
                t *= _soft_gate(abs(dip - self.expected_dip), self.dip_tolerance)

        # Slowly adapt the references, but only while trustworthy (and, for dip,
        # only from quasi-static samples) so noise cannot pull them off.
        if self.adaptive_reference and t > 0.6:
            self.expected_field += 0.01 * (fmag - self.expected_field)
            if quasi_static and not math.isnan(dip) and self.expected_dip is not None:
                self.expected_dip += 0.01 * (dip - self.expected_dip)
        return t, ratio, dip

    def _wave_damping(self, gravity: Optional[Vec3], dt: float) -> tuple:
        """Update the wave-level estimate and scale the display damping.

        Returns ``(wave_level, damping_scale)``. Waves are sensed as dynamic
        acceleration (``| |accel| - 1g |``); a slow turn produces almost none,
        so turns are not mistaken for waves. When rough, the smoother's time
        constant and deadband are multiplied up so the readout gets steadier.
        """
        if gravity is not None:
            dyn = abs(_norm(gravity) - 1.0)
            self._wave_level += _time_gain(dt, self.wave_level_tau) * (
                dyn - self._wave_level)
        scale = 1.0
        if self.wave_adaptive and self._smoother is not None:
            frac = (min(self._wave_level / self.wave_full_accel, 1.0)
                    if self.wave_full_accel > 0 else 1.0)
            scale = 1.0 + (self.wave_damping_max - 1.0) * frac
            self._smoother.tau = self._base_smooth_tau * scale
            self._smoother.deadband = self._base_deadband * scale
        return self._wave_level, scale

    # -- main step -----------------------------------------------------------
    def update(
        self,
        yaw_deg: float,
        dt: float,
        *,
        rate_of_turn_dps: Optional[float] = None,
        mag: Optional[Vec3] = None,
        gravity: Optional[Vec3] = None,
        cog_deg: Optional[float] = None,
        speed: Optional[float] = None,
        declination_deg: float = 0.0,
    ) -> StabilizedHeading:
        """Fuse one sample and return the stabilized heading.

        Parameters
        ----------
        yaw_deg:
            The module's fused yaw (``state.angle.yaw``). Treated as the
            magnetometer-informed heading measurement.
        dt:
            Seconds since the previous :meth:`update` call.
        rate_of_turn_dps:
            Gyro yaw rate (``state.angular_velocity.z``), used to coast through
            magnetic disturbances. If ``None``, the last heading is held.
        mag:
            Calibrated magnetometer vector, for the disturbance gate. If
            ``None`` the magnetometer is assumed untrusted (pure gyro coast).
        gravity:
            Accelerometer vector (``state.acceleration``), enabling the dip gate.
        cog_deg, speed:
            GPS course-over-ground (deg, true) and speed, **from any source you
            like** -- an NMEA receiver, gpsd, a UDP feed, another vehicle bus.
            Supply them whenever you have a fresh fix; pass ``None`` when you do
            not. ``speed`` must be in the same unit as ``cog_min_speed``.
        declination_deg:
            Magnetic declination to add so the output (and the COG comparison)
            are in true north. Pass 0 to stay in magnetic north.

        Notes
        -----
        The method is robust to irregular timing: ``dt`` scales every gain, so
        calling it at 5 Hz or 50 Hz, evenly or not, gives consistent behaviour.
        """
        # Module yaw is CCW-positive; compass heading is CW-positive -> negate
        # (matches yaw_to_heading / heading_from_magnetic).
        meas = wrap360(declination_deg - yaw_deg)
        prev_meas = self._prev_meas  # capture before it is overwritten below
        trust, ratio, dip = self._mag_trust(mag, gravity)

        # Auto-learn the gyro->heading sign from clean data: while the field is
        # trusted the fused yaw is reliable, so its rate should match the gyro
        # rate. Accumulate evidence and flip rate_sign if they disagree. This
        # removes the manual-sign footgun before any disturbance coast.
        if (self.auto_rate_sign and dt > 0 and trust > 0.6
                and self._prev_meas is not None and rate_of_turn_dps is not None):
            yaw_rate = angle_diff(meas, self._prev_meas) / dt
            if abs(yaw_rate) > 5.0 and abs(rate_of_turn_dps) > 5.0:
                agree = (yaw_rate > 0) == (rate_of_turn_dps > 0)
                self._sign_evidence += 1 if agree else -1
                self._sign_evidence = max(-20.0, min(20.0, self._sign_evidence))
                if abs(self._sign_evidence) >= 3:
                    self.rate_sign = 1.0 if self._sign_evidence > 0 else -1.0
        self._prev_meas = meas

        # First call: snap to the measurement.
        if self.heading is None:
            self.heading = meas
            wlvl, wscale = self._wave_damping(gravity, dt)
            shown = self._smoother.update(meas, dt) if self._smoother else meas
            return StabilizedHeading(shown, trust, 0.0, ratio, dip,
                                     coasting=trust < 0.2, raw_heading=meas,
                                     wave_level=wlvl, damping_scale=wscale)

        # 1. Predict forward with the gyro (immune to mag disturbance).
        if rate_of_turn_dps is not None and dt > 0:
            pred = wrap360(self.heading + self.rate_sign * rate_of_turn_dps * dt)
        else:
            pred = self.heading

        # 2. Correct toward the magnetometer yaw, weighted by how trusted it is.
        #    Adaptive gain: use a short time constant when not turning (settle
        #    fast) and the long one while turning (let the gyro lead). The turn
        #    rate is taken from the gyro if available, else the yaw change.
        tau = self.mag_tau
        if self.adaptive_gain:
            if rate_of_turn_dps is not None:
                turn = abs(rate_of_turn_dps)
            elif dt > 0 and prev_meas is not None:
                turn = abs(angle_diff(meas, prev_meas)) / dt
            else:
                turn = 0.0
            frac = min(turn / self.turn_rate_threshold, 1.0) if self.turn_rate_threshold > 0 else 1.0
            tau = self.mag_tau_static + frac * (self.mag_tau - self.mag_tau_static)
        k_mag = trust * _time_gain(dt, tau)
        heading = wrap360(pred + k_mag * angle_diff(meas, pred))

        # 3. Aid slowly toward GPS course when moving.
        cog_weight = 0.0
        if cog_deg is not None and speed is not None and speed >= self.cog_min_speed:
            span = max(self.cog_full_speed - self.cog_min_speed, 1e-6)
            speed_trust = min((speed - self.cog_min_speed) / span, 1.0)
            cog_weight = speed_trust * _time_gain(dt, self.cog_tau)
            cog_true = wrap360(cog_deg)  # GPS COG is already true north
            heading = wrap360(heading + cog_weight * angle_diff(cog_true, heading))

        # 4. Optional slew-rate limit.
        if self.max_rate is not None and dt > 0:
            step = angle_diff(heading, self.heading)
            cap = self.max_rate * dt
            if step > cap:
                heading = wrap360(self.heading + cap)
            elif step < -cap:
                heading = wrap360(self.heading - cap)

        self.heading = heading  # internal estimate stays un-damped
        # 5. Wave-adaptive display damping on the OUTPUT only (never fed back):
        #    stiffen the readout as waves are detected, relax it when calm.
        wlvl, wscale = self._wave_damping(gravity, dt)
        shown = self._smoother.update(heading, dt) if self._smoother else heading
        return StabilizedHeading(shown, trust, cog_weight, ratio, dip,
                                 coasting=trust < 0.2, raw_heading=heading,
                                 wave_level=wlvl, damping_scale=wscale)

    # -- convenience over a State -------------------------------------------
    def update_from_state(
        self,
        state,
        dt: float,
        *,
        mag_calibration=None,
        cog_deg: Optional[float] = None,
        speed: Optional[float] = None,
        declination_deg: float = 0.0,
    ) -> Optional[StabilizedHeading]:
        """Update directly from a :class:`hwt901b.sensor.State`.

        Extracts fused yaw, magnetometer and accelerometer, and computes the
        **earth-frame yaw rate** from the gyro and roll/pitch -- not the raw
        body-frame gyro Z, which is not the heading rate when the sensor is
        tilted (as it is on every wave). Using the earth-frame rate keeps the
        gyro prediction and turn detection valid under roll/pitch.

        If *mag_calibration* (a :class:`~hwt901b.calibration.MagCalibration`) is
        given, the magnetometer is corrected before gating. Returns ``None`` if
        the state has no fused angle yet.
        """
        if state.angle is None:
            return None
        rot = None
        if state.angular_velocity is not None:
            gy = state.angular_velocity.y
            gz = state.angular_velocity.z
            roll = math.radians(state.angle.roll)
            pitch = math.radians(state.angle.pitch)
            cp = math.cos(pitch)
            if abs(cp) < 0.2:              # guard the singularity near +/-90 pitch
                cp = math.copysign(0.2, cp) if cp else 0.2
            # earth-frame yaw rate (deg/s); auto_rate_sign resolves its sign.
            rot = (math.sin(roll) * gy + math.cos(roll) * gz) / cp
        mag = None
        if state.magnetic is not None:
            raw = (state.magnetic.x, state.magnetic.y, state.magnetic.z)
            mag = mag_calibration.apply(raw) if mag_calibration else raw
        gravity = None
        if state.acceleration is not None:
            gravity = (state.acceleration.x, state.acceleration.y,
                       state.acceleration.z)
        return self.update(
            state.angle.yaw, dt,
            rate_of_turn_dps=rot, mag=mag, gravity=gravity,
            cog_deg=cog_deg, speed=speed, declination_deg=declination_deg,
        )


def _time_gain(dt: float, tau: float) -> float:
    """Complementary-filter blend factor for step *dt* and time constant *tau*."""
    if tau <= 0 or dt <= 0:
        return 1.0
    return min(dt / tau, 1.0)
