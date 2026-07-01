"""
Motion profiling and auto-tuning -- pure standard library.

Run :class:`WaveProfiler` during a normal outing (calm, then whatever small and
medium waves the day offers) and it characterises the boat's real motion, then
**extrapolates** to rougher seas you did not experience to recommend
:class:`~hwt901b.stabilizer.HeadingStabilizer` settings tuned to *your* hardware
and vessel -- instead of hand-picked defaults.

What it measures
----------------
* the distribution of **dynamic acceleration** ``| |accel| - 1g |`` (the wave
  intensity the stabilizer keys on), via a short-window RMS -> percentiles;
* roll / pitch amplitude;
* the **dominant motion period** (wave encounter period) from roll zero-crossings;
* the heading **jitter when calm** (baseline noise to hide with a deadband).

What it recommends
------------------
Settings for the wave-adaptive damping and display damping, with the "full
damping" threshold set for seas rougher than you saw (linear extrapolation of
dynamic acceleration with sea state). Save the profile to JSON and load it back
to configure a stabilizer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

Vec3 = Sequence[float]


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


@dataclass
class WaveProfile:
    """Result of a profiling session: observed stats + recommended settings."""

    duration_s: float
    samples: int
    # observed dynamic-acceleration intensity (g), windowed RMS percentiles
    calm_intensity: float          # p10
    typical_intensity: float       # p50
    rough_intensity: float         # p95
    peak_intensity: float          # max
    roll_amplitude: float          # deg, ~p95 of |roll|
    pitch_amplitude: float         # deg
    dominant_period_s: float       # s, from roll zero-crossings
    calm_heading_jitter: float     # deg, |dheading| per sample when calm
    recommended: Dict[str, float] = field(default_factory=dict)
    extrapolation: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WaveProfile":
        return cls(**d)

    def stabilizer_kwargs(self) -> Dict[str, float]:
        """Return kwargs to splat into ``HeadingStabilizer(**...)``."""
        return dict(self.recommended)


class WaveProfiler:
    """Accumulate motion samples during a session, then summarise + recommend.

    Feed one sample per sensor tick with :meth:`feed` (or :meth:`feed_from_state`
    plus a stabilized heading). Call :meth:`summarize` at the end.
    """

    def __init__(self, intensity_tau: float = 4.0,
                 store_hz: float = 5.0) -> None:
        self._tau = intensity_tau           # window for the intensity estimate
        self._ema_dyn2 = 0.0
        self._store_dt = 1.0 / store_hz
        self._since_store = 0.0
        self._t = 0.0
        self._n = 0
        # subsampled records to bound memory on long sessions
        self._intensity: List[float] = []
        self._roll: List[float] = []
        self._pitch: List[float] = []
        self._dheading: List[float] = []
        self._prev_heading: Optional[float] = None

    def feed(self, dyn_accel_g: float, roll_deg: float, pitch_deg: float,
             heading_deg: Optional[float], dt: float) -> None:
        """Feed one sample.

        *dyn_accel_g* is ``| |accel| - 1 |`` in g (0 at rest). *heading_deg* is
        the (stabilized) heading, used only to measure calm-time jitter; pass
        ``None`` if unavailable.
        """
        if dt <= 0:
            return
        self._t += dt
        self._n += 1
        a = 1.0 - math.exp(-dt / self._tau) if self._tau > 0 else 1.0
        self._ema_dyn2 += a * (dyn_accel_g * dyn_accel_g - self._ema_dyn2)
        dh = 0.0
        if heading_deg is not None and self._prev_heading is not None:
            dh = abs((heading_deg - self._prev_heading + 180) % 360 - 180)
        if heading_deg is not None:
            self._prev_heading = heading_deg
        self._since_store += dt
        if self._since_store >= self._store_dt:
            self._since_store = 0.0
            self._intensity.append(math.sqrt(max(0.0, self._ema_dyn2)))
            self._roll.append(roll_deg)
            self._pitch.append(pitch_deg)
            self._dheading.append(dh)

    def feed_from_state(self, state, heading_deg, dt: float) -> None:
        if state.acceleration is None or state.angle is None:
            return
        a = state.acceleration
        amag = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        self.feed(abs(amag - 1.0), state.angle.roll, state.angle.pitch,
                  heading_deg, dt)

    # -- analysis ------------------------------------------------------------
    def summarize(self, extrap_factor: float = 2.0,
                  damping_max: float = 4.0) -> WaveProfile:
        """Compute statistics and recommended settings.

        *extrap_factor*: full damping is reserved for seas this many times
        rougher (in dynamic acceleration) than the worst you observed.
        """
        inten = sorted(self._intensity)
        calm = _percentile(inten, 0.10)
        typ = _percentile(inten, 0.50)
        rough = _percentile(inten, 0.95)
        peak = inten[-1] if inten else 0.0
        roll_amp = _percentile(sorted(abs(r) for r in self._roll), 0.95)
        pitch_amp = _percentile(sorted(abs(p) for p in self._pitch), 0.95)
        period = self._dominant_period()
        # calm jitter: mean |dheading| over the calmest third of samples
        calm_jitter = self._calm_jitter(calm, typ)

        # --- recommendations (extrapolated) ---
        # full-damping threshold: rougher than the worst seen, floored so a flat
        # calm day still yields a sane value.
        wave_full = max(rough * extrap_factor, 0.10)
        # base deadband hides calm heading noise; keep modest.
        deadband = min(max(2.0 * calm_jitter, 0.5), 3.0)
        # base smoothing ~ a fraction of the wave period, clamped.
        smoothing = min(max(period / 8.0, 0.3), 1.5) if period > 0 else 0.8
        level_tau = min(max(2.0 * period, 3.0), 10.0) if period > 0 else 4.0

        rec = {
            "output_smoothing": round(smoothing, 2),
            "output_deadband": round(deadband, 2),
            "wave_full_accel": round(wave_full, 3),
            "wave_damping_max": round(damping_max, 1),
            "wave_level_tau": round(level_tau, 1),
        }

        # extrapolation table: predicted damping scale at multiples of the worst
        # observed intensity (what to expect as seas build beyond this session).
        extrap = []
        base = max(rough, 1e-3)
        for label, mult in (("observed calm", calm / base if base else 0),
                            ("observed typical", typ / base if base else 0),
                            ("observed worst", 1.0),
                            ("~2x worst", 2.0), ("~3x worst", 3.0)):
            lvl = base * mult
            frac = min(lvl / wave_full, 1.0) if wave_full > 0 else 1.0
            scale = 1.0 + (damping_max - 1.0) * frac
            extrap.append({"regime": label, "intensity_g": round(lvl, 3),
                           "damping_scale": round(scale, 2)})

        return WaveProfile(
            duration_s=round(self._t, 1), samples=self._n,
            calm_intensity=round(calm, 4), typical_intensity=round(typ, 4),
            rough_intensity=round(rough, 4), peak_intensity=round(peak, 4),
            roll_amplitude=round(roll_amp, 1), pitch_amplitude=round(pitch_amp, 1),
            dominant_period_s=round(period, 2),
            calm_heading_jitter=round(calm_jitter, 3),
            recommended=rec, extrapolation=extrap)

    def _dominant_period(self) -> float:
        # Median spacing between roll zero-crossings (robust to calm stretches,
        # which produce no crossings and would bias a total-time/count estimate).
        r = self._roll
        if len(r) < 8:
            return 0.0
        mean = sum(r) / len(r)
        idx = []
        for i in range(1, len(r)):
            a, b = r[i - 1] - mean, r[i] - mean
            if (a <= 0 < b) or (a >= 0 > b):
                idx.append(i)
        if len(idx) < 3:
            return 0.0
        gaps = sorted(idx[k] - idx[k - 1] for k in range(1, len(idx)))
        med = gaps[len(gaps) // 2]
        return 2.0 * med * self._store_dt   # two crossings per full period

    def _calm_jitter(self, calm: float, typ: float) -> float:
        thr = max(calm, 0.5 * (calm + typ))
        vals = [self._dheading[i] for i in range(len(self._intensity))
                if self._intensity[i] <= thr]
        if not vals:
            vals = self._dheading
        return sum(vals) / len(vals) if vals else 0.0
