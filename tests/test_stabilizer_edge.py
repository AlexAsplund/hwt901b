"""Regression tests for zero-magnitude magnetometer edge cases in the stabilizer."""

import math

from hwt901b.stabilizer import HeadingStabilizer


def test_zero_mag_first_sample_does_not_crash_and_recovers():
    """A zero-vector mag on the very first update must not raise, must yield
    trust 0.0, and must not poison the learned field reference: a subsequent
    good sample learns the reference and is fully trusted."""
    stab = HeadingStabilizer()

    out = stab.update(0.0, 0.1, mag=(0.0, 0.0, 0.0), gravity=(0.0, 0.0, 1.0))
    assert out.mag_trust == 0.0
    assert math.isnan(out.field_ratio)
    # Reference must stay unlearned so a later good sample can set it.
    assert stab.expected_field is None

    out2 = stab.update(0.0, 0.1, mag=(1000.0, 0.0, 0.0), gravity=(0.0, 0.0, 1.0))
    assert stab.expected_field == 1000.0
    assert out2.mag_trust > 0.99
    assert abs(out2.field_ratio - 1.0) < 1e-9


def test_constructed_zero_expected_field_does_not_crash():
    """expected_field=0.0 is falsy but not None; it must not divide by zero."""
    stab = HeadingStabilizer(expected_field=0.0)
    out = stab.update(0.0, 0.1, mag=(1000.0, 0.0, 0.0), gravity=(0.0, 0.0, 1.0))
    assert out.mag_trust == 0.0
    assert math.isnan(out.field_ratio)


def test_zero_mag_mid_stream_does_not_crash():
    """A zero-vector mag after the reference is learned must not raise."""
    stab = HeadingStabilizer()
    stab.update(0.0, 0.1, mag=(1000.0, 0.0, 0.0), gravity=(0.0, 0.0, 1.0))
    out = stab.update(0.0, 0.1, mag=(0.0, 0.0, 0.0), gravity=(0.0, 0.0, 1.0))
    # Far outside the tolerance band: fully distrusted, but no exception.
    assert out.mag_trust == 0.0
