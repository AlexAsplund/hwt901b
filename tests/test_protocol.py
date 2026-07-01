"""Hardware-free tests for framing, decoding, commands and calibration."""

import math
import struct

import pytest

from hwt901b import protocol as P
from hwt901b import (
    BytesTransport,
    HWT901B,
    fit_ellipsoid,
    fit_hard_iron,
    heading_from_magnetic,
)


def make_frame(ptype: int, *int16_values: int) -> bytes:
    """Build a valid 11-byte frame from up to four int16 values."""
    payload = struct.pack("<4h", *int16_values)
    body = bytes((P.FRAME_HEADER, ptype)) + payload
    return body + bytes((P.checksum(body + b"\x00"),))


# --------------------------------------------------------------------------- #
# Framing / checksum
# --------------------------------------------------------------------------- #


def test_checksum_and_verify():
    frame = make_frame(P.PacketType.ANGLE, 100, 200, 300, 1)
    assert P.verify(frame)
    # Corrupt one byte -> verification fails.
    bad = bytearray(frame)
    bad[3] ^= 0xFF
    assert not P.verify(bytes(bad))


def test_parser_resyncs_after_garbage():
    good = make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
    stream = b"\x00\x12\x55\xff" + good + b"\x55" + good
    parser = P.FrameParser()
    frames = list(parser.feed(stream))
    assert len(frames) == 2
    assert all(f.type == P.PacketType.ACCELERATION for f in frames)


def test_parser_handles_split_chunks():
    frame = make_frame(P.PacketType.ANGLE, 1, 2, 3, 4)
    parser = P.FrameParser()
    out = list(parser.feed(frame[:5]))
    assert out == []
    out = list(parser.feed(frame[5:]))
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Decoding & scaling
# --------------------------------------------------------------------------- #


def test_decode_acceleration_scaling():
    # 16384/32768 * 16g = 8g on Z, temperature 2500/100 = 25.0 C
    frame = make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
    raw = P.RawFrame(P.PacketType.ACCELERATION, frame[2:10])
    acc = P.decode(raw)
    assert acc.z == pytest.approx(8.0)
    assert acc.temperature == pytest.approx(25.0)


def test_decode_angle_scaling_and_negative():
    # -16384/32768 * 180 = -90 deg roll
    frame = make_frame(P.PacketType.ANGLE, -16384, 8192, 0, 1)
    ang = P.decode(P.RawFrame(P.PacketType.ANGLE, frame[2:10]))
    assert ang.roll == pytest.approx(-90.0)
    assert ang.pitch == pytest.approx(45.0)


def test_decode_gyro_and_quaternion():
    fr = make_frame(P.PacketType.ANGULAR_VELOCITY, 16384, 0, 0, 330)
    g = P.decode(P.RawFrame(P.PacketType.ANGULAR_VELOCITY, fr[2:10]))
    assert g.x == pytest.approx(1000.0)  # 16384/32768 * 2000

    fq = make_frame(P.PacketType.QUATERNION, 32768 - 65536, 0, 0, 0)  # -1.0
    q = P.decode(P.RawFrame(P.PacketType.QUATERNION, fq[2:10]))
    assert q.w == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #


def test_command_builders():
    assert P.unlock_command() == bytes((0xFF, 0xAA, 0x69, 0x88, 0xB5))
    assert P.save_command() == bytes((0xFF, 0xAA, 0x00, 0x00, 0x00))
    assert P.write_command(0x03, 0x0009) == bytes((0xFF, 0xAA, 0x03, 0x09, 0x00))
    assert P.read_command(0x2E) == bytes((0xFF, 0xAA, 0x27, 0x2E, 0x00))


def test_rsw_mask():
    mask = P.rsw_mask(P.RswBit.ACCELERATION, P.RswBit.ANGLE, P.RswBit.MAGNETIC)
    assert mask == (1 << 1) | (1 << 3) | (1 << 4)


# --------------------------------------------------------------------------- #
# Sensor over an in-memory transport
# --------------------------------------------------------------------------- #


def test_sensor_reads_from_bytes_transport():
    stream = (
        make_frame(P.PacketType.ACCELERATION, 0, 0, 16384, 2500)
        + make_frame(P.PacketType.ANGLE, 0, 0, 16384, 1)
    )
    imu = HWT901B(BytesTransport(stream))
    state = imu.read_state(timeout=1.0)
    assert state.acceleration.z == pytest.approx(8.0)
    assert state.angle.yaw == pytest.approx(90.0)


def test_write_register_emits_unlock_then_command():
    t = BytesTransport()
    imu = HWT901B(t)
    imu.write_register(P.Register.RRATE, int(P.OutputRate.HZ_100))
    assert bytes(t.written) == P.unlock_command() + P.write_command(
        P.Register.RRATE, 0x09)


# --------------------------------------------------------------------------- #
# Calibration math
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# NMEA 0183 output
# --------------------------------------------------------------------------- #


def test_nmea_checksum_known_sentence():
    from hwt901b import nmea
    # Textbook NMEA example: $GPGGA,123519,...,,*47
    body = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
    assert nmea.checksum(body) == "47"


def test_nmea_sentence_framing_and_roundtrip():
    from hwt901b import nmea
    s = nmea.hdt(123.4)
    assert s.startswith("$HCHDT,123.4,T*")
    assert s.endswith("\r\n")
    body, cs = s[1:].split("*")
    assert cs.strip() == nmea.checksum(body)


def test_nmea_hdg_and_rot_and_xdr():
    from hwt901b import nmea
    assert nmea.hdg(90.0, None, 5.0).startswith("$HCHDG,90.0,,,5.0,E*")
    assert nmea.hdg(90.0, None, -3.0).startswith("$HCHDG,90.0,,,3.0,W*")
    assert nmea.rot(-12.0).startswith("$TIROT,-12.0,A*")
    assert nmea.xdr_attitude(roll_deg=2.0, pitch_deg=-1.5).startswith(
        "$YXXDR,A,-1.5,D,PTCH,A,2.0,D,ROLL*")


def test_nmea_from_heading_stabilized():
    from hwt901b import nmea
    # A stabilized TRUE heading of 100, variation 10E -> magnetic 90.
    out = nmea.sentences_from_heading(100.0, variation_deg=10.0,
                                      roll_deg=1.0, pitch_deg=-2.0,
                                      rate_of_turn_dpm=180.0)
    joined = "".join(out)
    assert "$HCHDT,100.0,T*" in joined
    assert "$HCHDM,90.0,M*" in joined
    assert "$TIROT,180.0,A*" in joined
    assert "$YXXDR,A,-2.0,D,PTCH,A,1.0,D,ROLL*" in joined
    for line in out:
        body, cs = line[1:].split("*")
        assert cs.strip() == nmea.checksum(body)


def test_nmea_from_state_uses_fused_yaw():
    from hwt901b import nmea
    from hwt901b.sensor import State

    st = State()
    st.angle = P.Angle(roll=1.0, pitch=-2.0, yaw=-90.0, version=1)
    st.angular_velocity = P.AngularVelocity(0.0, 0.0, 3.0, 0.0)
    out = nmea.sentences_from_state(st, variation_deg=10.0)
    joined = "".join(out)
    # yaw is negated for compass: -(-90) = 90 magnetic; true = 10 - (-90) = 100
    assert "$HCHDM,90.0,M*" in joined
    assert "$HCHDT,100.0,T*" in joined
    # gyro z +3 deg/s is a CCW (port) turn; NMEA ROT negative = bow to port,
    # so -3 deg/s * 60 = -180 deg/min
    assert "$TIROT,-180.0,A*" in joined
    # every emitted sentence must carry a valid checksum
    for line in out:
        body, cs = line[1:].split("*")
        assert cs.strip() == nmea.checksum(body)


# --------------------------------------------------------------------------- #
# Heading stabilizer + GPS
# --------------------------------------------------------------------------- #


def test_stabilizer_tracks_yaw_when_field_is_clean():
    from hwt901b import HeadingStabilizer, yaw_to_heading
    from hwt901b.stabilizer import angle_diff
    st = HeadingStabilizer(expected_field=1000.0, mag_time_constant=0.1)
    st.update(100.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
    for _ in range(20):
        out = st.update(120.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
    assert not out.coasting
    assert out.mag_trust > 0.9
    # Heading tracks the module yaw *through the compass conversion* (negated).
    assert abs(angle_diff(out.heading, yaw_to_heading(120.0))) < 1.0


def test_stabilizer_auto_learns_rate_sign_for_correct_coasting():
    from hwt901b import HeadingStabilizer
    from hwt901b.stabilizer import angle_diff
    st = HeadingStabilizer(expected_field=1000.0)
    # Feed a consistent clean turn so the sign detector can lock on.
    yaw = 0.0
    for _ in range(8):
        yaw += 10.0     # module yaw rising; gyro reads -100 dps
        st.update(yaw, 0.1, rate_of_turn_dps=-100.0, mag=(1000, 0, 0))
    # Now lose the magnetometer and coast: heading must keep moving the same
    # direction the trusted heading was going (compass = -yaw, i.e. decreasing).
    before = st.heading
    coast = st.update(yaw, 0.1, rate_of_turn_dps=-100.0, mag=None)
    assert angle_diff(coast.heading, before) < 0


def test_stabilizer_adaptive_gain_settles_faster_when_stationary():
    from hwt901b import HeadingStabilizer
    from hwt901b.stabilizer import angle_diff

    def settle_error(adaptive):
        st = HeadingStabilizer(expected_field=1000.0, mag_time_constant=3.0,
                               adaptive_gain=adaptive, mag_time_constant_static=0.3)
        st.update(0.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
        # Introduce a 30 deg offset in the measurement while stationary
        # (gyro reads ~0) and let it settle for 1 second.
        out = None
        for _ in range(10):
            out = st.update(-30.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
        # target compass heading for yaw=-30 is +30
        return abs(angle_diff(out.heading, 30.0))

    fast = settle_error(adaptive=True)
    slow = settle_error(adaptive=False)
    assert fast < slow            # adaptive converges faster when stationary
    assert fast < 2.0             # and gets close within a second


def test_stabilizer_coasts_on_gyro_during_mag_disturbance():
    from hwt901b import HeadingStabilizer, yaw_to_heading
    from hwt901b.stabilizer import angle_diff
    st = HeadingStabilizer(expected_field=1000.0, adaptive_reference=False)
    st.update(100.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
    held = yaw_to_heading(100.0)  # the compass heading it should hold
    # Field magnitude doubles and the fused yaw lurches -- classic magnetic
    # disturbance. We must ignore the bad yaw and hold the pre-disturbance value.
    out = st.update(180.0, 0.1, rate_of_turn_dps=0.0, mag=(2000, 0, 0))
    assert out.coasting
    assert out.mag_trust < 0.2
    assert out.field_ratio == pytest.approx(2.0)
    assert abs(angle_diff(out.heading, held)) < 5.0


def test_stabilizer_dip_gate_rejects_rotated_field():
    from hwt901b import HeadingStabilizer
    st = HeadingStabilizer(expected_field=1000.0, dip_tolerance_deg=10.0,
                           adaptive_reference=False)
    # Learn a horizontal field (dip 0) with gravity up.
    st.update(0.0, 0.1, mag=(1000, 0, 0), gravity=(0, 0, 1))
    # Same magnitude, but now tilted 45 deg out of plane -> dip gate should bite.
    out = st.update(0.0, 0.1, mag=(707, 0, -707), gravity=(0, 0, 1))
    assert out.field_ratio == pytest.approx(1.0, abs=0.01)  # magnitude unchanged
    assert out.mag_trust < 0.2                              # but dip rejects it


def test_stabilizer_wave_adaptive_stiffens_damping_in_waves():
    from hwt901b import HeadingStabilizer
    # With display damping enabled, feeding sustained dynamic acceleration (a
    # seaway) must raise the detected wave level and scale the damping up.
    st = HeadingStabilizer(expected_field=1000.0, output_smoothing=0.8,
                           output_deadband=1.0, wave_level_tau=1.0)
    # calm: |accel| ~ 1g -> no waves detected, damping unscaled
    for _ in range(30):
        out_calm = st.update(0.0, 0.1, rate_of_turn_dps=0.0,
                             mag=(1000, 0, 0), gravity=(0, 0, 1.0))
    assert out_calm.wave_level < 0.05
    assert out_calm.damping_scale == pytest.approx(1.0, abs=0.1)
    # waves: |accel| swings well away from 1g -> level rises, damping stiffens
    for i in range(60):
        az = 1.0 + (0.4 if i % 2 else -0.4)   # +/-0.4 g dynamic
        out_wave = st.update(0.0, 0.1, rate_of_turn_dps=0.0,
                             mag=(1000, 0, 0), gravity=(0, 0, az))
    assert out_wave.wave_level > 0.2
    assert out_wave.damping_scale > 2.0


def test_stabilizer_wave_detector_ignores_slow_turn():
    from hwt901b import HeadingStabilizer
    # A steady turn (no dynamic acceleration) must NOT be seen as waves.
    st = HeadingStabilizer(expected_field=1000.0, output_smoothing=0.8,
                           output_deadband=1.0, wave_level_tau=1.0)
    yaw = 0.0
    out = None
    for _ in range(50):
        yaw += 2.0    # turning at 20 deg/s, but |accel| stays 1g
        out = st.update(yaw, 0.1, rate_of_turn_dps=-20.0,
                        mag=(1000, 0, 0), gravity=(0, 0, 1.0))
    assert out.wave_level < 0.05
    assert out.damping_scale == pytest.approx(1.0, abs=0.1)


def test_stabilizer_dip_gate_ignored_under_dynamic_accel():
    # Regression for the wave finding: under large acceleration the dip gate
    # must be skipped (accel isn't gravity), so trust stays high on the robust
    # magnitude gate instead of false-coasting.
    from hwt901b import HeadingStabilizer
    st = HeadingStabilizer(expected_field=1000.0, dip_tolerance_deg=10.0,
                           adaptive_reference=False)
    st.update(0.0, 0.1, mag=(1000, 0, 0), gravity=(0, 0, 1))   # learn dip 0, static
    # Same field magnitude, but 3 g of acceleration and a field tilted 45 deg:
    # a static dip gate would reject this; accel-gating must let it through.
    out = st.update(0.0, 0.1, mag=(707, 0, -707), gravity=(0, 0, 3))
    assert out.field_ratio == pytest.approx(1.0, abs=0.01)
    assert out.mag_trust > 0.8


def test_stabilizer_cog_aiding_pulls_heading():
    from hwt901b import HeadingStabilizer
    from hwt901b.stabilizer import angle_diff
    st = HeadingStabilizer(expected_field=1000.0, cog_time_constant=1.0,
                           cog_min_speed=1.0, cog_full_speed=1.0)
    st.update(100.0, 0.1, mag=None)                 # init at 100, no mag
    for _ in range(100):
        out = st.update(100.0, 0.1, mag=None, cog_deg=150.0, speed=5.0)
    assert out.cog_weight > 0.0
    assert abs(angle_diff(out.heading, 150.0)) < 5.0


def test_heading_smoother_holds_steady_through_jitter():
    from hwt901b import HeadingSmoother
    sm = HeadingSmoother(time_constant=1.0, deadband_deg=1.5)
    outs = []
    # Signal bounces 180<->182 (centre 181). dt=0.1.
    for i in range(80):
        outs.append(sm.update(180.0 if i % 2 else 182.0, 0.1))
    tail = outs[-20:]
    spread = max(tail) - min(tail)
    assert spread < 0.5                     # readout is rock-steady...
    assert 180.0 <= sum(tail) / len(tail) <= 182.0   # ...near the centre


def test_heading_smoother_snaps_on_real_turn():
    from hwt901b import HeadingSmoother
    from hwt901b.stabilizer import angle_diff
    sm = HeadingSmoother(time_constant=2.0, deadband_deg=1.0, snap_threshold_deg=30.0)
    sm.update(100.0, 0.1)
    out = sm.update(200.0, 0.1)             # 100 deg jump -> immediate
    assert abs(angle_diff(out, 200.0)) < 1.0


def test_stabilizer_output_damping_keeps_raw():
    from hwt901b import HeadingStabilizer
    st = HeadingStabilizer(expected_field=1000.0, output_deadband=2.0,
                           mag_time_constant=0.1)
    st.update(0.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
    out = None
    for _ in range(10):
        out = st.update(-1.0, 0.1, rate_of_turn_dps=0.0, mag=(1000, 0, 0))
    # 1 deg wander is inside the 2 deg deadband -> shown heading holds at 0,
    # but raw_heading tracks the fused value (~1 in compass terms).
    assert abs(out.heading - 0.0) < 0.5
    assert abs(out.raw_heading - 1.0) < 0.5


def test_wave_profiler_recommends_and_extrapolates():
    from hwt901b import WaveProfiler, WaveProfile
    import math
    prof = WaveProfiler(intensity_tau=1.0)
    # 30 s calm, then 30 s of ~0.15 g oscillatory chop with 3 s roll period
    t = 0.0
    for i in range(600):      # calm, 20 Hz
        prof.feed(0.01, 0.0, 0.0, 45.0, 0.05); t += 0.05
    for i in range(1200):     # chop
        dyn = 0.15 * abs(math.sin(2 * math.pi * (1 / 3.0) * t))
        roll = 12.0 * math.sin(2 * math.pi * (1 / 3.0) * t)
        prof.feed(dyn, roll, 2.0, 45.0 + roll * 0.1, 0.05); t += 0.05
    p = prof.summarize()
    assert p.samples > 1000
    assert p.rough_intensity > p.calm_intensity          # detected the chop
    assert 2.0 < p.dominant_period_s < 4.5               # ~3 s roll period
    # full-damping threshold extrapolated beyond the worst observed intensity
    assert p.recommended["wave_full_accel"] > p.rough_intensity
    # round-trips through JSON-able dict, yields stabilizer kwargs
    p2 = WaveProfile.from_dict(p.to_dict())
    assert "wave_full_accel" in p2.stabilizer_kwargs()
    # extrapolation table ramps damping toward the max as seas build
    scales = [e["damping_scale"] for e in p.extrapolation]
    assert scales == sorted(scales)
    assert scales[-1] > scales[0]


def test_magnetic_dip_angle():
    from hwt901b import magnetic_dip_deg
    assert abs(magnetic_dip_deg((1, 0, 0), (0, 0, 1))) < 1e-6       # horizontal
    assert magnetic_dip_deg((0, 0, -1), (0, 0, 1)) == pytest.approx(90.0)  # down


def test_gps_parse_rmc_and_vtg():
    from hwt901b import parse_nmea_gps
    rmc = parse_nmea_gps(
        "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A")
    assert rmc is not None and rmc.valid
    assert rmc.speed_knots == pytest.approx(22.4)
    assert rmc.course_deg == pytest.approx(84.4)
    vtg = parse_nmea_gps("$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48")
    assert vtg is not None
    assert vtg.course_deg == pytest.approx(54.7)
    assert vtg.speed_knots == pytest.approx(5.5)
    assert parse_nmea_gps("$GPGGA,123519,4807.038,N,...") is None  # other type


def _make_ellipsoid_samples(centre, scale, n=800):
    """Sample a distorted sphere: hard-iron offset + per-axis soft-iron scale."""
    samples = []
    # Deterministic quasi-uniform spread over the sphere (Fibonacci lattice).
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - z * z))
        theta = golden * i
        x, y = r * math.cos(theta), r * math.sin(theta)
        samples.append((
            centre[0] + scale[0] * x,
            centre[1] + scale[1] * y,
            centre[2] + scale[2] * z,
        ))
    return samples


def test_fit_ellipsoid_recovers_hard_iron():
    centre = (120.0, -45.0, 300.0)
    scale = (500.0, 650.0, 480.0)  # anisotropic -> needs soft-iron
    samples = _make_ellipsoid_samples(centre, scale)
    cal = fit_ellipsoid(samples)
    for got, want in zip(cal.hard_iron, centre):
        assert got == pytest.approx(want, abs=1.0)
    # After correction all points should lie on a sphere -> tiny residual.
    assert cal.residual(samples) < 0.02 * cal.field_strength


def test_fit_ellipsoid_makes_magnitudes_uniform():
    samples = _make_ellipsoid_samples((10, 20, -30), (300, 450, 380))
    cal = fit_ellipsoid(samples)
    mags = []
    for s in samples:
        v = cal.apply(s)
        mags.append(math.sqrt(sum(c * c for c in v)))
    spread = (max(mags) - min(mags)) / (sum(mags) / len(mags))
    assert spread < 0.05  # magnitudes uniform to a few percent


def test_fit_hard_iron_centre():
    samples = _make_ellipsoid_samples((7, -3, 11), (100, 100, 100))
    cal = fit_hard_iron(samples)
    assert cal.hard_iron[0] == pytest.approx(7, abs=1.0)
    assert cal.hard_iron[2] == pytest.approx(11, abs=1.0)


def test_heading_cardinal_directions():
    # Y+ forward, compass bearings. The vector is the (calibrated) field, which
    # points at magnetic north; heading is the bearing of the Y+ axis.
    # Field along +Y  -> Y+ points north -> 0 deg.
    assert heading_from_magnetic((0, 1, 0)) == pytest.approx(0.0)
    # Field along -X  -> Y+ points east  -> 90 deg.
    assert heading_from_magnetic((-1, 0, 0)) == pytest.approx(90.0)
    # Field along -Y  -> Y+ points south -> 180 deg.
    assert heading_from_magnetic((0, -1, 0)) == pytest.approx(180.0)
    # Field along +X  -> Y+ points west  -> 270 deg.
    assert heading_from_magnetic((1, 0, 0)) == pytest.approx(270.0)


def test_yaw_to_heading_negates_and_applies_declination():
    from hwt901b import yaw_to_heading
    # Module yaw is CCW-positive; compass heading is CW-positive -> negate.
    assert yaw_to_heading(0.0) == pytest.approx(0.0)
    assert yaw_to_heading(90.0) == pytest.approx(270.0)      # turning left != +90
    assert yaw_to_heading(-90.0) == pytest.approx(90.0)
    assert yaw_to_heading(180.0) == pytest.approx(180.0)     # fixed point
    assert yaw_to_heading(90.0, 5.0) == pytest.approx(275.0)  # (5 - 90) % 360


def test_tilt_compensation_matches_level_when_flat():
    from hwt901b import tilt_compensated_heading
    # Level sensor (gravity on +z). Tilt-compensated heading should agree with
    # the naive level heading for any horizontal field direction.
    for mag in [(1, 0, 0), (0.5, 0.87, 0), (-1, 0, 0), (0, -1, 0)]:
        flat = heading_from_magnetic(mag)
        tilt = tilt_compensated_heading((0, 0, 1), mag)
        assert tilt == pytest.approx(flat, abs=1e-6)


def test_tilt_compensation_stable_under_pitch():
    from hwt901b import tilt_compensated_heading
    # A field pointing at magnetic north with a downward (dip) component.
    # Heading must stay constant as the sensor is pitched forward.
    import math
    dip = math.radians(60)  # steep inclination to stress the correction
    north_field = (math.cos(dip), 0.0, math.sin(dip))

    def rotate_pitch(v, theta):
        # Rotate a vector into the sensor frame for a forward pitch of theta.
        x, y, z = v
        return (x * math.cos(theta) + z * math.sin(theta),
                y,
                -x * math.sin(theta) + z * math.cos(theta))

    headings = []
    for deg in (0, 15, 30, 45):
        theta = math.radians(deg)
        g = rotate_pitch((0, 0, 1), theta)      # gravity in sensor frame
        m = rotate_pitch(north_field, theta)    # field in sensor frame
        headings.append(tilt_compensated_heading(g, m))
    # All headings should be (nearly) identical despite the changing tilt.
    assert max(headings) - min(headings) < 1.0


# --------------------------------------------------------------------------- #
# Mounting-orientation remap
# --------------------------------------------------------------------------- #


def test_mount_identity_is_noop():
    from hwt901b import Mount
    m = Mount.identity()
    assert m.is_identity
    assert m.apply_vector((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    ang = P.Angle(12.0, -7.0, 130.0, 0)
    out = m.apply_angle(ang)
    assert (out.roll, out.pitch, out.yaw) == pytest.approx((12.0, -7.0, 130.0))


def test_mount_z_up_to_y_remaps_vectors():
    from hwt901b import Mount
    m = Mount.z_up_to_y()
    assert not m.is_identity
    # A vertically-mounted sensor at rest reads gravity on -Y; remap -> level +Z.
    assert m.apply_vector((0.0, -1.0, 0.0)) == pytest.approx((0.0, 0.0, 1.0))
    # Former up-axis (Z+) becomes body left (Y+); X (bow) unchanged.
    assert m.apply_vector((0.0, 0.0, 1.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert m.apply_vector((1.0, 0.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


def test_mount_rejects_left_handed_map():
    from hwt901b import Mount
    with pytest.raises(ValueError):
        Mount.from_axes(x="x", y="z", z="y")   # det = -1, not a rotation


def test_mount_preserves_heading_when_mounted_vertical():
    from hwt901b import Mount
    from hwt901b.mount import _euler_to_matrix, _matrix_to_euler, _matmul
    m = Mount.z_up_to_y()
    # Boat dead level, heading 40 deg. The physically vertical sensor reports a
    # rolled-over attitude; after the remap the body frame must read level with
    # the heading intact.
    r_es = _matmul(_euler_to_matrix(0.0, 0.0, 40.0), m.matrix)
    dev_roll, dev_pitch, dev_yaw = _matrix_to_euler(r_es)
    out = m.apply_angle(P.Angle(dev_roll, dev_pitch, dev_yaw, 0))
    assert out.roll == pytest.approx(0.0, abs=1e-6)
    assert out.pitch == pytest.approx(0.0, abs=1e-6)
    assert out.yaw == pytest.approx(40.0, abs=1e-6)


def test_mag_calibration_rotated_matches_body_frame():
    from hwt901b import MagCalibration, Mount
    from hwt901b.mount import _matvec
    cal = MagCalibration(
        hard_iron=(19.0, -37.0, 23.0),
        soft_iron=((1.02, 0.01, -0.03), (0.01, 0.98, 0.02), (-0.03, 0.02, 1.05)),
        field_strength=3279.7)
    m = Mount.from_axes(x="z", y="x", z="y")
    calb = cal.rotated(m)
    raw = (123.0, -455.0, 678.0)
    # Applying the rotated cal to the remapped vector must equal rotating the
    # raw-frame correction: soft'@(R@raw - hard') == R@(soft@(raw-hard)).
    via_body = calb.apply(_matvec(m.matrix, raw))
    via_ref = _matvec(m.matrix, cal.apply(raw))
    assert via_body == pytest.approx(via_ref, abs=1e-9)
    # Identity / None is a no-op that returns the same object.
    assert cal.rotated(None) is cal
    assert cal.rotated(Mount.identity()) is cal


def test_driver_applies_mount_to_decoded_state():
    from hwt901b import Mount
    # Feed an acceleration frame of (0, -1g, 0) through a vertically-mounted
    # driver; the rolling state should surface it as level (0, 0, 1g).
    raw = int(round(-1.0 / P.ACC_RANGE_G * P.SHORT_SCALE))
    frame = make_frame(P.PacketType.ACCELERATION, 0, raw, 0, 2500)
    imu = HWT901B(BytesTransport(frame), mount=Mount.z_up_to_y())
    state = imu.read_state()
    assert state.acceleration.x == pytest.approx(0.0, abs=1e-3)
    assert state.acceleration.y == pytest.approx(0.0, abs=1e-3)
    assert state.acceleration.z == pytest.approx(1.0, abs=1e-3)
