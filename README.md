# hwt901b

A dependency-light Python driver and calibration toolkit for the **WitMotion
HWT901B-TTL** 9/10-axis AHRS (accelerometer, gyroscope, magnetometer,
barometer, on-board Kalman fusion).

- **Pure-stdlib core.** Protocol parsing, packet decoding, command building and
  the full magnetometer calibration math have **zero** third-party
  dependencies.
- **Serial is optional.** `pyserial` is imported lazily and only when you open a
  live port. Install it with `pip install hwt901b[serial]`.
- **Cross-platform.** Same code on Windows (`COM3`) and Linux
  (`/dev/ttyUSB0`).
- **Real calibration.** On-board accel/mag calibration commands *plus* an
  offline **hard-iron + soft-iron ellipsoid fit** implemented in plain Python
  (no numpy).

## Install

```bash
# core only (parse captures, build commands, run calibration math)
pip install .

# with live serial support
pip install ".[serial]"

# dev (pytest + pyserial)
pip install ".[dev]"
```

## Quick start

```python
from hwt901b import HWT901B

with HWT901B.open("COM3", baudrate=9600) as imu:   # or "/dev/ttyUSB0"
    state = imu.read_state()          # blocks until a frame arrives
    print(state.angle)                # Angle(roll=..., pitch=..., yaw=...)
    print(state.acceleration)         # Acceleration(x, y, z [g], temperature)
    print(state.magnetic)             # Magnetic(x, y, z [raw], temperature)

    for state in imu.stream(min_interval=0.1):   # 10 Hz snapshots
        a = state.angle
        print(f"{a.roll:7.2f} {a.pitch:7.2f} {a.yaw:7.2f}")
```

`state` is a rolling snapshot; each field is `None` until that packet type has
been seen. By default the module emits acceleration, angular velocity and angle.
Enable more:

```python
from hwt901b import RswBit
imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGULAR_VELOCITY,
                RswBit.ANGLE, RswBit.MAGNETIC, RswBit.QUATERNION)
```

## Command-line tool

```bash
python -m hwt901b monitor COM3                 # live dashboard
python -m hwt901b monitor /dev/ttyUSB0 -b 115200
python -m hwt901b calibrate accel COM3         # on-board gravity calibration
python -m hwt901b calibrate mag COM3 -s 20     # on-board magnetic calibration
python -m hwt901b magfit COM3 --write-offsets  # offline hard/soft-iron fit
python -m hwt901b config COM3 --rate 100 --six-axis false
python -m hwt901b info COM3
```

## Configuration

All writes follow the WIT **unlock → write → save** sequence automatically; the
module relocks its config registers ~10 s after the last write.

```python
from hwt901b import OutputRate, BaudRate, Bandwidth

imu.set_output_rate(OutputRate.HZ_100)    # 0.2 Hz .. 200 Hz
imu.set_bandwidth(Bandwidth.HZ_20)        # low-pass filter cutoff
imu.set_algorithm(six_axis=False)         # False = 9-axis absolute compass yaw
imu.set_orientation_vertical(True)        # mounting orientation hint
imu.set_baudrate(BaudRate.BPS_115200)     # host side follows automatically
imu.factory_reset()

# Low-level register access is available too:
version = imu.read_register(0x2E)         # returns four int16 words
imu.write_register(0x03, 0x09, save=True)
```

## Mounting orientation (mount it upright, not just flat)

The module assumes it lies **flat**: sensor X forward, Y left, Z up. Bolt it in
any other attitude and every reading — accel, gyro, mag, and the fused
roll/pitch/yaw — comes out in the *sensor's* frame. There are two ways to fix
that; pick by whether you want the correction in software or on the chip.

### Software remap — `Mount` (any orientation, exact axis control)

Pass a `Mount` to `open()` and every reading is transformed into your body frame
automatically. Say exactly which sensor axis becomes which body axis — e.g. stand
the unit on its edge so its old up-axis (**Z+**) becomes the new left (**Y+**):

```python
from hwt901b import HWT901B, Mount

# preset for the "upright / on-edge" case: Z+ (was up) -> Y+ (left), X (bow) kept
with HWT901B.open("COM3", mount=Mount.z_up_to_y()) as imu:
    s = imu.read_state()        # roll/pitch/yaw + vectors already in body frame
    print(s.angle, s.acceleration)

# or spell out the mapping yourself: body X<-sensor X, Y<-sensor Z, Z<-sensor -Y
mount = Mount.from_axes(x="x", y="z", z="-y")
```

Presets: `Mount.identity()` (flat, the default), `Mount.z_up_to_y()` /
`Mount.vertical()`, `Mount.upside_down()`, `Mount.yaw_90_ccw()`. The fused Euler
angles are recomputed by composing with the module's own orientation, so the
gyro-backed heading is preserved (not thrown away for an accel-only estimate).
Any left-handed / non-rotation axis map is rejected up front.

From the CLI, every data command takes `--mount`:

```bash
python -m hwt901b monitor    COM3 --mount vertical
python -m hwt901b stabilized COM3 --mount z-up-to-y
python -m hwt901b nmea       COM3 --mount x:z:-y      # custom body X:Y:Z map
```

### On-chip toggle — WitMotion's install direction

This is WitMotion's own setting (register `ORIENT`), the direct SDK equivalent.
It offers a single fixed vertical remap, but the module then fuses the heading
around true vertical on-chip:

```python
imu.set_orientation_vertical(True)        # horizontal (default) <-> vertical
```

```bash
python -m hwt901b config COM3 --orient vertical
```

Use the on-chip toggle if that one fixed vertical happens to match your install;
use the software `Mount` when you need a different attitude or precise control
over the axis assignment.

## Calibration

### 1. On-board (firmware) calibration

The simplest path — the module fits and stores the correction itself.

```python
# Accelerometer: keep the unit STILL and LEVEL for the whole call.
imu.calibrate_acceleration()

# Magnetometer: rotate slowly through all orientations while this runs.
imu.calibrate_magnetic(rotate_seconds=20, progress=lambda f: print(f"{f:.0%}"))
```

### 2. Offline hard-iron + soft-iron fit (recommended for accuracy)

The firmware corrects hard-iron and a coarse scale. When the sensor sits near
motors, steel or current-carrying wires you usually want a full **soft-iron**
correction too. Collect a cloud of raw samples (tumble the sensor through every
orientation) and fit an ellipsoid:

```python
from hwt901b import HWT901B, RswBit, fit_ellipsoid

with HWT901B.open("COM3") as imu:
    imu.set_outputs(RswBit.ANGLE, RswBit.MAGNETIC)
    samples = []
    while len(samples) < 600:
        imu.poll()
        if imu.state.magnetic:
            m = imu.state.magnetic
            samples.append((m.x, m.y, m.z))
            imu.state.magnetic = None

    cal = fit_ellipsoid(samples)
    print(cal.hard_iron_int())          # -> write to the module if you like
    print(cal.soft_iron)                # 3x3 matrix, apply in software
    print(f"RMS residual: {cal.residual(samples):.1f}")

    # Apply to a live reading:
    m = imu.read_state().magnetic
    x, y, z = cal.apply((m.x, m.y, m.z))
```

**Model:** `h_cal = soft_iron @ (h - hard_iron)`. The fit solves the algebraic
least-squares ellipsoid, recovers the centre (hard-iron) and shape matrix, then
takes a symmetric matrix square root (via an in-house Jacobi eigensolver) to
build the soft-iron matrix that maps the ellipsoid back to a sphere. Everything
is plain-Python lists — no numpy.

`fit_hard_iron()` is a faster min/max-only alternative when soft-iron skew is
negligible.

**Where to apply the correction — pick one source of truth:**

- *Software-only (recommended for full accuracy):* keep the module's offsets at
  zero and apply the whole `cal.apply(raw)` (hard **and** soft iron) in your
  code. One place does the correction, no double-counting.
- *Module hard-iron + software soft-iron:* push the hard-iron offsets into the
  module with `imu.set_magnetic_offsets(*cal.hard_iron_int())` and then apply
  **only** the soft-iron matrix in software. Do **not** also subtract
  `cal.hard_iron` again, or you correct the bias twice.

The module has no soft-iron registers, so soft-iron always lives in software.

### 3. Tilt-compensated heading

A plain compass heading is only valid when the sensor is level. Use the
accelerometer to correct for tilt:

```python
from hwt901b import tilt_compensated_heading

s = imu.read_state()
raw = (s.magnetic.x, s.magnetic.y, s.magnetic.z)
cal = mag_cal.apply(raw)                      # apply your MagCalibration first
heading = tilt_compensated_heading(
    (s.acceleration.x, s.acceleration.y, s.acceleration.z),
    cal,
    declination_deg=3.0,                      # your local declination for true north
)
```

This stays accurate as the sensor pitches and rolls (verified to hold within 1°
across a 45° pitch sweep in the test suite). Note its zero reference is the
magnetometer +X axis and will differ from the module's fused `Angle.yaw`, which
uses its own internal frame — both are valid, just different references.

## Marine / boat use

On a dynamic platform the **accelerometer-based** `tilt_compensated_heading` is
*not* enough: waves, turns and vibration add real acceleration, so the "down"
estimate (and thus the heading) swings. Use the module's **gyro-fused yaw**
instead — its onboard Kalman filter rides through that motion:

```python
heading = imu.read_true_heading(declination_deg=5.0)   # 0..360, gyro-stabilised
```

The harder problem afloat is **magnetic deviation**: the engine, hull,
batteries and current-carrying wiring distort the field far more than a laptop
did (we measured 1.16× from a laptop alone). A bench calibration is useless once
it's installed. So:

```bash
# 1. Mount the sensor in its final position, away from engine/steel/speakers/
#    DC cables. Then apply a boat-friendly baseline (9-axis, vibration filter):
python -m hwt901b marine-setup COM3            # 9-axis absolute + 20 Hz filter

# 2. "Compass swing": turn the WHOLE BOAT slowly through 360 deg (twice) while
#    the magnetometer calibrates in place, capturing the boat's own field:
python -m hwt901b swing COM3 -s 120

# 3. Verify yaw against a known bearing; fold any constant error into declination.
python -m hwt901b monitor COM3
```

Keep it in **9-axis** mode for an absolute (non-drifting) compass. Stay away
from the engine and anything carrying DC current, whose field changes with load.

### NMEA 0183 output

Feed a chartplotter, autopilot or nav app (OpenCPN, SignalK) with standard
sentences — heading (`HDT`/`HDM`/`HDG`), rate of turn (`ROT`) and attitude
(`XDR` pitch/roll). Headings come from the gyro-fused yaw.

```bash
python -m hwt901b nmea COM3 --variation 5.0            # print sentences to stdout
python -m hwt901b nmea COM3 --udp 255.255.255.255:10110  # UDP broadcast to OpenCPN
python -m hwt901b nmea COM3 --include hdt,rot,xdr --rate 10
```

Or build sentences yourself from a `State`:

```python
from hwt901b import nmea
for line in nmea.sentences_from_state(imu.read_state(), variation_deg=5.0):
    print(line, end="")     # $HCHDT,181.2,T*23  etc.
```

The `live_heading.py` example defaults to the **fused** heading (`--source
fused`) and shows the accelerometer-only `tilt` heading beside it for
comparison; use `--source tilt` only for a static/handheld sensor.

## Heading stabilization (magnetic gating + GPS aiding)

The module's fused yaw is good, but it can't know when the *magnetometer* is
being distorted (switched DC loads, passing steel) or which way the boat is
actually travelling. `HeadingStabilizer` adds those downstream corrections, the
same ones marine autopilots and AHRS research use:

- **Magnetic disturbance gating** — watches the calibrated field magnitude and
  the magnetic dip angle; when either strays from its reference it *down-weights*
  the magnetometer (soft, graduated) and coasts on the gyro rate-of-turn.
- **GPS course-over-ground aiding** — slowly pulls heading toward GPS course when
  moving, removing slow compass bias. **GPS comes from any source you supply.**
- **Smoothing / slew limiting** — optional.
- **Wave-adaptive damping** — detects a seaway from dynamic acceleration and
  automatically stiffens the display damping when waves are present, relaxing it
  when calm. A slow turn produces no dynamic acceleration, so turns are not
  mistaken for waves. On synthetic beam seas this cut readout jitter ~40% and
  lowered RMS heading error, with no cost to calm responsiveness. On by default
  when display damping is enabled (`wave_adaptive`, `wave_full_accel`,
  `wave_damping_max`). `StabilizedHeading.wave_level` / `.damping_scale` expose
  the current sea-state estimate.

```python
from hwt901b import HWT901B, HeadingStabilizer, RswBit, MagCalibration

stab = HeadingStabilizer(expected_field=3279.7)   # from your calibration

with HWT901B.open("COM3") as imu:
    imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGULAR_VELOCITY,
                    RswBit.ANGLE, RswBit.MAGNETIC)
    import time
    last = time.monotonic()
    while True:
        imu.poll()
        now = time.monotonic(); dt = now - last; last = now
        out = stab.update_from_state(
            imu.state, dt,
            mag_calibration=my_cal,          # a MagCalibration, or None
            cog_deg=gps_course, speed=gps_speed,   # from ANY source, or None
            declination_deg=5.0,
        )
        if out:
            print(out.heading, "COAST" if out.coasting else "",
                  "trust", out.mag_trust)
```

### Profiling: auto-tune to your boat and conditions

Rather than hand-picking thresholds, **profile a normal outing** — let it sit
calm, then run through whatever small/medium waves the day offers (wakes, chop) —
and the library characterises your real motion and **extrapolates** to rougher
seas to recommend tuned `HeadingStabilizer` settings.

```bash
python -m hwt901b profile COM3 -m 10        # profile for up to 10 min (Ctrl-C to stop)
# -> prints observed intensity/roll/period, recommended settings + an
#    extrapolation table, and saves wave_profile.json
python -m hwt901b stabilized COM3 --profile wave_profile.json   # use the tuning
```

It measures the distribution of dynamic acceleration (the wave signal), roll/
pitch amplitude, the dominant wave period, and calm-time heading jitter, then
sets `wave_full_accel` for seas rougher than you saw, plus base
`output_smoothing`/`output_deadband`/`wave_level_tau`. In code:

```python
from hwt901b import WaveProfiler, HeadingStabilizer
prof = WaveProfiler()
# ... each tick: prof.feed_from_state(imu.state, stabilized_heading, dt) ...
profile = prof.summarize()                       # WaveProfile
stab = HeadingStabilizer(expected_field=..., **profile.stabilizer_kwargs())
```

### Integrating GPS from your own source (for other software using this lib)

The stabilizer is intentionally decoupled from GPS I/O. `update()` takes two
plain numbers — `cog_deg` (course over ground, degrees true) and `speed` —
supplied **whenever you have a fresh fix**, and `None` when you don't. Wire them
from wherever your GPS lives:

```python
# You have your own GPS however you like. Just keep two latest values:
latest_cog, latest_speed = None, None

# ... update them from your source, e.g. one of:
#   * gpsd:      report = gpsd.next(); latest_cog = report.track; latest_speed = report.speed
#   * NMEA2000:  from a PGN 129026 (COG/SOG) handler
#   * UDP feed:  parse a datagram from your plotter / phone
#   * NMEA0183:  from hwt901b import parse_nmea_gps
#                fix = parse_nmea_gps(line)
#                if fix and fix.valid:
#                    latest_cog, latest_speed = fix.course_deg, fix.speed_knots

# Then, every sensor tick, hand the current values in (thread-safe: just reads):
out = stab.update(
    yaw_deg=imu.state.angle.yaw,
    dt=dt,
    rate_of_turn_dps=imu.state.angular_velocity.z,
    mag=my_cal.apply((m.x, m.y, m.z)),      # calibrated magnetometer vector
    gravity=(a.x, a.y, a.z),                # accelerometer, enables the dip gate
    cog_deg=latest_cog, speed=latest_speed, # <-- YOUR gps, any source, or None
    declination_deg=5.0,
)
```

Key contract for integrators:

| You provide | Meaning | If you omit it (`None`) |
|---|---|---|
| `yaw_deg` (required) | module fused yaw | — |
| `dt` (required) | seconds since last `update` | — |
| `rate_of_turn_dps` | gyro Z rate | can't coast; heading is held during disturbances |
| `mag` | **calibrated** field vector | magnetometer assumed untrusted (pure gyro) |
| `gravity` | accelerometer vector | dip gate disabled (magnitude gate still works) |
| `cog_deg`, `speed` | GPS course + speed, any source | no GPS aiding |

Notes:
- `speed` and the `cog_min_speed`/`cog_full_speed` constructor args must share a
  unit (knots or m/s — your choice; `GpsFix` exposes both `speed_knots` and
  `speed_ms`).
- GPS need not arrive at the sensor rate. Pass the **most recent** fix each tick;
  a 1 Hz GPS driving a 50 Hz sensor loop is fine.
- `declination_deg` puts the output (and the COG comparison) in true north; the
  stabilizer treats GPS COG as already-true.
- `HeadingStabilizer` holds no I/O, no threads and no third-party deps — safe to
  run in your own loop or feed from a queue.

The bundled `python -m hwt901b stabilized COM3 --gps-port COM7` and
`examples/stabilized_heading.py` show the NMEA-GPS case end to end; swap the
`get_gps()` / GPS-pump section for your own source.

## Wave / seakeeping simulation

`examples/wave_sim.py` generates **research-grounded synthetic sea data** to test
heading behaviour without going afloat, and `synthetic_waves/*.csv` holds
pre-generated datasets. It builds an irregular sea from a **JONSWAP spectrum**,
drives a small-boat motion model, synthesizes the raw IMU signals, and replays
them through the real library (naive tilt heading vs. device fused yaw vs.
`HeadingStabilizer`).

Physics used (see sources below):
- **JONSWAP spectrum** `S(ω)=(5/16)Hs²ωp⁴ω⁻⁵exp(-1.25(ωp/ω)⁴)·γ^r·(1-0.287 ln γ)`
- irregular surface as a **sum of sinusoids**, `aᵢ=√(2S(ωᵢ)Δω)`, random phase
- deep-water dispersion `k=ω²/g`, wave slope `k·a`
- encounter frequency `ωe = ω − (ω²/g)·U·cos μ`
- **roll resonance** near the boat's natural roll period (small craft ≈4 s,
  lightly damped → large resonant rolls in beam seas); pitch in head/following
  seas; heave follows long waves
- sea states from the bareboat-math validation set: Hs/Tp of 0.27 m/3 s,
  1.5 m/5.7 s, 4 m/8.5 s, 8.5 m/11.4 s, plus directional spread and a confused
  cross-sea

```bash
python examples/wave_sim.py           # comparison table across sea states/angles
python examples/wave_sim.py --csv     # also (re)write synthetic_waves/*.csv
```

Headline finding (RMS heading error vs. truth): a **naive tilt compass is
unusable in beam/quartering seas** (5–14° RMS, 30–41° peaks — the magnetic-dip
amplification of wave-tilt error), while the **fused yaw + stabilizer holds
~2–5° RMS** even in rough (Hs 4 m) and high (Hs 8.5 m) seas. Beam and quartering
seas are the worst case; head/following seas put the motion into pitch, which
barely affects heading.

**Sources / further reading:**
- [bareboat-math (spectra, sea states, estimation)](https://bareboat-necessities.github.io/my-bareboat/bareboat-math.html)
- [NTNU — Sea state parameters & engineering wave spectra](https://oivarn.folk.ntnu.no/hercules_ntnu/LWTcourse/partB/3seastate/3%20SEA%20STATE%20PARAMETERS%20AND%20ENGINEERING%20WAVE%20SPECTRA.htm)
- [US Naval Academy EN400 — Seakeeping (Ch. 8)](https://usna.edu/NAOE/_files/documents/Courses/EN400/02.08%20Chapter%208.pdf)
- [Pierson–Moskowitz spectrum overview](https://www.sciencedirect.com/topics/engineering/pierson-moskowitz-spectrum) · [Response Amplitude Operator overview](https://www.sciencedirect.com/topics/engineering/response-amplitude-operator)
- [Wave-Induced Loads and Fatigue Life of Small Vessels Under Complex Sea States (MDPI 2025)](https://www.mdpi.com/2077-1312/13/10/1920)

## Working without hardware

The parser and decoder are pure functions, and any object with `read`/`write`
is a valid transport. Replay a capture or unit-test the full stack:

```python
from hwt901b import HWT901B, BytesTransport
imu = HWT901B(BytesTransport(open("capture.bin", "rb").read()))
print(imu.read_state().angle)
```

## Protocol reference

Data frames are 11 bytes: `0x55`, a type byte, four little-endian signed int16
values, and a `sum(bytes[0:10]) & 0xFF` checksum.

| Type | Meaning | Units after decoding |
|------|---------|----------------------|
| `0x50` | Time | Y/M/D H:M:S.ms |
| `0x51` | Acceleration | g (+ die °C) |
| `0x52` | Angular velocity | °/s |
| `0x53` | Angle | ° (roll/pitch/yaw) + fw version |
| `0x54` | Magnetic field | raw counts (+ die °C) |
| `0x56` | Pressure / altitude | Pa / cm |
| `0x59` | Quaternion | normalized w,x,y,z |
| `0x5F` | Register read return | four int16 words |

Scaling: accel `= raw/32768 × 16 g`, gyro `= raw/32768 × 2000 °/s`, angle
`= raw/32768 × 180°`, temperature `= raw/100 °C`.

Commands are 5 bytes: `0xFF 0xAA <reg> <lo> <hi>`. Unlock is
`FF AA 69 88 B5`; save is `FF AA 00 00 00`.

Sources:
- [WIT Standard Communication Protocol](https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/wit-standard-communication-protocol)
- [WitMotion Python SDK](https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/sdk/python_sdk-quick-start)
- HWT901B-TTL datasheet v20-0707

## Tests

```bash
python -m pytest        # 14 hardware-free tests (framing, decode, calibration)
```

## License

MIT
