"""
Command-line interface: ``python -m hwt901b``.

Subcommands
-----------
    monitor    live-print the fused angle / accel / gyro / mag
    calibrate  run on-board accel or magnetic calibration
    magfit     collect samples and compute an offline hard/soft-iron fit
    config     set output rate / baud / which packets are emitted
    info       read back firmware version and key registers

Only ``import hwt901b`` is required; pyserial is pulled in lazily when a port is
opened. Uses stdlib ``argparse`` only.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import nmea as N
from . import protocol as P
from .calibration import fit_ellipsoid, fit_hard_iron, heading_from_magnetic
from .gps import parse_nmea_gps
from .sensor import HWT901B


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("port", help="serial port, e.g. COM3 or /dev/ttyUSB0")
    sp.add_argument("-b", "--baud", type=int, default=9600,
                    help="baud rate (default: 9600)")


def _add_mount(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--mount", default="level",
        help="mounting orientation remap (software): level (default), vertical / "
             "z-up-to-y, upside-down, yaw-90-ccw, or a custom 'X:Y:Z' axis map "
             "like 'x:z:-y' naming the sensor axis for each body axis")


def _parse_mount(spec: str):
    """Turn a --mount string into a Mount, or None for the flat default."""
    from .mount import Mount
    try:
        return Mount.parse(spec)
    except ValueError as exc:
        raise SystemExit(f"--mount: {exc}")


def _load_mag_cal(path):
    """Load a mag calibration JSON. Returns (MagCalibration, field) or (None, None)."""
    import json
    import os
    from .calibration import MagCalibration
    if not path or not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    field = data.get("field_strength")
    mag_cal = MagCalibration(
        hard_iron=tuple(data.get("hard_iron", [0, 0, 0])),
        soft_iron=tuple(tuple(r) for r in data.get(
            "soft_iron", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        field_strength=field or 0.0)
    return mag_cal, field


def cmd_monitor(args: argparse.Namespace) -> int:
    with HWT901B.open(args.port, baudrate=args.baud,
                      mount=_parse_mount(args.mount)) as imu:
        try:
            for state in imu.stream(min_interval=1.0 / args.rate):
                parts = []
                if state.angle:
                    a = state.angle
                    parts.append(
                        f"R{a.roll:7.2f} P{a.pitch:7.2f} Y{a.yaw:7.2f}")
                if state.acceleration:
                    ac = state.acceleration
                    parts.append(
                        f"| a[g] {ac.x:6.3f} {ac.y:6.3f} {ac.z:6.3f}")
                    parts.append(f"| {ac.temperature:4.1f}C")
                if state.angular_velocity:
                    g = state.angular_velocity
                    parts.append(
                        f"| w[/s] {g.x:7.2f} {g.y:7.2f} {g.z:7.2f}")
                if state.magnetic:
                    m = state.magnetic
                    parts.append(
                        f"| m {m.x:7.0f} {m.y:7.0f} {m.z:7.0f}")
                sys.stdout.write("\r" + " ".join(parts) + "   ")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        if args.what == "accel":
            print("Keep the sensor STILL and LEVEL...")
            imu.calibrate_acceleration()
            print("Accelerometer calibration saved.")
        elif args.what == "mag":
            print("Rotate the sensor slowly through ALL orientations...")

            def progress(f: float) -> None:
                bar = "#" * int(f * 30)
                sys.stdout.write(f"\r[{bar:<30}] {f*100:3.0f}%")
                sys.stdout.flush()

            imu.calibrate_magnetic(rotate_seconds=args.seconds,
                                   progress=progress)
            print("\nMagnetic calibration saved.")
    return 0


def cmd_magfit(args: argparse.Namespace) -> int:
    """Collect raw magnetometer samples and compute an offline fit."""
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        # Make sure magnetic output is enabled.
        imu.set_outputs(P.RswBit.ACCELERATION, P.RswBit.ANGLE,
                        P.RswBit.ANGULAR_VELOCITY, P.RswBit.MAGNETIC)
        time.sleep(0.2)
        imu.state.magnetic = None
        samples = []
        print(f"Collecting {args.samples} samples -- tumble the sensor slowly "
              "through every orientation...")
        deadline = time.monotonic() + args.timeout
        last_len = 0
        while len(samples) < args.samples and time.monotonic() < deadline:
            imu.poll()
            m = imu.state.magnetic
            if m is not None:
                samples.append((m.x, m.y, m.z))
                imu.state.magnetic = None
                if len(samples) - last_len >= 10:
                    last_len = len(samples)
                    sys.stdout.write(f"\r  {len(samples)} samples")
                    sys.stdout.flush()
            else:
                time.sleep(0.005)
        print(f"\r  collected {len(samples)} samples")

        if len(samples) < 20:
            print("Too few samples collected; is the magnetometer output on?")
            return 1

        cal = (fit_hard_iron if args.hard_iron_only else fit_ellipsoid)(samples)
        rms = cal.residual(samples)
        print("\n--- Magnetometer calibration ---")
        print(f"hard-iron offset : {cal.hard_iron_int()}")
        print("soft-iron matrix :")
        for row in cal.soft_iron:
            print("    [{:9.5f} {:9.5f} {:9.5f}]".format(*row))
        print(f"field strength   : {cal.field_strength:.1f} raw units")
        print(f"fit RMS residual : {rms:.1f} "
              f"({100*rms/cal.field_strength:.2f}% of field)")

        if args.write_offsets:
            hx, hy, hz = cal.hard_iron_int()
            imu.set_magnetic_offsets(hx, hy, hz)
            print(f"\nWrote hard-iron offsets to the module and saved.")
            if not args.hard_iron_only:
                print("Note: the module stores only hard-iron offsets; apply "
                      "the soft-iron matrix in your own code for full accuracy.")
    return 0


def cmd_swing(args: argparse.Namespace) -> int:
    """Marine compass swing: calibrate the magnetometer mounted in the boat."""
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        print("COMPASS SWING -- sensor must be MOUNTED in its final position.")
        print(f"Slowly turn the WHOLE BOAT through 360 deg (twice if you can) "
              f"during the next {args.seconds:.0f}s...")

        def progress(f: float) -> None:
            bar = "#" * int(f * 30)
            sys.stdout.write(f"\r[{bar:<30}] {f*100:3.0f}%")
            sys.stdout.flush()

        imu.compass_swing(seconds=args.seconds, progress=progress)
        print("\nCompass swing saved to the module.")
        print("Tip: verify yaw against a known bearing and fold any constant "
              "offset into your declination.")
    return 0


def cmd_marine_setup(args: argparse.Namespace) -> int:
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        rate = {5: P.OutputRate.HZ_5, 10: P.OutputRate.HZ_10,
                20: P.OutputRate.HZ_20, 50: P.OutputRate.HZ_50}[args.rate]
        bw = {5: P.Bandwidth.HZ_5, 10: P.Bandwidth.HZ_10, 20: P.Bandwidth.HZ_20,
              42: P.Bandwidth.HZ_42}[args.bandwidth]
        imu.configure_marine(rate=rate, bandwidth=bw)
        print(f"Marine config saved: 9-axis absolute heading, "
              f"{args.bandwidth} Hz filter, {args.rate} Hz output, "
              f"accel/gyro/angle/mag enabled.")
        print("Next: mount the sensor, then run 'hwt901b swing' and turn the boat.")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        if args.rate is not None:
            rate = {0.2: P.OutputRate.HZ_0_2, 0.5: P.OutputRate.HZ_0_5,
                    1: P.OutputRate.HZ_1, 2: P.OutputRate.HZ_2,
                    5: P.OutputRate.HZ_5, 10: P.OutputRate.HZ_10,
                    20: P.OutputRate.HZ_20, 50: P.OutputRate.HZ_50,
                    100: P.OutputRate.HZ_100, 125: P.OutputRate.HZ_125,
                    200: P.OutputRate.HZ_200}[args.rate]
            imu.set_output_rate(rate)
            print(f"output rate -> {args.rate} Hz")
        if args.set_baud is not None:
            imu.set_baudrate(P.BaudRate.from_bps(args.set_baud))
            print(f"module baud -> {args.set_baud} "
                  "(reconnect at the new rate next time)")
        if args.six_axis is not None:
            imu.set_algorithm(six_axis=args.six_axis)
            print(f"algorithm -> {'6-axis' if args.six_axis else '9-axis'}")
        if args.orient is not None:
            imu.set_orientation_vertical(args.orient == "vertical")
            print(f"on-chip install direction -> {args.orient}")
        if args.factory_reset:
            imu.factory_reset()
            print("factory reset + save done")
    return 0


def cmd_nmea(args: argparse.Namespace) -> int:
    """Stream NMEA 0183 heading/attitude sentences to stdout (and optional UDP).

    By default the heading is *stabilized* (magnetic-disturbance gating + optional
    GPS COG aiding); pass --no-stabilize to emit the raw module fused yaw instead.
    """
    import time

    sock = None
    addr = None
    if args.udp:
        import socket
        host, port = args.udp.rsplit(":", 1)
        addr = (host, int(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sys.stderr.write(f"broadcasting NMEA UDP -> {host}:{port}\n")

    include = args.include.split(",") if args.include else N.DEFAULT_INCLUDE

    def emit(lines):
        for line in lines:
            sys.stdout.write(line)
            if sock is not None:
                sock.sendto(line.encode("ascii"), addr)
        sys.stdout.flush()

    # --- raw fused-yaw mode -------------------------------------------------
    if args.no_stabilize:
        with HWT901B.open(args.port, baudrate=args.baud,
                          mount=_parse_mount(args.mount)) as imu:
            try:
                for state in imu.stream(min_interval=1.0 / args.rate):
                    emit(N.sentences_from_state(
                        state, variation_deg=args.variation, include=include))
            except KeyboardInterrupt:
                pass
        return 0

    # --- stabilized mode (default) -----------------------------------------
    from .stabilizer import HeadingStabilizer

    mag_cal, field = _load_mag_cal(args.cal)
    mount = _parse_mount(args.mount)
    if mag_cal is not None and mount is not None:
        mag_cal = mag_cal.rotated(mount)   # match the axis remap
    stab = HeadingStabilizer(expected_field=field,
                             output_smoothing=args.smooth,
                             output_deadband=args.deadband)

    gps = None
    if args.gps_port:
        import serial
        gps = serial.Serial(args.gps_port, args.gps_baud, timeout=0)
    gps_buf = ""
    cog = spd = None

    try:
        with HWT901B.open(args.port, baudrate=args.baud, timeout=0.02,
                          mount=mount) as imu:
            imu.set_outputs(P.RswBit.ACCELERATION, P.RswBit.ANGULAR_VELOCITY,
                            P.RswBit.ANGLE, P.RswBit.MAGNETIC)
            last_t = time.monotonic()
            next_emit = 0.0
            try:
                while True:
                    imu.poll()
                    if gps is not None:
                        d = gps.read(512)
                        if d:
                            gps_buf += d.decode("ascii", "ignore")
                            while "\n" in gps_buf:
                                line, gps_buf = gps_buf.split("\n", 1)
                                f = parse_nmea_gps(line)
                                if f and f.valid and f.course_deg is not None:
                                    cog, spd = f.course_deg, f.speed_knots
                    now = time.monotonic()
                    dt = now - last_t
                    last_t = now
                    out = stab.update_from_state(
                        imu.state, dt, mag_calibration=mag_cal,
                        cog_deg=cog, speed=spd, declination_deg=args.variation)
                    if out is not None and now >= next_emit:
                        next_emit = now + 1.0 / args.rate
                        st = imu.state
                        # Gyro Z is CCW-positive; NMEA ROT is negative for a
                        # turn to port (CCW), so negate.
                        rot_dpm = (-st.angular_velocity.z * 60.0
                                   if st.angular_velocity else None)
                        roll = st.angle.roll if st.angle else None
                        pitch = st.angle.pitch if st.angle else None
                        emit(N.sentences_from_heading(
                            out.heading, variation_deg=args.variation,
                            roll_deg=roll, pitch_deg=pitch,
                            rate_of_turn_dpm=rot_dpm, include=include))
            except KeyboardInterrupt:
                pass
    finally:
        if gps is not None:
            gps.close()
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Profile real motion (calm -> small/medium waves) and recommend tuning."""
    import json
    import time
    from .profiler import WaveProfiler
    from .stabilizer import HeadingStabilizer

    mag_cal, field = _load_mag_cal(args.cal)
    mount = _parse_mount(args.mount)
    if mag_cal is not None and mount is not None:
        mag_cal = mag_cal.rotated(mount)   # match the axis remap
    prof = WaveProfiler()
    stab = HeadingStabilizer(expected_field=field)
    with HWT901B.open(args.port, baudrate=args.baud, timeout=0.02,
                      mount=mount) as imu:
        imu.set_outputs(P.RswBit.ACCELERATION, P.RswBit.ANGULAR_VELOCITY,
                        P.RswBit.ANGLE, P.RswBit.MAGNETIC)
        print(f"PROFILING for up to {args.minutes:.0f} min "
              "(Ctrl-C to stop early).")
        print("Go about a normal outing: let it sit calm, then run through "
              "whatever small/medium waves you can find (wakes, chop).")
        last = time.monotonic()
        t0 = last
        next_status = t0
        try:
            while time.monotonic() - t0 < args.minutes * 60:
                imu.poll()
                now = time.monotonic()
                dt = now - last
                last = now
                out = stab.update_from_state(imu.state, dt,
                                             mag_calibration=mag_cal)
                hdg = out.heading if out else None
                prof.feed_from_state(imu.state, hdg, dt)
                if now >= next_status:
                    next_status = now + 15.0
                    sys.stdout.write(f"\r  {int(now-t0):4d}s  samples={prof._n}"
                                     "   ")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            pass

    profile = prof.summarize()
    print("\n\n=== MOTION PROFILE ===")
    print(f"duration {profile.duration_s:.0f}s, {profile.samples} samples")
    print(f"dynamic accel (g): calm {profile.calm_intensity:.3f}  "
          f"typical {profile.typical_intensity:.3f}  "
          f"rough(seen) {profile.rough_intensity:.3f}  peak {profile.peak_intensity:.3f}")
    print(f"roll amp {profile.roll_amplitude:.1f}d  pitch amp "
          f"{profile.pitch_amplitude:.1f}d  wave period ~{profile.dominant_period_s:.1f}s")
    print(f"calm heading jitter {profile.calm_heading_jitter:.2f}d/sample")
    print("\nRecommended HeadingStabilizer settings (extrapolated to rougher seas):")
    for k, v in profile.recommended.items():
        print(f"    {k} = {v}")
    print("\nExpected adaptive damping as seas build:")
    for e in profile.extrapolation:
        print(f"    {e['regime']:16s} intensity {e['intensity_g']:.3f}g "
              f"-> x{e['damping_scale']:.1f} damping")
    with open(args.out, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    print(f"\nsaved profile -> {args.out}")
    print("use it with:  hwt901b stabilized {} --profile {}".format(
        args.port, args.out))
    return 0


def _load_profile_kwargs(path):
    """Return stabilizer kwargs from a saved profile JSON, or {} if absent."""
    import json
    import os
    from .profiler import WaveProfile
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return WaveProfile.from_dict(data).stabilizer_kwargs()


def cmd_stabilized(args: argparse.Namespace) -> int:
    """Stabilized heading: fused yaw + magnetic gating + optional NMEA GPS COG."""
    from .stabilizer import HeadingStabilizer

    mag_cal, field = _load_mag_cal(args.cal)
    mount = _parse_mount(args.mount)
    if mag_cal is not None and mount is not None:
        mag_cal = mag_cal.rotated(mount)   # match the axis remap

    kwargs = dict(output_smoothing=args.smooth, output_deadband=args.deadband)
    kwargs.update(_load_profile_kwargs(args.profile))   # profile overrides
    stab = HeadingStabilizer(expected_field=field, **kwargs)
    if args.profile:
        print(f"loaded tuning from {args.profile}")
    gps = None
    if args.gps_port:
        import serial
        gps = serial.Serial(args.gps_port, args.gps_baud, timeout=0)
    gps_buf = ""
    cog = spd = None

    try:
        with HWT901B.open(args.port, baudrate=args.baud, timeout=0.02,
                          mount=mount) as imu:
            imu.set_outputs(P.RswBit.ACCELERATION, P.RswBit.ANGULAR_VELOCITY,
                            P.RswBit.ANGLE, P.RswBit.MAGNETIC)
            last_t = time.monotonic()
            try:
                while True:
                    imu.poll()
                    if gps is not None:
                        data = gps.read(512)
                        if data:
                            gps_buf += data.decode("ascii", "ignore")
                            while "\n" in gps_buf:
                                line, gps_buf = gps_buf.split("\n", 1)
                                f = parse_nmea_gps(line)
                                if f and f.valid and f.course_deg is not None:
                                    cog, spd = f.course_deg, f.speed_knots
                    now = time.monotonic()
                    dt = now - last_t
                    last_t = now
                    out = stab.update_from_state(
                        imu.state, dt, mag_calibration=mag_cal,
                        cog_deg=cog, speed=spd, declination_deg=args.declination)
                    if out is not None:
                        flag = "COAST" if out.coasting else "     "
                        sys.stdout.write(
                            f"\rheading {out.heading:6.1f}  {flag}  "
                            f"trust {out.mag_trust:4.2f}  field {out.field_ratio:4.2f}x "
                            f" dip {out.dip_deg:5.1f}  cogW {out.cog_weight:4.2f}   ")
                        sys.stdout.flush()
                    time.sleep(1.0 / args.rate)
            except KeyboardInterrupt:
                print()
    finally:
        if gps is not None:
            gps.close()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    with HWT901B.open(args.port, baudrate=args.baud) as imu:
        try:
            ver = imu.read_register(P.Register.VERSION)
            print(f"version register (0x2E): {ver}")
        except TimeoutError:
            print("version read timed out (older firmware may not answer)")
        s = imu.read_state()
        print(f"angle       : {s.angle}")
        print(f"acceleration: {s.acceleration}")
        print(f"magnetic    : {s.magnetic}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hwt901b", description="WitMotion HWT901B-TTL command-line tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("monitor", help="live-print sensor data")
    _add_common(m)
    _add_mount(m)
    m.add_argument("-r", "--rate", type=float, default=10.0,
                   help="console refresh rate in Hz (default 10)")
    m.set_defaults(func=cmd_monitor)

    c = sub.add_parser("calibrate", help="run on-board calibration")
    _add_common(c)
    c.add_argument("what", choices=["accel", "mag"])
    c.add_argument("-s", "--seconds", type=float, default=15.0,
                   help="rotation window for magnetic calibration")
    c.set_defaults(func=cmd_calibrate)

    f = sub.add_parser("magfit", help="offline hard/soft-iron magnetometer fit")
    _add_common(f)
    f.add_argument("-n", "--samples", type=int, default=600)
    f.add_argument("-t", "--timeout", type=float, default=60.0)
    f.add_argument("--hard-iron-only", action="store_true",
                   help="min/max hard-iron fit only (no soft-iron)")
    f.add_argument("--write-offsets", action="store_true",
                   help="write the hard-iron offsets back to the module")
    f.set_defaults(func=cmd_magfit)

    w = sub.add_parser("swing", help="marine compass swing (mag cal in the boat)")
    _add_common(w)
    w.add_argument("-s", "--seconds", type=float, default=90.0,
                   help="time window to turn the boat through 360 deg")
    w.set_defaults(func=cmd_swing)

    ms = sub.add_parser("marine-setup",
                        help="apply boat-friendly config (9-axis, filtered)")
    _add_common(ms)
    ms.add_argument("--rate", type=int, choices=[5, 10, 20, 50], default=10,
                    help="output rate in Hz (default 10)")
    ms.add_argument("--bandwidth", type=int, choices=[5, 10, 20, 42], default=20,
                    help="low-pass filter cutoff in Hz (default 20)")
    ms.set_defaults(func=cmd_marine_setup)

    g = sub.add_parser("config", help="change module configuration")
    _add_common(g)
    g.add_argument("--rate", type=float,
                   choices=[0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 125, 200],
                   help="output rate in Hz")
    g.add_argument("--set-baud", type=int,
                   choices=[4800, 9600, 19200, 38400, 57600, 115200, 230400])
    g.add_argument("--six-axis", type=lambda s: s.lower() in ("1", "true", "yes"),
                   help="true=6-axis relative yaw, false=9-axis compass yaw")
    g.add_argument("--orient", choices=["horizontal", "vertical"],
                   help="on-chip install direction (WitMotion ORIENT register); "
                        "for arbitrary axis remaps use the software --mount instead")
    g.add_argument("--factory-reset", action="store_true")
    g.set_defaults(func=cmd_config)

    nm = sub.add_parser("nmea", help="stream NMEA 0183 heading/attitude sentences")
    _add_common(nm)
    nm.add_argument("-r", "--rate", type=float, default=10.0,
                    help="sentence output rate in Hz (default 10)")
    nm.add_argument("--variation", type=float, default=0.0,
                    help="magnetic variation/declination in deg (East positive)")
    nm.add_argument("--include", default="",
                    help="comma list of families: hdt,hdm,hdg,rot,xdr "
                         "(default hdt,hdm,rot,xdr)")
    nm.add_argument("--udp", default="",
                    help="also broadcast to HOST:PORT over UDP "
                         "(e.g. 255.255.255.255:10110 for OpenCPN)")
    nm.add_argument("--no-stabilize", action="store_true",
                    help="emit the raw module fused yaw instead of the "
                         "stabilized heading (stabilized is the default)")
    nm.add_argument("--cal", default="mag_calibration.json",
                    help="mag calibration json used by the stabilizer")
    nm.add_argument("--gps-port", default="",
                    help="optional NMEA GPS serial port for COG aiding")
    nm.add_argument("--gps-baud", type=int, default=9600)
    nm.add_argument("--deadband", type=float, default=0.0,
                    help="display hold band in deg (0 = off, faithful feed)")
    nm.add_argument("--smooth", type=float, default=0.0,
                    help="display low-pass time constant in s (0 = off)")
    _add_mount(nm)
    nm.set_defaults(func=cmd_nmea)

    sb = sub.add_parser("stabilized",
                        help="stabilized heading (mag gating + optional GPS COG)")
    _add_common(sb)
    sb.add_argument("--declination", type=float, default=0.0)
    sb.add_argument("--cal", default="mag_calibration.json",
                    help="mag calibration json (for field ref + soft-iron)")
    sb.add_argument("--gps-port", default="",
                    help="optional NMEA GPS serial port for COG aiding")
    sb.add_argument("--gps-baud", type=int, default=9600)
    sb.add_argument("--deadband", type=float, default=1.0,
                    help="display hold band in deg (0 = off; default 1.0)")
    sb.add_argument("--smooth", type=float, default=0.8,
                    help="display low-pass time constant in s (0 = off)")
    sb.add_argument("--profile", default="",
                    help="load tuning from a 'hwt901b profile' JSON")
    sb.add_argument("-r", "--rate", type=float, default=10.0)
    _add_mount(sb)
    sb.set_defaults(func=cmd_stabilized)

    pf = sub.add_parser("profile",
                        help="profile real motion and recommend/save tuning")
    _add_common(pf)
    pf.add_argument("-m", "--minutes", type=float, default=10.0,
                    help="max profiling duration in minutes (Ctrl-C to stop)")
    pf.add_argument("--cal", default="mag_calibration.json")
    pf.add_argument("-o", "--out", default="wave_profile.json")
    _add_mount(pf)
    pf.set_defaults(func=cmd_profile)

    i = sub.add_parser("info", help="read version and a snapshot")
    _add_common(i)
    i.set_defaults(func=cmd_info)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
