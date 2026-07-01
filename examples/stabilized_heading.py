"""
Stabilized marine heading: gyro-fused yaw + magnetic disturbance gating + GPS
course-over-ground aiding.

This shows the recommended production pattern and, importantly, how to feed GPS
from *any* source. The stabilizer only wants two numbers -- course (deg true)
and speed -- so you can source them from an NMEA receiver, gpsd, an NMEA2000
gateway, a UDP feed, a phone, or a simulator. Here we support an optional NMEA
GPS on a second serial port, but the ``get_gps()`` function is the only thing
you would swap.

    python examples/stabilized_heading.py COM3
    python examples/stabilized_heading.py COM3 --gps-port COM7 --gps-baud 9600
    python examples/stabilized_heading.py COM3 --declination 5.0
    python examples/stabilized_heading.py COM3 --mount vertical   # sensor on edge
    python examples/stabilized_heading.py COM3 --orient vertical  # on-chip toggle

If the module is not mounted flat, use ``--mount`` to remap the axes in software
(any orientation, e.g. ``vertical`` puts former up/Z+ on left/Y+, or a custom
``x:z:-y`` map) or ``--orient`` to flip the module's on-chip install direction
(session-only; persist with ``hwt901b config --orient``).
"""

import argparse
import json
import os
import sys
import time

from hwt901b import (
    HWT901B, HeadingStabilizer, MagCalibration, Mount, RswBit, parse_nmea_gps)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAL = os.path.join(REPO_ROOT, "mag_calibration.json")


def load_mag_calibration(cal_path):
    """Build a MagCalibration from the json file, or (None, None) if absent."""
    if not os.path.exists(cal_path):
        return None, None
    with open(cal_path) as f:
        data = json.load(f)
    cal = MagCalibration(
        hard_iron=tuple(data.get("hard_iron", [0, 0, 0])),
        soft_iron=tuple(tuple(r) for r in
                        data.get("soft_iron", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        field_strength=data.get("field_strength", 0.0),
    )
    return cal, data.get("field_strength")


def open_gps(port, baud):
    """Open an optional NMEA GPS on a second serial port. Returns a Serial or None."""
    if not port:
        return None
    import serial
    return serial.Serial(port, baud, timeout=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM3")
    ap.add_argument("baud", nargs="?", type=int, default=9600)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--cal", default=DEFAULT_CAL)
    ap.add_argument("--gps-port", default="", help="optional NMEA GPS serial port")
    ap.add_argument("--gps-baud", type=int, default=9600)
    ap.add_argument("--deadband", type=float, default=1.0,
                    help="display hold band in deg (0 = off; default 1.0)")
    ap.add_argument("--smooth", type=float, default=0.8,
                    help="display low-pass time constant in s (0 = off)")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--mount", default="level",
                    help="software axis remap for a non-flat install: level "
                         "(default), vertical / z-up-to-y, upside-down, "
                         "yaw-90-ccw, or a custom 'X:Y:Z' map like x:z:-y")
    ap.add_argument("--orient", choices=["horizontal", "vertical"], default=None,
                    help="on-chip install direction (WitMotion ORIENT register); "
                         "applied for this session only")
    args = ap.parse_args()

    try:
        mount = Mount.parse(args.mount)
    except ValueError as exc:
        ap.error(f"--mount: {exc}")

    mag_cal, field = load_mag_calibration(args.cal)
    # The mag fit is in the sensor's own axes; if we remap the axes, rotate the
    # calibration into the same body frame so it is applied consistently.
    if mag_cal is not None and mount is not None:
        mag_cal = mag_cal.rotated(mount)
    stab = HeadingStabilizer(expected_field=field,       # learns dip/field if None
                             output_smoothing=args.smooth,
                             output_deadband=args.deadband)

    gps = open_gps(args.gps_port, args.gps_baud)
    gps_buf = ""
    last_cog = last_speed = None

    def pump_gps():
        """Drain the GPS port; update last_cog/last_speed. Swap this out for
        your own source (gpsd, UDP, CAN, ...): just set last_cog/last_speed."""
        nonlocal gps_buf, last_cog, last_speed
        if gps is None:
            return
        data = gps.read(512)
        if not data:
            return
        gps_buf += data.decode("ascii", "ignore")
        while "\n" in gps_buf:
            line, gps_buf = gps_buf.split("\n", 1)
            fix = parse_nmea_gps(line)
            if fix and fix.valid and fix.course_deg is not None:
                last_cog = fix.course_deg
                last_speed = fix.speed_knots  # knots -> keep cog_min_speed in knots

    with HWT901B.open(args.port, baudrate=args.baud, timeout=0.02,
                      mount=mount) as imu:
        imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGULAR_VELOCITY,
                        RswBit.ANGLE, RswBit.MAGNETIC)
        if args.orient is not None:
            imu.set_orientation_vertical(args.orient == "vertical", save=False)
        gps_note = f"GPS on {args.gps_port}" if gps else "no GPS (mag+gyro only)"
        mount_note = (f"mount={mount.name}" if mount else
                      f"on-chip orient={args.orient}" if args.orient else
                      "mount=flat")
        print(f"Stabilized heading | {mount_note} | {gps_note} | Ctrl-C to stop")
        print("-" * 90)

        last_t = time.monotonic()
        next_draw = 0.0
        try:
            while True:
                imu.poll()
                pump_gps()
                now = time.monotonic()
                if now < next_draw:
                    continue
                next_draw = now + 1.0 / args.rate
                dt = now - last_t
                last_t = now

                out = stab.update_from_state(
                    imu.state, dt,
                    mag_calibration=mag_cal,  # applies hard+soft iron before gating
                    cog_deg=last_cog, speed=last_speed,
                    declination_deg=args.declination,
                )
                if out is None:
                    continue
                cog_txt = (f"COG {last_cog:5.1f}@{last_speed:.1f}"
                           if last_cog is not None else "COG   --")
                flag = "COAST" if out.coasting else "     "
                sea = ("WAVES" if out.wave_level > 0.15 else
                       "swell" if out.wave_level > 0.05 else "calm ")
                sys.stdout.write(
                    f"\rheading {out.heading:6.1f} (raw {out.raw_heading:6.1f})  {flag}  "
                    f"trust {out.mag_trust:4.2f}  dip {out.dip_deg:5.1f}  "
                    f"{sea} {out.wave_level:.2f}g x{out.damping_scale:.1f}  "
                    f"cogW {out.cog_weight:4.2f}  {cog_txt}   ")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
