"""
Live dashboard: tilt-compensated heading + attitude + sensors, in a loop.

Applies your saved magnetometer calibration (hard + soft iron, in software) and
prints a refreshing single line you can watch while you move the sensor around.

Usage
-----
    python examples/live_heading.py                      # COM3, mag_calibration.json
    python examples/live_heading.py COM3 9600
    python examples/live_heading.py /dev/ttyUSB0 9600 --declination 3.0
    python examples/live_heading.py COM3 --no-cal        # skip calibration
    python examples/live_heading.py COM3 --mount vertical    # sensor bolted on edge
    python examples/live_heading.py COM3 --orient vertical   # on-chip toggle instead

Mounting: if the module is not lying flat, either remap the axes in software with
``--mount`` (any orientation; e.g. ``vertical`` = Z+ up becomes Y+ left, or a
custom ``x:z:-y`` map) or flip the module's own on-chip install direction with
``--orient`` (applied for this session only; use ``hwt901b config --orient`` to
persist it).

The calibration file defaults to ``mag_calibration.json`` in the repo root (the
one produced by the calibration run). Its ``hard_iron`` is applied in software
here, so make sure the module's own offsets are zeroed (the recommended
software-only path).
"""

import argparse
import json
import math
import os
import sys
import time

from hwt901b import HWT901B, Mount, RswBit, tilt_compensated_heading, yaw_to_heading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAL = os.path.join(REPO_ROOT, "mag_calibration.json")


def load_calibration(path):
    """Return (hard_iron[3], soft_iron[3][3], heading_offset) or Nones."""
    if not path or not os.path.exists(path):
        return None, None, 0.0
    with open(path) as f:
        data = json.load(f)
    # Support both the "written to module" file and a pure-software file.
    hard = data.get("hard_iron") or data.get("hard_iron_in_module") or [0, 0, 0]
    soft = data.get("soft_iron") or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    offset = float(data.get("heading_offset_deg", 0.0))
    return hard, soft, offset


def apply_cal(raw, hard, soft):
    c = [raw[i] - hard[i] for i in range(3)]
    return tuple(sum(soft[i][j] * c[j] for j in range(3)) for i in range(3))


def main():
    ap = argparse.ArgumentParser(description="Live HWT901B heading/attitude monitor")
    ap.add_argument("port", nargs="?", default="COM3")
    ap.add_argument("baud", nargs="?", type=int, default=9600)
    ap.add_argument("--declination", type=float, default=0.0,
                    help="local magnetic declination (deg) for true north")
    ap.add_argument("--trim", type=float, default=None,
                    help="fixed heading offset (deg); overrides the value in "
                         "the calibration file")
    ap.add_argument("--cal", default=DEFAULT_CAL, help="mag calibration json path")
    ap.add_argument("--no-cal", action="store_true", help="ignore calibration file")
    ap.add_argument("--rate", type=float, default=10.0, help="refresh Hz")
    ap.add_argument("--source", choices=["fused", "tilt"], default="fused",
                    help="primary heading: 'fused' = gyro-fused yaw "
                         "(recommended for boats/vehicles); 'tilt' = "
                         "accelerometer-based tilt compensation (static only)")
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

    if args.no_cal:
        hard, soft, offset = None, None, 0.0
    else:
        hard, soft, offset = load_calibration(args.cal)
    if args.trim is not None:      # CLI overrides the file's trim
        offset = args.trim
    total_offset = args.declination + offset
    cal_state = "software cal ON" if hard else "NO cal (raw magnetometer)"

    with HWT901B.open(args.port, baudrate=args.baud, timeout=0.05,
                      mount=mount) as imu:
        # Ensure the packets we display are actually being emitted.
        imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGULAR_VELOCITY,
                        RswBit.ANGLE, RswBit.MAGNETIC)
        if args.orient is not None:
            imu.set_orientation_vertical(args.orient == "vertical", save=False)
        time.sleep(0.2)

        mount_note = (f"mount={mount.name}" if mount else
                      f"on-chip orient={args.orient}" if args.orient else
                      "mount=flat")
        print(f"HWT901B @ {args.port} {args.baud}baud | {cal_state} | {mount_note} | "
              f"heading offset {total_offset:+g}deg "
              f"(decl {args.declination:g} + trim {offset:g}) | "
              f"primary=[{args.source}] | Ctrl-C to stop")
        print("head = primary heading; tilt/fused shown for comparison")
        print("-" * 100)

        period = 1.0 / args.rate
        next_draw = 0.0
        try:
            for state in imu.stream(min_interval=0.0):
                now = time.monotonic()
                if now < next_draw:
                    continue
                next_draw = now + period

                a = state.angle
                acc = state.acceleration
                g = state.angular_velocity
                m = state.magnetic

                # Accelerometer-based tilt compensation (static/handheld only).
                tilt_hdg = float("nan")
                if m and acc:
                    raw = (m.x, m.y, m.z)
                    cal = apply_cal(raw, hard, soft) if hard else raw
                    tilt_hdg = tilt_compensated_heading(
                        (acc.x, acc.y, acc.z), cal, total_offset)

                # Gyro-fused yaw (motion-robust: use this on a boat/vehicle).
                fused_hdg = yaw_to_heading(a.yaw, total_offset) if a else float("nan")

                primary = fused_hdg if args.source == "fused" else tilt_hdg

                roll = a.roll if a else float("nan")
                pitch = a.pitch if a else float("nan")
                temp = acc.temperature if acc else float("nan")

                line = (
                    f"\rhead {primary:6.1f}  |  "
                    f"fused {fused_hdg:6.1f} tilt {tilt_hdg:6.1f}  |  "
                    f"R {roll:6.1f} P {pitch:6.1f}  |  "
                    f"a[g] {acc.x:6.3f} {acc.y:6.3f} {acc.z:6.3f}  |  "
                    if acc else "\r(waiting for data)"
                )
                if g:
                    line += f"w[/s] {g.x:6.1f} {g.y:6.1f} {g.z:6.1f}  |  "
                line += f"{temp:4.1f}C   "
                sys.stdout.write(line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
