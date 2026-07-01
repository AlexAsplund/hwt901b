"""Minimal live read: open the port, print fused angles at 10 Hz.

    python examples/basic_read.py COM3            # Windows
    python examples/basic_read.py /dev/ttyUSB0    # Linux
    python examples/basic_read.py COM3 --mount vertical    # sensor bolted on edge
    python examples/basic_read.py COM3 --orient vertical   # on-chip toggle instead

If the module is not lying flat, use ``--mount`` to remap the axes in software
(e.g. ``vertical`` = former up/Z+ becomes left/Y+, or a custom ``x:z:-y`` map),
or ``--orient`` to flip the module's own on-chip install direction.
"""

import argparse

from hwt901b import HWT901B, Mount


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal HWT901B angle reader")
    ap.add_argument("port", nargs="?", default="COM3")
    ap.add_argument("baud", nargs="?", type=int, default=9600)
    ap.add_argument("--mount", default="level",
                    help="software axis remap: level (default), vertical / "
                         "z-up-to-y, upside-down, yaw-90-ccw, or 'X:Y:Z' e.g. x:z:-y")
    ap.add_argument("--orient", choices=["horizontal", "vertical"], default=None,
                    help="on-chip install direction (session only)")
    args = ap.parse_args()

    try:
        mount = Mount.parse(args.mount)
    except ValueError as exc:
        ap.error(f"--mount: {exc}")

    with HWT901B.open(args.port, baudrate=args.baud, mount=mount) as imu:
        if args.orient is not None:
            imu.set_orientation_vertical(args.orient == "vertical", save=False)
        note = mount.name if mount else (args.orient or "flat")
        print(f"Reading from {args.port} @ {args.baud} baud [{note}]. Ctrl-C to stop.")
        try:
            for state in imu.stream(min_interval=0.1):
                a = state.angle
                if a:
                    print(f"roll={a.roll:7.2f}  pitch={a.pitch:7.2f}  "
                          f"yaw={a.yaw:7.2f}")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
