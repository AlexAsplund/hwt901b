"""End-to-end offline magnetometer calibration.

Collects raw magnetometer samples while you tumble the sensor, fits a full
hard-iron + soft-iron ellipsoid correction (pure Python, no numpy), reports the
fit quality, and shows how to apply the correction to live readings.

    python examples/calibrate_magnetometer.py COM3
"""

import sys
import time

from hwt901b import HWT901B, RswBit, fit_ellipsoid, heading_from_magnetic


def collect(imu: HWT901B, n: int, timeout: float):
    imu.set_outputs(RswBit.ACCELERATION, RswBit.ANGLE,
                    RswBit.ANGULAR_VELOCITY, RswBit.MAGNETIC)
    time.sleep(0.2)
    imu.state.magnetic = None
    samples = []
    deadline = time.monotonic() + timeout
    print(f"Tumble the sensor slowly through every orientation "
          f"({n} samples)...")
    while len(samples) < n and time.monotonic() < deadline:
        imu.poll()
        m = imu.state.magnetic
        if m is not None:
            samples.append((m.x, m.y, m.z))
            imu.state.magnetic = None
            if len(samples) % 25 == 0:
                print(f"  {len(samples)} / {n}")
        else:
            time.sleep(0.005)
    return samples


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    with HWT901B.open(port) as imu:
        samples = collect(imu, n=600, timeout=60)
        if len(samples) < 50:
            print("Not enough samples -- check the magnetometer is enabled.")
            return

        cal = fit_ellipsoid(samples)
        print("\nhard-iron offset:", cal.hard_iron_int())
        print("soft-iron matrix:")
        for row in cal.soft_iron:
            print("  [{:9.5f} {:9.5f} {:9.5f}]".format(*row))
        print(f"field strength : {cal.field_strength:.1f}")
        print(f"RMS residual   : {cal.residual(samples):.2f} "
              f"({100 * cal.residual(samples) / cal.field_strength:.2f}%)")

        # Optionally push hard-iron offsets to the module (soft-iron stays in SW):
        # hx, hy, hz = cal.hard_iron_int()
        # imu.set_magnetic_offsets(hx, hy, hz)

        print("\nLive calibrated heading (Ctrl-C to stop):")
        try:
            for state in imu.stream(min_interval=0.2):
                if state.magnetic:
                    raw = (state.magnetic.x, state.magnetic.y, state.magnetic.z)
                    cal_vec = cal.apply(raw)
                    print(f"  heading ~= {heading_from_magnetic(cal_vec):6.1f} deg")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
