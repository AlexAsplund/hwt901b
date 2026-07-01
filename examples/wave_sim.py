"""
Research-grounded wave simulation of heading stabilization (no hardware needed).

Instead of a single sine, this synthesizes a realistic *irregular* sea from a
JONSWAP spectrum and drives a small-boat motion model, then generates the raw
IMU signals a body-fixed HWT901B would see and runs them through the actual
library (naive tilt heading vs. device fused yaw vs. HeadingStabilizer).

Physics (textbook / cited in the README research notes):
  * JONSWAP spectrum:
        S(w) = (5/16) Hs^2 wp^4 w^-5 exp(-1.25 (wp/w)^4) * g^r * (1-0.287 ln g)
        r = exp(-(w-wp)^2 / (2 s^2 wp^2)),  s=0.07 (w<=wp) else 0.09,  g=3.3
  * irregular surface = sum of sinusoids, a_i = sqrt(2 S(w_i) dw), random phase
  * deep-water dispersion: k = w^2 / g,  wave slope = k * a
  * encounter frequency: we = w - (w^2/g) U cos(mu)     (U=0 here: drifting)
  * roll excited by wave slope in beam seas, resonant near the boat's natural
    roll frequency (small craft ~4 s, lightly damped -> big resonant rolls);
    pitch excited in head/following seas; heave follows long waves.

Sea states use the bareboat-math validation set (Hs, Tp):
    W1 0.27m/3.0s  W2 1.50m/5.7s  W3 4.00m/8.5s  W4 8.50m/11.4s
plus directional spreading and a confused cross-sea (two systems).

    python examples/wave_sim.py            # print comparison table
    python examples/wave_sim.py --csv      # also write CSVs to synthetic_waves/
"""

import argparse
import math
import os
import random

from hwt901b import HeadingStabilizer
from hwt901b.stabilizer import angle_diff, wrap360

G = 9.80665
FIELD = 3280.0
DIP = 65.0
DT = 0.05                 # 20 Hz
DUR = 120.0               # s per scenario (long enough for irregular statistics)
GYRO_BIAS = 0.4           # deg/s residual gyro bias the module fusion corrects

# small-boat response params
T_ROLL = 4.0              # natural roll period (s) -- lively small craft
Z_ROLL = 0.08             # roll damping ratio (low -> strong resonance)
T_PITCH = 2.8             # natural pitch period (s)
Z_PITCH = 0.25            # pitch better damped
ROLL_CAP = 45.0
PITCH_CAP = 30.0


# --- spectrum & sea synthesis ----------------------------------------------
def jonswap(w, hs, tp, gamma=3.3):
    if w <= 0:
        return 0.0
    wp = 2 * math.pi / tp
    sig = 0.07 if w <= wp else 0.09
    r = math.exp(-((w - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
    pm = (5.0 / 16.0) * hs ** 2 * wp ** 4 / w ** 5 * math.exp(-1.25 * (wp / w) ** 4)
    return pm * gamma ** r * (1 - 0.287 * math.log(gamma))


def sea_components(hs, tp, main_dir_deg, spread_deg, rng, n=60):
    """Return list of wave components: (w, amp, phase, dir_rad)."""
    w_lo, w_hi = 0.2, 3.0
    dw = (w_hi - w_lo) / n
    comps = []
    for i in range(n):
        w = w_lo + (i + 0.5) * dw
        a = math.sqrt(max(0.0, 2 * jonswap(w, hs, tp) * dw))
        if a < 1e-4:
            continue
        phase = rng.uniform(0, 2 * math.pi)
        # cos^2 directional spread around the main direction
        off = rng.uniform(-spread_deg, spread_deg) * abs(rng.uniform(-1, 1))
        d = math.radians(main_dir_deg + off)
        comps.append((w, a, phase, d))
    return comps


def resonant_rao(we, tn, zeta):
    """2nd-order magnitude, normalized to 1 at low freq, 1/(2z) at resonance."""
    wn = 2 * math.pi / tn
    x = we / wn
    return 1.0 / math.sqrt((1 - x * x) ** 2 + (2 * zeta * x) ** 2)


def heave_rao(w):
    """Boat heaves with long waves, attenuates short ones (~boat length)."""
    return 1.0 / math.sqrt(1 + (w / 1.4) ** 4)


# --- IMU synthesis (reused geometry) ---------------------------------------
def matvec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _Rx(a):
    c, s = math.cos(a), math.sin(a); return [[1, 0, 0], [0, c, s], [0, -s, c]]


def _Ry(a):
    c, s = math.cos(a), math.sin(a); return [[c, 0, -s], [0, 1, 0], [s, 0, c]]


def _Rz(a):
    c, s = math.cos(a), math.sin(a); return [[c, s, 0], [-s, c, 0], [0, 0, 1]]


def synth(roll, pitch, yaw_compass, a_lin_ned):
    R = matmul(_Rx(math.radians(roll)),
               matmul(_Ry(math.radians(pitch)), _Rz(math.radians(yaw_compass))))
    d = math.radians(DIP)
    mag = matvec(R, [FIELD * math.cos(d), 0.0, FIELD * math.sin(d)])
    acc = [c / G for c in matvec(R, [a_lin_ned[0], a_lin_ned[1], a_lin_ned[2] - G])]
    return mag, acc


def naive_heading(mag, acc):
    gx, gy, gz = -acc[0], -acc[1], -acc[2]
    roll = math.atan2(gy, gz)
    pitch = math.atan2(-gx, math.sqrt(gy * gy + gz * gz))
    mx, my, mz = mag
    sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
    xh = mx * cp + my * sp * sr + mz * sp * cr
    yh = my * cr - mz * sr
    return math.degrees(math.atan2(-yh, xh)) % 360.0


# --- one scenario ----------------------------------------------------------
def simulate(comps, true_h=90.0, csv_path=None):
    stab = HeadingStabilizer(expected_field=FIELD, expected_dip_deg=DIP,
                             rate_of_turn_sign=1.0, auto_rate_sign=False,
                             output_smoothing=0.8, output_deadband=1.0)
    module = true_h
    prev_true = true_h
    prev_mod = true_h
    rows = []
    en, em, es = [], [], []
    roll_pk = pitch_pk = acc_pk = 0.0
    min_trust = 1.0
    t = 0.0
    steps = int(DUR / DT)
    for _ in range(steps):
        roll = pitch = yaw_w = 0.0
        az = ax = ay = 0.0
        for (w, a, ph, d) in comps:
            k = w * w / G
            slope = k * a                      # wave slope amplitude (rad)
            ang = w * t + ph
            c = math.cos(ang)
            roll += math.degrees(slope * math.sin(d) *
                                 resonant_rao(w, T_ROLL, Z_ROLL)) * c
            pitch += math.degrees(slope * math.cos(d) *
                                  resonant_rao(w, T_PITCH, Z_PITCH)) * c
            yaw_w += math.degrees(slope * math.sin(2 * d) * 0.5) * c
            hv = a * heave_rao(w)
            az += w * w * hv * math.cos(ang)          # vertical accel (heave)
            ah = w * w * a * heave_rao(w)             # horizontal orbital accel
            ax += ah * math.cos(d) * math.cos(ang)    # surge
            ay += ah * math.sin(d) * math.cos(ang)    # sway
        roll = max(-ROLL_CAP, min(ROLL_CAP, roll))
        pitch = max(-PITCH_CAP, min(PITCH_CAP, pitch))
        true_now = true_h + yaw_w
        mag_b, acc_b = synth(roll, pitch, true_now, (ax, ay, az))

        nh = naive_heading(mag_b, acc_b)
        true_rate = angle_diff(true_now, prev_true) / DT
        module = wrap360(module + (true_rate + GYRO_BIAS) * DT)
        module = wrap360(module + (DT / 3.0) * angle_diff(nh, module))
        mod_rate = angle_diff(module, prev_mod) / DT
        out = stab.update(-module, DT, rate_of_turn_dps=mod_rate,
                          mag=mag_b, gravity=acc_b)

        amag = math.sqrt(acc_b[0] ** 2 + acc_b[1] ** 2 + acc_b[2] ** 2)
        roll_pk = max(roll_pk, abs(roll)); pitch_pk = max(pitch_pk, abs(pitch))
        acc_pk = max(acc_pk, amag)
        min_trust = min(min_trust, out.mag_trust)
        en.append(angle_diff(nh, true_now))
        em.append(angle_diff(module, true_now))
        es.append(angle_diff(out.heading, true_now))
        if csv_path is not None:
            rows.append([round(t, 3), round(roll, 2), round(pitch, 2),
                         round(true_now, 2), round(amag, 3), round(nh, 2),
                         round(module, 2), round(out.heading, 2),
                         round(out.mag_trust, 3)])
        prev_true = true_now
        prev_mod = module
        t += DT

    if csv_path is not None:
        import csv
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t", "roll", "pitch", "true_hdg", "acc_g",
                         "naive_hdg", "module_hdg", "stab_hdg", "mag_trust"])
            wr.writerows(rows)
    s = int(15 / DT)
    band = lambda e: max(abs(min(e[s:])), abs(max(e[s:])))
    rms = lambda e: math.sqrt(sum(x * x for x in e[s:]) / len(e[s:]))
    return dict(roll_pk=roll_pk, pitch_pk=pitch_pk, acc_pk=acc_pk,
                min_trust=min_trust,
                n_band=band(en), n_rms=rms(en), m_band=band(em), m_rms=rms(em),
                s_band=band(es), s_rms=rms(es))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true",
                    help="also write per-scenario CSVs to synthetic_waves/")
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "synthetic_waves")
    if args.csv:
        os.makedirs(outdir, exist_ok=True)

    seas = [  # name, Hs (m), Tp (s)
        ("W1 smooth", 0.27, 3.0),
        ("W2 moderate", 1.50, 5.7),
        ("W3 rough", 4.00, 8.5),
        ("W4 high", 8.50, 11.4),
    ]
    dirs = [("head", 0), ("bow-qtr", 45), ("beam", 90),
            ("stern-qtr", 135), ("following", 180)]

    print(f"Small boat (roll Tn={T_ROLL}s z={Z_ROLL}), dip={DIP}, drifting (U=0)")
    print("worst-case | RMS heading error vs true (deg). "
          "naive=tilt compass, module=fused yaw, stab=HeadingStabilizer\n")
    print(f"{'sea / dir':22s}| roll pk pitch pk |a|pk | "
          f"{'naive':>11}| {'module':>11}| {'stab':>11}| trust")
    print("-" * 104)
    for sname, hs, tp in seas:
        for dname, ddeg in dirs:
            rng = random.Random(hash((sname, dname)) & 0xffffffff)
            comps = sea_components(hs, tp, ddeg, 30.0, rng)
            csvp = (os.path.join(outdir, f"{sname.split()[0]}_{dname}.csv")
                    if (args.csv and dname == "beam") else None)
            r = simulate(comps, csv_path=csvp)
            print(f"{sname+'/'+dname:22s}| {r['roll_pk']:5.1f}d {r['pitch_pk']:6.1f}d "
                  f"{r['acc_pk']:4.1f}g | "
                  f"{r['n_band']:5.1f}/{r['n_rms']:4.1f} | "
                  f"{r['m_band']:5.1f}/{r['m_rms']:4.1f} | "
                  f"{r['s_band']:5.1f}/{r['s_rms']:4.1f} | {r['min_trust']:.2f}")
        print()

    # Confused cross-sea: beam swell (W3) + quartering wind-wave (W2), summed.
    rng = random.Random(4242)
    comps = (sea_components(4.0, 9.0, 90, 20, rng)
             + sea_components(1.5, 5.0, 135, 25, rng))
    csvp = os.path.join(outdir, "confused_cross_sea.csv") if args.csv else None
    r = simulate(comps, csv_path=csvp)
    print(f"{'confused cross-sea':22s}| {r['roll_pk']:5.1f}d {r['pitch_pk']:6.1f}d "
          f"{r['acc_pk']:4.1f}g | {r['n_band']:5.1f}/{r['n_rms']:4.1f} | "
          f"{r['m_band']:5.1f}/{r['m_rms']:4.1f} | {r['s_band']:5.1f}/{r['s_rms']:4.1f} "
          f"| {r['min_trust']:.2f}")
    if args.csv:
        print(f"\nCSVs written to {outdir}/")


if __name__ == "__main__":
    main()
