#!/usr/bin/env python3
"""Measure the car's real forward speed from marker ranges.

Usage (on the Pi):
    python3 speed_check.py ~/replay/m7_run1.npz

Method: during each straight (forward or back) command segment, take the
sightings of any marker roughly ahead (|bearing| < 20 deg) and fit a line to
range versus time. Driving toward a marker shrinks its range at the car's
speed, so

    speed = -slope / cos(mean bearing)

The first 0.2 s of each segment (spin-up) is skipped, so this is the cruise
speed. Segments need at least 4 sightings of the same marker over 0.5 s or more.
"""

import argparse
import os

import numpy as np

import ekf_localize as E

SKIP_START = 0.2           # s, spin-up
MAX_BEARING = np.radians(20.0)
MIN_PTS = 4
MIN_SPAN = 0.5             # s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()
    d = np.load(os.path.expanduser(args.npz))

    imu_t = d["imu_t"]
    v, w, mode = E.command_per_sample(imu_t, d["cmd_t"], d["cmd_v"], d["cmd_w"])
    t0 = imu_t[0]

    o_t, o_id, o_r, o_b = d["obs_t"], d["obs_id"], d["obs_range"], d["obs_bearing"]
    ok = (np.isin(o_id, list(E.MARKER_MAP)) & (o_r >= E.MIN_RANGE) & (o_r <= 2.5)
          & (np.abs(o_b) < MAX_BEARING))

    fwd = mode == E.FORWARD
    segs = []
    i = 0
    while i < len(fwd):
        if fwd[i]:
            k = i
            while k + 1 < len(fwd) and fwd[k + 1]:
                k += 1
            segs.append((imu_t[i], imu_t[k], np.sign(v[i])))
            i = k + 1
        else:
            i += 1

    rows = []
    print("  start   dur   dir    ID    n   range from -> to    speed     fit resid")
    for ts, te, sgn in segs:
        sel_t = ok & (o_t >= ts + SKIP_START) & (o_t <= te)
        for mid in sorted(set(o_id[sel_t])):
            s = sel_t & (o_id == mid)
            tt, rr, bb = o_t[s], o_r[s], o_b[s]
            if len(tt) < MIN_PTS or tt[-1] - tt[0] < MIN_SPAN:
                continue
            A = np.vstack([tt - tt[0], np.ones_like(tt)]).T
            (slope, icpt), *_ = np.linalg.lstsq(A, rr, rcond=None)
            resid = np.std(rr - A @ np.array([slope, icpt]))
            speed = -slope / np.cos(np.mean(bb)) * sgn
            rows.append((ts - t0, te - ts, sgn, mid, len(tt), speed))
            print(f"  {ts - t0:6.1f}  {te - ts:4.1f}  {'fwd' if sgn > 0 else 'back'}"
                  f"  {mid:4d}  {len(tt):3d}   {rr[0]:5.2f} -> {rr[-1]:5.2f} m"
                  f"   {speed:5.3f} m/s   {100 * resid:4.1f} cm")

    if not rows:
        print("No usable segments.")
        return
    R = np.array(rows)
    sp = R[:, 5]
    f = R[:, 2] > 0
    print(f"\nSegments used: {len(R)} ({int(f.sum())} forward, {int((~f).sum())} back)")
    for label, m in (("forward", f), ("back", ~f), ("all", np.ones_like(f))):
        if m.any():
            print(f"  {label:7s}: median {np.median(sp[m]):.3f} m/s, mean {sp[m].mean():.3f},"
                  f" sd {sp[m].std():.3f}, n {int(m.sum())}")
    print(f"  (filter default: {E.SPEED_DEFAULT} m/s)")


if __name__ == "__main__":
    main()
