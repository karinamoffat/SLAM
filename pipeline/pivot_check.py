#!/usr/bin/env python3
"""Measure how the car really translates during braked pivots.

Usage (on the Pi):
    python3 pivot_check.py ~/replay/m8_run4.npz ~/replay/m8_run5.npz ~/replay/m8_run6.npz

Input: the .npz files written by replay_extract.py.

Idea:
  For every pivot with the car stopped before AND after it, find the car's pose
  during both stops from the markers alone (no motion model), with the heading
  change between them taken from the gyro. That gives the real swing of
  base_link during the pivot, in the car's frame before the pivot.

  Model: the pivot is a rotation by dtheta about a fixed point c in the car frame.
  Then the swing is linear in c:
      swing = (I - R(dtheta)) c
  so c is fitted by weighted least squares, separately for left and right pivots.
  The earlier model was c = (0, +0.07) for left pivots and (0, -0.07) for right
  pivots (the braked side's midpoint). The fit from this script replaced it
  (ekf_localize.PIVOT_C_LEFT / PIVOT_C_RIGHT).

  A pose fix needs 5 numbers per pivot (x, y, heading before; x, y after), and each
  marker seen during a stop gives 2. So a pivot is used only if the two stops see
  3 or more markers in total, and at least one stop sees 2 or more.

Repeated sightings of one marker during one stop are averaged into one
measurement with the single-sighting sigma (they share the same bias, so they
are not independent).
"""

import argparse
import os

import numpy as np

from ekf_localize import MARKER_MAP  # noqa: E402  (the tape-measured map, one copy)

MODEL_OFFSET = 0.07     # earlier model: rotation about (0, +-0.07) in base_link
K_PIVOT_NOW = 0.0016 * 4.0   # current pivot position noise, m^2 per rad (Q x4)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------
# Command segments (same classes as bag_check.py)

def classify(v, w, eps=0.01):
    if v > eps:
        return "forward"
    if v < -eps:
        return "rev pivot" if abs(w) > eps else "back"
    if w > eps:
        return "left"
    if w < -eps:
        return "right"
    return "stop"


def segments(cmd_t, cmd_v, cmd_w, t_end):
    cls = [classify(v, w) for v, w in zip(cmd_v, cmd_w)]
    segs = []
    start = 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[start]:
            t1 = cmd_t[i] if i < len(cls) else t_end
            segs.append((cls[start], cmd_t[start], t1))
            start = i
    return segs


# ----------------------------------------------------------------------------
# Pose fix from two stops

def stop_measurements(d, t0, t1):
    """Average each known marker's sightings in [t0, t1] into one (r, b) measurement."""
    m = (d["obs_t"] >= t0) & (d["obs_t"] <= t1)
    out = []
    for mid in np.unique(d["obs_id"][m]):
        mid = int(mid)
        if mid not in MARKER_MAP:
            continue
        k = m & (d["obs_id"] == mid)
        r = d["obs_range"][k]
        ok = (r >= 0.6) & (r <= 2.5)
        if not ok.any():
            continue
        r = r[ok]
        b = d["obs_bearing"][k][ok]
        sr = d["obs_sigma_r"][k][ok].mean()
        sb = d["obs_sigma_b"][k][ok].mean()
        mx, my, smx, smy = MARKER_MAP[mid]
        smap2 = 0.5 * (smx ** 2 + smy ** 2)
        rm = r.mean()
        bm = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
        out.append((mid, rm, bm, np.sqrt(sr ** 2 + smap2), np.sqrt(sb ** 2 + smap2 / rm ** 2),
                    len(r)))
    return out


def predict(pose, mid):
    px, py, th = pose
    mx, my = MARKER_MAP[mid][:2]
    dx, dy = mx - px, my - py
    r = np.hypot(dx, dy)
    return r, wrap(np.arctan2(dy, dx) - th), dx, dy


def residuals(z, meas1, meas2, dth, prior=None):
    """z = (x1, y1, th1, x2, y2); returns whitened residuals and Jacobian of the predictions.
    prior = (heading, sigma) for th1, from the gyro plus a fitted offset, or None."""
    x1, y1, th1, x2, y2 = z
    res, J = [], []
    if prior is not None:
        res.append(wrap(prior[0] - th1) / prior[1])
        jp = np.zeros(5)
        jp[2] = 1.0 / prior[1]
        J.append(jp)
    for meas, pose, cols in ((meas1, (x1, y1, th1), (0, 1)), (meas2, (x2, y2, th1 + dth), (3, 4))):
        for mid, r, b, sr, sb, _ in meas:
            rh, bh, dx, dy = predict(pose, mid)
            res.append((r - rh) / sr)
            res.append(wrap(b - bh) / sb)
            jr = np.zeros(5)
            jb = np.zeros(5)
            jr[cols[0]] = -dx / rh / sr
            jr[cols[1]] = -dy / rh / sr
            jb[cols[0]] = dy / rh ** 2 / sb
            jb[cols[1]] = -dx / rh ** 2 / sb
            jb[2] = -1.0 / sb
            J.append(jr)
            J.append(jb)
    return np.array(res), np.array(J)


def single_fix(meas):
    """Pose (x, y, th) from one stop with 2 or more markers. Returns pose, sigma_th, chi2."""
    best = None
    for th0 in np.radians(np.arange(-180, 180, 10)):
        pts = [np.array(MARKER_MAP[mid][:2]) - r * np.array([np.cos(th0 + b), np.sin(th0 + b)])
               for mid, r, b, *_ in meas]
        z = np.array([*np.mean(pts, axis=0), th0])
        for _ in range(20):
            res, J = [], []
            for mid, r, b, sr, sb, _ in meas:
                rh, bh, dx, dy = predict(z, mid)
                res += [(r - rh) / sr, wrap(b - bh) / sb]
                J += [[-dx / rh / sr, -dy / rh / sr, 0.0], [dy / rh ** 2 / sb, -dx / rh ** 2 / sb, -1.0 / sb]]
            res, J = np.array(res), np.array(J)
            try:
                step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(3), J.T @ res)
            except np.linalg.LinAlgError:
                break
            z = z + step
            z[2] = wrap(z[2])
            if np.abs(step).max() < 1e-7:
                break
        chi2 = float(res @ res)
        if best is None or chi2 < best[2] - 1e-9:
            try:
                cov = np.linalg.inv(J.T @ J)
            except np.linalg.LinAlgError:
                continue
            best = (z, np.sqrt(cov[2, 2]), chi2)
    return best


def solve_pair(meas1, meas2, dth, prior=None):
    """Gauss-Newton from a heading grid. Returns z, covariance, chi2, dof."""
    best = None
    grid = np.radians(np.arange(-180, 180, 10))
    if prior is not None:
        grid = np.concatenate([[prior[0]], grid])
    for th0 in grid:
        p = []
        for meas, th in ((meas1, th0), (meas2, th0 + dth)):
            pts = [np.array(MARKER_MAP[mid][:2]) - r * np.array([np.cos(th + b), np.sin(th + b)])
                   for mid, r, b, *_ in meas]
            p.append(np.mean(pts, axis=0))
        z = np.array([p[0][0], p[0][1], th0, p[1][0], p[1][1]])
        for _ in range(20):
            res, J = residuals(z, meas1, meas2, dth, prior)
            H = J.T @ J
            try:
                step = np.linalg.solve(H + 1e-9 * np.eye(5), J.T @ res)  # J is d(prediction)/dz
            except np.linalg.LinAlgError:
                break
            z = z + step
            z[2] = wrap(z[2])
            if np.abs(step).max() < 1e-7:
                break
        res, J = residuals(z, meas1, meas2, dth, prior)
        chi2 = float(res @ res)
        if best is None or chi2 < best[2] - 1e-9:
            try:
                cov = np.linalg.inv(J.T @ J)
            except np.linalg.LinAlgError:
                continue
            best = (z, cov, chi2, len(res) - 5)
    return best


# ----------------------------------------------------------------------------

def analyse_run(path, gyro_scale, min_stop, settle, min_angle, window):
    d = dict(np.load(path))
    name = os.path.splitext(os.path.basename(path))[0]
    t_ref = d["imu_t"][0]
    segs = segments(d["cmd_t"], d["cmd_v"], d["cmd_w"], d["imu_t"][-1])

    # gyro bias from the start still period (before the first motion command)
    moving = [s for s in segs if s[0] != "stop"]
    t_move = moving[0][1] if moving else d["imu_t"][-1]
    still = (d["imu_t"] > t_ref + 0.5) & (d["imu_t"] < t_move - 0.5)
    bias = d["imu_wz"][still].mean()
    w = gyro_scale * (d["imu_wz"] - bias)
    heading = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(d["imu_t"]))])

    def gyro_angle(ta, tb):
        return np.interp(tb, d["imu_t"], heading) - np.interp(ta, d["imu_t"], heading)

    def gyro_heading(t):
        return np.interp(t, d["imu_t"], heading)

    # Heading prior. The gyro heading is the true heading plus one unknown offset (slowly
    # drifting). Stops that see 2 or more markers give a full pose fix, so each gives the
    # offset. A stop that turned by hand (more than 10 deg while "stopped") starts a new piece.
    stops = [s for s in segs if s[0] == "stop" and s[2] - s[1] >= min_stop]
    breaks = []
    for s in stops:
        if abs(np.degrees(gyro_angle(s[1], s[2]))) > 10:
            k = (d["imu_t"] >= s[1]) & (d["imu_t"] <= s[2])
            breaks.append(d["imu_t"][k][np.argmax(np.abs(w[k]))])
    breaks = np.array(sorted(breaks))

    def piece(t):
        return int(np.searchsorted(breaks, t))

    anchors = {}
    for s in stops:
        for wa, wb in ((s[1] + settle, min(s[2], s[1] + settle + window)), (max(s[1] + settle, s[2] - window), s[2])):
            if wb - wa < 0.2:
                continue
            meas = stop_measurements(d, wa, wb)
            if len(meas) < 2:
                continue
            fix = single_fix(meas)
            if fix is None or np.degrees(fix[1]) > 3.0:
                continue
            tm = 0.5 * (wa + wb)
            anchors.setdefault(piece(tm), []).append((tm, wrap(fix[0][2] - gyro_heading(tm))))
    offsets = {}
    for p, lst in anchors.items():
        off = np.array([o for _, o in lst])
        med = np.arctan2(np.median(np.sin(off)), np.median(np.cos(off)))
        mad = np.median(np.abs(wrap(off - med)))
        offsets[p] = (med, max(1.4826 * mad, np.radians(1.0)), len(off))
    print(f"  {name}: heading pieces {len(breaks) + 1} (hand turns at "
          + (", ".join(f"{b - t_ref:.0f}" for b in breaks) or "none") + " s); anchors per piece: "
          + ", ".join(f"{p}: n {v[2]}, spread {np.degrees(v[1]):.1f} deg" for p, v in sorted(offsets.items())))

    def heading_prior(t):
        p = piece(t)
        if p not in offsets:
            return None
        med, sig, _ = offsets[p]
        return (wrap(gyro_heading(t) + med), sig)

    rows = []
    skipped = {"short stop": 0, "too few markers": 0, "small angle": 0, "poor fix": 0}
    for i in range(1, len(segs) - 1):
        kind, ta, tb = segs[i]
        if kind not in ("left", "right"):
            continue
        s0, s1 = segs[i - 1], segs[i + 1]
        if s0[0] != "stop" or s1[0] != "stop" or s0[2] - s0[1] < min_stop or s1[2] - s1[1] < min_stop:
            skipped["short stop"] += 1
            continue
        # only the end of the stop before and the start of the stop after, so a hand
        # repositioning earlier or later in a long stop is not mixed in
        w0 = (max(s0[1] + settle, s0[2] - window), s0[2])
        w1 = (s1[1] + settle, min(s1[2], s1[1] + settle + window))
        meas1 = stop_measurements(d, *w0)
        meas2 = stop_measurements(d, *w1)
        n1, n2 = len(meas1), len(meas2)
        tm0 = np.mean(w0)
        tm1 = np.mean(w1)
        prior = heading_prior(tm0) if piece(tm0) == piece(tm1) else None
        enough = (min(n1, n2) >= 1) if prior is not None else (n1 + n2 >= 3 and max(n1, n2) >= 2 and min(n1, n2) >= 1)
        if not enough:
            skipped["too few markers"] += 1
            continue
        dth = gyro_angle(tm0, tm1)
        if abs(np.degrees(dth)) < min_angle:
            skipped["small angle"] += 1
            continue
        sol = solve_pair(meas1, meas2, dth, prior)
        if sol is None:
            skipped["poor fix"] += 1
            continue
        z, cov, chi2, dof = sol
        th1 = z[2]
        R1t = rot(-th1)
        dp = np.array([z[3] - z[0], z[4] - z[1]])
        swing = R1t @ dp
        # covariance of the swing in the car frame (needs heading too, small effect)
        Jd = np.array([[-1.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, -1.0, 0.0, 0.0, 1.0]])
        Jfull = R1t @ Jd
        dR = np.array([[-np.sin(th1), np.cos(th1)], [-np.cos(th1), -np.sin(th1)]])  # d R(-th)/d th
        Jfull[:, 2] = dR @ dp
        Cs = Jfull @ cov @ Jfull.T
        sig = np.sqrt(np.diag(Cs))
        if sig.max() > 0.08:
            skipped["poor fix"] += 1
            continue
        c_model = np.array([0.0, MODEL_OFFSET if kind == "left" else -MODEL_OFFSET])
        model = (np.eye(2) - rot(dth)) @ c_model
        rows.append(dict(run=name, t=ta - t_ref, kind=kind, dth=dth, swing=swing, cov=Cs, sig=sig,
                         model=model, n1=n1, n2=n2, ids1=[m[0] for m in meas1],
                         ids2=[m[0] for m in meas2], chi2=chi2, dof=dof,
                         stops=(s0[2] - s0[1], s1[2] - s1[1])))
    return name, bias, rows, skipped


def design(r, lurch):
    """Rows of the linear model for one pivot: swing = (I - R(dth)) c [+ (L, 0)]."""
    A = np.eye(2) - rot(r["dth"])
    if lurch:
        A = np.hstack([A, np.array([[1.0], [0.0]])])
    return A


def fit_c(rows, lurch=False):
    """Weighted least squares for the pivot point c (and a forward lurch L per pivot)."""
    n = 3 if lurch else 2
    A = np.zeros((n, n))
    bvec = np.zeros(n)
    for r in rows:
        Ai = design(r, lurch)
        W = np.linalg.inv(r["cov"] + 1e-6 * np.eye(2))
        A += Ai.T @ W @ Ai
        bvec += Ai.T @ W @ r["swing"]
    C = np.linalg.inv(A)
    return C @ bvec, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--gyro-scale", type=float, default=1.03)
    ap.add_argument("--min-stop", type=float, default=0.6, help="shortest usable stop, s")
    ap.add_argument("--settle", type=float, default=0.3, help="ignore the first part of each stop, s")
    ap.add_argument("--min-angle", type=float, default=15.0, help="smallest pivot used, deg")
    ap.add_argument("--window", type=float, default=1.5,
                    help="use at most this much of each stop next to the pivot, s")
    a = ap.parse_args()

    print(f"Earlier model: rotation about (0, +{MODEL_OFFSET*100:.0f} cm) for left pivots, "
          f"(0, -{MODEL_OFFSET*100:.0f} cm) for right (x forward, y left, base_link before the pivot)")
    print(f"Gyro scale {a.gyro_scale}, stops of {a.min_stop} s or more, first {a.settle} s of each "
          f"stop ignored, at most {a.window} s of each stop used, pivots of {a.min_angle:.0f} deg or more\n")

    allrows = []
    for path in a.npz:
        name, bias, rows, skipped = analyse_run(path, a.gyro_scale, a.min_stop, a.settle, a.min_angle, a.window)
        print(f"{name}: gyro bias {np.degrees(bias):+.3f} deg/s, usable pivots {len(rows)}; skipped: "
              + ", ".join(f"{k} {v}" for k, v in skipped.items() if v))
        allrows += rows

    if not allrows:
        print("\nNo usable pivots.")
        return

    print("\nEvery usable pivot (swing of base_link in cm, in the car frame before the pivot: "
          "fwd = forward, left = left)")
    print(f"{'run':10} {'time s':>7} {'dir':5} {'angle':>7} {'markers':>15}   "
          f"{'measured fwd/left':>18} {'+-':>9}   {'model fwd/left':>15}   {'diff fwd/left':>14}  fix chi2/dof")
    for r in allrows:
        ids = f"{','.join(map(str, r['ids1']))}|{','.join(map(str, r['ids2']))}"
        s, m, sg = r["swing"] * 100, r["model"] * 100, r["sig"] * 100
        dd = s - m
        fixq = f"{r['chi2']:.1f}/{r['dof']}" if r["dof"] > 0 else "exact"
        print(f"{r['run']:10} {r['t']:7.1f} {r['kind']:5} {np.degrees(r['dth']):+7.1f} {ids:>15}   "
              f"{s[0]:+7.1f} {s[1]:+7.1f}   {sg[0]:3.1f} {sg[1]:3.1f}   {m[0]:+6.1f} {m[1]:+6.1f}   "
              f"{dd[0]:+6.1f} {dd[1]:+6.1f}   {fixq}")

    print("\nFit per direction (pivot point c in base_link, cm; x forward, y left):")
    for kind in ("left", "right"):
        rows = [r for r in allrows if r["kind"] == kind]
        c_now = MODEL_OFFSET if kind == "left" else -MODEL_OFFSET
        if len(rows) < 3:
            print(f"  {kind:5}: only {len(rows)} pivot(s), no fit")
            continue
        res_m = np.array([r["swing"] - r["model"] for r in rows])
        ang = np.array([abs(r["dth"]) for r in rows])
        print(f"  {kind:5}: n {len(rows)} pivots, {np.degrees(ang.min()):.0f} to {np.degrees(ang.max()):.0f} deg."
              f"  Model now c = (+0.0, {c_now*100:+.1f}): RMS miss fwd "
              f"{np.sqrt(np.mean(res_m[:, 0]**2))*100:.1f}, left {np.sqrt(np.mean(res_m[:, 1]**2))*100:.1f} cm")
        for lurch in (False, True):
            p, C = fit_c(rows, lurch)
            sp = np.sqrt(np.diag(C))
            res = np.array([r["swing"] - design(r, lurch) @ p for r in rows])
            chi2 = sum(e @ np.linalg.inv(r["cov"]) @ e for e, r in zip(res, rows))
            dof = 2 * len(rows) - len(p)
            extra = [(e @ e - np.trace(r["cov"])) / (2 * a_) for e, r, a_ in zip(res, rows, ang)]
            k_lo = max(np.mean(extra), 0.0)
            k_hi = np.mean([(e @ e) / (2 * a_) for e, a_ in zip(res, ang)])
            label = "B: point c + lurch L" if lurch else "A: point c only     "
            extra_txt = f", L = {p[2]*100:+.1f} +- {sp[2]*100:.1f}" if lurch else ""
            print(f"    {label}  c = ({p[0]*100:+.1f} +- {sp[0]*100:.1f}, {p[1]*100:+.1f} +- {sp[1]*100:.1f})"
                  f"{extra_txt} cm;  chi2 {chi2:.1f} / {dof} dof;  RMS miss fwd "
                  f"{np.sqrt(np.mean(res[:, 0]**2))*100:.1f}, left {np.sqrt(np.mean(res[:, 1]**2))*100:.1f} cm")
            sw90 = design(dict(dth=np.pi / 2 * (1 if kind == "left" else -1)), lurch) @ p
            print(f"       swing for a 90 deg pivot: fwd {sw90[0]*100:+.1f}, left {sw90[1]*100:+.1f} cm;  "
                  f"leftover scatter {k_lo:.4f} to {k_hi:.4f} m^2/rad "
                  f"(sigma {np.sqrt(k_lo*np.pi/2)*100:.1f} to {np.sqrt(k_hi*np.pi/2)*100:.1f} cm per 90 deg)")
    print(f"\n  Filter now: pivot noise {K_PIVOT_NOW:.4f} m^2/rad (sigma {np.sqrt(K_PIVOT_NOW*np.pi/2)*100:.1f} cm per 90 deg)."
          "\n  Leftover scatter: low end subtracts the fix error, high end counts all of it as pivot scatter."
          "\n  B beats A if its chi2 drops by much more than 1 (one extra parameter).")

if __name__ == "__main__":
    main()
