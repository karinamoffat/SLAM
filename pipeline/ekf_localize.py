#!/usr/bin/env python3
"""EKF localization against the hand-measured map.

Usage (on the Pi):
    python3 ekf_localize.py ~/replay/m7_run1.npz
    python3 ekf_localize.py ~/replay/m7_run1.npz --bias interp --gyro-scale 1.03

Input: the .npz written by replay_extract.py.
State: x = (px, py, theta), pose of base_link in the room (map) frame.
  Predict at every gyro sample (100 Hz):
    rotation    = gyro z minus bias (from the still period), times dt
    translation = /cmd_applied: straight at the nominal speed, or a pivot
                  about a measured point per direction (pivot_check.py, Nov 10;
                  forward pivots assumed)
  Update at every marker sighting (range, bearing in base_link) against the
  known map, with the marker's map uncertainty added to R, after a gate.

Outputs:
  printed: biases, timing, update and gate stats, innovation stats per ID,
           checkpoint comparison at every pause, final pose vs home
  ~/replay/<name>_ekf_map.png    room, markers, path, 95% ellipses, checkpoints
  ~/replay/<name>_ekf_time.png   heading, sigmas, innovations over time
  ~/replay/<name>_ekf.npz        filter output (t, x, P)
"""

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

# ----------------------------------------------------------------------------
# Hand-measured map and checkpoints (tape-measured Nov 10, 2026). Metres. Room frame:
# origin at the south-west corner, x east, y north.
# ----------------------------------------------------------------------------
MARKER_MAP = {  # id: (x, y, sigma_x, sigma_y)
    0: (1.679, 3.924, 0.013, 0.013),
    1: (0.841, 4.902, 0.013, 0.006),
    2: (1.298, 5.644, 0.013, 0.015),
    3: (2.302, 5.644, 0.006, 0.015),
    4: (2.980, 4.750, 0.015, 0.006),
    5: (2.136, 3.543, 0.013, 0.013),
    6: (2.930, 1.676, 0.030, 0.013),
    7: (2.980, 1.054, 0.015, 0.013),
    8: (1.425, 0.020, 0.013, 0.025),
    9: (0.511, 0.020, 0.013, 0.025),
    10: (0.079, 1.181, 0.015, 0.024),
    11: (0.866, 2.184, 0.013, 0.024),
    12: (1.832, 2.184, 0.013, 0.024),
}
CHECKPOINTS = {  # name: (x, y, unused); the checkpoint sigma used is CP_SIGMA
    "home": (1.603, 4.762, 0.025),
    "corridor": (2.498, 3.403, 0.025),
    "south": (1.501, 1.092, 0.025),
}
HOME_POSE = (1.603, 4.762, -np.pi / 2)  # on the home X, facing south

# Room drawing (map frame)
ROOM_W, ROOM_L = 3.00, 5.664
BED = (0.0, 2.184, 2.03, 3.864)  # x0, y0, x1, y1
DESK2 = (0.0, 4.264, 0.80, 5.664)
BOARD = ((3.00, 0.48), (2.55, 0.0))  # poster board, south end approximate

# ----------------------------------------------------------------------------
# Noise model: measured in controlled tests, then scaled (Q_SCALE) on real drives
# ----------------------------------------------------------------------------
P0_POS = 0.08                # m, car placed on the home X by eye (not exact)
P0_TH = np.radians(5.0)      # rad, squared to "south" by eye
SPEED_DEFAULT = 0.40         # m/s, measured from marker ranges Nov 10 (speed_check.py)
GYRO_SCALE_DEFAULT = 1.03    # gyro reads about 3% low (heading shortfall, 4 runs, Nov 10)
K_ALONG = 0.03               # m^2 per m driven: sigma 13 cm after 0.6 m (stalls, rubbing)
Q_SCALE_DEFAULT = 4.0        # all process noise x4: tuned Nov 10 on runs 1 and 2 so that
                             # first sightings after blind driving pass the NIS test
                             # (run 1: NIS 2.4, 91% inside; run 2: NIS 1.0, 100% inside)
K_LAT = 0.0015               # m^2 per m driven: sideways slip
K_TURN = 0.00175             # rad^2 per rad turned: about 3 deg per 90 deg
K_PIVOT = 0.0016             # m^2 per rad pivoted: about 5 cm per 90 deg
Q_BIAS = np.radians(0.1) ** 2  # rad^2 per s: slow gyro bias drift
PIVOT_HALF_TRACK = 0.07      # m, braked side midpoint from base_link (old model, --pivot-model old)
# Pivot points in base_link (x forward, y left), measured Nov 10 with pivot_check.py from
# 15 pivots (m8_pivots and m8_run4 to 6): a 90 deg pivot moves base_link about 22 cm
# forward (left pivots) and 24 cm forward, 18 cm right (right pivots), 2 to 3 times the
# old model. Scatter about the fit is 8 to 9 cm per 90 deg, inside K_PIVOT x Q scale.
PIVOT_C_LEFT = (0.099, 0.122)
PIVOT_C_RIGHT = (0.026, -0.208)
PIVOT_C = {1: PIVOT_C_LEFT, -1: PIVOT_C_RIGHT}
MIN_RANGE, MAX_RANGE = 0.6, 2.5  # tested range of the measurement model
GATE = 9.21                  # chi2, 2 dof, 99%: reject surprising sightings
RECOVER_N = 5                # this many rejections in a row ...
RECOVER_DT = 0.5             # ... spanning at least this long: the filter is lost
RECOVER_POS = 0.15           # m, widen position sigma by this on recovery
RECOVER_TH = np.radians(5.0)  # rad, widen heading sigma by this on recovery
STILL_AFTER = 0.5            # s stopped before thinning starts
STILL_DT = 1.0               # s, at most one update per marker per this while stopped
BLIND_DT = 1.0               # s without an accepted update: next sighting is graded
CP_SIGMA = 0.10              # m, stopping on an X by eye (loose sanity check)
CP_MAX_DIST = 0.40           # m, a pause counts as an X stop if this close
CP_CHI2 = 5.99               # chi2, 2 dof, 95%: "inside the 95% ellipse"
CMD_TIMEOUT = 0.75           # s: no /cmd_applied for this long means stopped
MOVE_EPS = 0.01
PAUSE_MIN = 2.0              # s: a stop at least this long counts as a pause

STOP, FORWARD, PIVOT = 0, 1, 2


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def drot2(a):
    """Derivative of rot2(a) with respect to a."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[-s, -c], [c, -s]])


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def command_per_sample(imu_t, cmd_t, cmd_v, cmd_w):
    """Latest /cmd_applied at each IMU time (zero if stale)."""
    idx = np.searchsorted(cmd_t, imu_t, side="right") - 1
    valid = idx >= 0
    idx_c = idx.clip(0)
    v = np.where(valid, cmd_v[idx_c], 0.0)
    w = np.where(valid, cmd_w[idx_c], 0.0)
    stale = (~valid) | (imu_t - cmd_t[idx_c] > CMD_TIMEOUT)
    v[stale] = 0.0
    w[stale] = 0.0
    mode = np.full(len(imu_t), STOP)
    mode[np.abs(v) > MOVE_EPS] = FORWARD
    mode[(np.abs(v) <= MOVE_EPS) & (np.abs(w) > MOVE_EPS)] = PIVOT
    return v, w, mode


def gyro_biases(imu_t, imu_wz, mode):
    """Bias from the still period before the first motion, and after the last."""
    moving = np.flatnonzero(mode != STOP)
    if len(moving) == 0:
        raise SystemExit("no motion commands found in /cmd_applied")
    t_first, t_last = imu_t[moving[0]], imu_t[moving[-1]]
    start = imu_t < t_first - 0.3
    end = imu_t > t_last + 1.5
    b0 = imu_wz[start].mean()
    b1 = imu_wz[end].mean() if end.sum() >= 300 else None
    return b0, b1, t_first, t_last, start.sum(), end.sum()


# ----------------------------------------------------------------------------
# EKF steps
# ----------------------------------------------------------------------------
def predict(x, P, dt, dth, v, mode, turn_sign=0):
    F = np.eye(3)
    Q = np.zeros((3, 3))
    th = x[2]
    if mode == FORWARD:
        thm = th + 0.5 * dth
        ds = v * dt
        x[0] += ds * np.cos(thm)
        x[1] += ds * np.sin(thm)
        F[0, 2] = -ds * np.sin(thm)
        F[1, 2] = ds * np.cos(thm)
        d = abs(ds)
        R = rot2(thm)
        Q[:2, :2] = R @ np.diag([K_ALONG * d, K_LAT * d]) @ R.T
    elif mode == PIVOT:
        # rotate about a fixed point c of the car (base_link): that point stays put in
        # the room, so p + R(th) c = p' + R(th + dth) c. Direction from the command.
        sgn = turn_sign if turn_sign != 0 else (1 if dth >= 0 else -1)
        c = np.array(PIVOT_C[sgn])
        x[:2] += rot2(th) @ c - rot2(th + dth) @ c
        F[:2, 2] = drot2(th) @ c - drot2(th + dth) @ c
        Q[:2, :2] = K_PIVOT * abs(dth) * np.eye(2)
    x[2] = wrap(th + dth)
    Q[2, 2] += K_TURN * abs(dth) + Q_BIAS * dt
    P = F @ P @ F.T + Q
    return x, P


def update(x, P, z_r, z_b, sig_r, sig_b, mid):
    mx, my, smx, smy = MARKER_MAP[mid]
    dx, dy = mx - x[0], my - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    b_hat = wrap(np.arctan2(dy, dx) - x[2])
    H = np.array([[-dx / r, -dy / r, 0.0],
                  [dy / q, -dx / q, -1.0]])
    Hm = np.array([[dx / r, dy / r],
                   [-dy / q, dx / q]])
    R = np.diag([sig_r ** 2, sig_b ** 2]) + Hm @ np.diag([smx ** 2, smy ** 2]) @ Hm.T
    y = np.array([z_r - r, wrap(z_b - b_hat)])
    S = H @ P @ H.T + R
    S_inv = np.linalg.inv(S)
    d2 = float(y @ S_inv @ y)
    s_diag = np.sqrt(np.diag(S))
    if d2 > GATE:
        return x, P, y, d2, False, s_diag
    K = P @ H.T @ S_inv
    x = x + K @ y
    x[2] = wrap(x[2])
    IKH = np.eye(3) - K @ H
    P = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form: stays symmetric and positive
    return x, P, y, d2, True, s_diag


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def cov_ellipse(mean, cov, chi2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2.0 * np.sqrt(chi2 * vals[1]), 2.0 * np.sqrt(chi2 * vals[0])
    return Ellipse(mean, w, h, angle=angle, fill=False, **kw)


def draw_room(ax):
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_L, fill=False, lw=1.5, color="k"))
    for x0, y0, x1, y1 in (BED, DESK2):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.85"))
    ax.text(1.0, 3.0, "bed", ha="center", va="center", color="0.4")
    ax.text(0.4, 4.95, "desk 2", ha="center", va="center", color="0.4", fontsize=8)
    (bx0, by0), (bx1, by1) = BOARD
    ax.plot([bx0, bx1], [by0, by1], color="0.4", lw=2)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global PIVOT_HALF_TRACK, PIVOT_C, GATE, K_ALONG, K_LAT, K_TURN, K_PIVOT, Q_BIAS
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="output of replay_extract.py, e.g. ~/replay/m7_run1.npz")
    ap.add_argument("--bias", choices=["start", "interp"], default="start",
                    help="gyro bias: start still period only, or interpolate start to end")
    ap.add_argument("--gyro-scale", type=float, default=GYRO_SCALE_DEFAULT,
                    help="multiply bias-corrected gyro rate (default 1.03)")
    ap.add_argument("--pivot-model", choices=["measured", "old", "none"], default="measured",
                    help="measured: pivot points from pivot_check.py (default); old: braked side"
                         " midpoint, 7 cm; none: pivot in place")
    ap.add_argument("--max-range", type=float, default=MAX_RANGE)
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT,
                    help="real forward speed in m/s (default 0.40, measured)")
    ap.add_argument("--q-scale", type=float, default=Q_SCALE_DEFAULT,
                    help="multiply all process noise terms (default 4, tuned Nov 10)")
    ap.add_argument("--gate", type=float, default=GATE,
                    help="chi2 gate on sightings (default 9.21, 99%%)")
    args = ap.parse_args()

    if args.pivot_model == "old":
        PIVOT_C = {1: (0.0, PIVOT_HALF_TRACK), -1: (0.0, -PIVOT_HALF_TRACK)}
    elif args.pivot_model == "none":
        PIVOT_C = {1: (0.0, 0.0), -1: (0.0, 0.0)}
    GATE = args.gate
    K_ALONG *= args.q_scale
    K_LAT *= args.q_scale
    K_TURN *= args.q_scale
    K_PIVOT *= args.q_scale
    Q_BIAS *= args.q_scale

    path = os.path.expanduser(args.npz)
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path)
    d = np.load(path)

    imu_t, imu_wz = d["imu_t"], d["imu_wz"]
    cmd_t, cmd_v, cmd_w = d["cmd_t"], d["cmd_v"], d["cmd_w"]
    t0 = imu_t[0]

    v_s, w_s, mode = command_per_sample(imu_t, cmd_t, cmd_v, cmd_w)
    v_s = np.where(mode == FORWARD, np.sign(v_s) * args.speed, 0.0)
    b0, b1, t_first, t_last, n0, n1 = gyro_biases(imu_t, imu_wz, mode)
    print(f"Run: {name}")
    print(f"Gyro bias, start still ({n0 / 100:.1f} s): {np.degrees(b0):+.4f} deg/s")
    if b1 is not None:
        print(f"Gyro bias, end still   ({n1 / 100:.1f} s): {np.degrees(b1):+.4f} deg/s"
              f"   (drift {np.degrees(b1 - b0):+.4f} deg/s)")
    else:
        print("Gyro bias, end still: too short to measure")
    if args.bias == "interp" and b1 is not None:
        frac = np.clip((imu_t - t_first) / (t_last - t_first), 0.0, 1.0)
        bias = b0 + (b1 - b0) * frac
        print("Bias mode: interpolated start to end")
    else:
        bias = np.full(len(imu_t), b0)
        print("Bias mode: start still period")
    print(f"Pivot model: {args.pivot_model}, pivot points (cm, x fwd, y left): left"
          f" ({100 * PIVOT_C[1][0]:+.1f}, {100 * PIVOT_C[1][1]:+.1f}),"
          f" right ({100 * PIVOT_C[-1][0]:+.1f}, {100 * PIVOT_C[-1][1]:+.1f})")
    print(f"Gyro scale: {args.gyro_scale},"
          f" max range: {args.max_range} m, gate: {GATE}, q-scale: {args.q_scale}")
    print(f"Forward speed: {args.speed} m/s, start sigma {100 * P0_POS:.0f} cm /"
          f" {np.degrees(P0_TH):.0f} deg, checkpoint sigma {100 * CP_SIGMA:.0f} cm")

    # sightings usable by the filter
    o_t, o_id = d["obs_t"], d["obs_id"]
    o_r, o_b = d["obs_range"], d["obs_bearing"]
    o_sr, o_sb = d["obs_sigma_r"], d["obs_sigma_b"]
    known = np.array([i in MARKER_MAP for i in o_id], dtype=bool)
    in_range = (o_r >= MIN_RANGE) & (o_r <= args.max_range)
    use = known & in_range & (o_t >= imu_t[0])
    print(f"\nSightings: {len(o_t)} total, {int((~known).sum())} unknown IDs,"
          f" {int((known & ~in_range).sum())} outside {MIN_RANGE} to {args.max_range} m,"
          f" {int(use.sum())} used")
    order = np.argsort(o_t[use], kind="stable")
    O = {k: arr[use][order] for k, arr in
         (("t", o_t), ("id", o_id), ("r", o_r), ("b", o_b), ("sr", o_sr), ("sb", o_sb))}
    n_obs = len(O["t"])

    # filter
    x = np.array(HOME_POSE, dtype=float)
    P = np.diag([P0_POS ** 2, P0_POS ** 2, P0_TH ** 2])
    n = len(imu_t)
    X = np.zeros((n, 3))
    PP = np.zeros((n, 3, 3))
    X[0], PP[0] = x, P
    upd_rows = []  # t, id, y_r, y_b, d2, accepted, graded (first after blind)
    j = 0
    t_pred = t_upd = 0.0
    n_pred = 0
    n_thinned = 0
    recoveries = []
    last_upd = {}            # marker id -> time of its last update
    t_last_seen = imu_t[0]   # last accepted (or thinned) sighting
    reject_run, reject_t0 = 0, None
    t_stop_start = imu_t[0]
    for i in range(1, n):
        dt = imu_t[i] - imu_t[i - 1]
        dth = args.gyro_scale * (imu_wz[i] - bias[i]) * dt
        if mode[i] != STOP:
            t_stop_start = imu_t[i]
        still = imu_t[i] - t_stop_start > STILL_AFTER
        tp = time.perf_counter()
        x, P = predict(x, P, dt, dth, v_s[i], mode[i], int(np.sign(w_s[i])))
        t_pred += time.perf_counter() - tp
        n_pred += 1
        while j < n_obs and O["t"][j] <= imu_t[i]:
            tj, mid = O["t"][j], int(O["id"][j])
            j += 1
            # repeated sightings from a standing car share one bias: thin them
            if still and tj - last_upd.get(mid, -1e9) < STILL_DT:
                n_thinned += 1
                t_last_seen = tj
                continue
            graded = tj - t_last_seen >= BLIND_DT
            tu = time.perf_counter()
            x, P, y, d2, ok, sd = update(x, P, O["r"][j - 1], O["b"][j - 1],
                                         O["sr"][j - 1], O["sb"][j - 1], mid)
            blind = tj - t_last_seen
            if not ok:
                if reject_run == 0:
                    reject_t0 = tj
                reject_run += 1
                if reject_run >= RECOVER_N and tj - reject_t0 >= RECOVER_DT:
                    # the filter, not the markers, is probably wrong: widen and retry
                    P = P + np.diag([RECOVER_POS ** 2, RECOVER_POS ** 2, RECOVER_TH ** 2])
                    recoveries.append(tj)
                    reject_run = 0
                    # note: the logged row keeps the first (rejected) innovation
                    x, P, y2, d2b, ok, _ = update(x, P, O["r"][j - 1], O["b"][j - 1],
                                                  O["sr"][j - 1], O["sb"][j - 1], mid)
            t_upd += time.perf_counter() - tu
            if ok:
                reject_run = 0
                t_last_seen = tj
                last_upd[mid] = tj
            upd_rows.append((tj, mid, y[0], y[1], d2, ok, graded,
                             y[0] / sd[0], y[1] / sd[1], blind))
        X[i], PP[i] = x, P

    U = np.array(upd_rows, dtype=float).reshape(-1, 10)
    acc = U[:, 5] == 1
    print(f"Thinned (repeat sightings while stopped): {n_thinned}")
    print(f"Updates: {len(U)} tried, {int(acc.sum())} accepted,"
          f" {int((~acc).sum())} gated out ({100.0 * (~acc).mean():.1f}%)")
    print(f"Recoveries (filter widened after {RECOVER_N} rejections in a row):"
          f" {len(recoveries)}" + (" at " + ", ".join(f"{t - t0:.1f}" for t in recoveries) + " s"
                                   if recoveries else ""))
    print(f"Timing on this machine: predict {1e6 * t_pred / max(n_pred, 1):.0f} us,"
          f" update {1e6 * t_upd / max(len(U), 1):.0f} us")
    g = U[:, 6] == 1
    print(f"\nHonesty test (first sighting after {BLIND_DT:.0f} s or more without an update;"
          " NIS before the update):")
    if g.any():
        nis = U[g, 4]
        print(f"  graded sightings: {int(g.sum())}, mean NIS {nis.mean():.2f} (honest: about 2),"
              f" inside 95% region: {100.0 * np.mean(nis <= CP_CHI2):.0f}% (honest: about 95%)")
        print("  well under 95% = overconfident, ellipses too small;"
              " near 100% with mean far below 2 = too cautious")
        zr, zb = U[g, 7], U[g, 8]
        print(f"  normalized RMS (honest: about 1): range {np.sqrt(np.mean(zr ** 2)):.2f},"
              f" bearing {np.sqrt(np.mean(zb ** 2)):.2f}")
        bl = U[g, 9]
        for lo, hi in ((1.0, 2.0), (2.0, 4.0), (4.0, 1e9)):
            m = (bl >= lo) & (bl < hi)
            if m.any():
                label = f"{lo:.0f} to {hi:.0f} s" if hi < 1e8 else f"{lo:.0f} s or more"
                print(f"  blind {label:12s}: n {int(m.sum()):3d}, mean NIS {nis[m].mean():6.2f},"
                      f" inside 95% {100.0 * np.mean(nis[m] <= CP_CHI2):3.0f}%")
    else:
        print("  none")

    print("\nInnovations per ID (accepted; measured minus predicted):")
    print("   ID     n   range mean   range sd   bearing mean   bearing sd   gated")
    for mid in sorted(MARKER_MAP):
        sel = U[:, 1] == mid
        if not sel.any():
            continue
        a = sel & acc
        if a.any():
            print(f"  {mid:3d}  {int(a.sum()):4d}   {100 * U[a, 2].mean():+7.1f} cm"
                  f"   {100 * U[a, 2].std():6.1f} cm   {np.degrees(U[a, 3].mean()):+8.2f} deg"
                  f"   {np.degrees(U[a, 3].std()):7.2f} deg   {int((sel & ~acc).sum()):4d}")
        else:
            print(f"  {mid:3d}     0   (all {int(sel.sum())} gated out)")

    # pauses: stops of at least PAUSE_MIN seconds between first and last motion,
    # plus the final still period
    stop = mode == STOP
    pauses = []
    i = 0
    while i < n:
        if stop[i]:
            k = i
            while k + 1 < n and stop[k + 1]:
                k += 1
            if imu_t[k] - imu_t[i] >= PAUSE_MIN and imu_t[i] > t_first:
                pauses.append((i, k))
            i = k + 1
        else:
            i += 1

    print("\nPauses (estimate at the end of each stop of 2 s or more):")
    print("   time    nearest       dist   est x   est y   err x   err y   sig x   sig y"
          "    d2   in 95%")
    cp_results = []
    for i_start, i_end in pauses:
        p = X[i_end, :2]
        best = min(CHECKPOINTS, key=lambda c: np.hypot(*(p - CHECKPOINTS[c][:2])))
        cx, cy, _ = CHECKPOINTS[best]
        cs = CP_SIGMA
        dist = np.hypot(p[0] - cx, p[1] - cy)
        e = p - np.array([cx, cy])
        C = PP[i_end, :2, :2] + cs ** 2 * np.eye(2)
        d2 = float(e @ np.linalg.inv(C) @ e)
        sx, sy = np.sqrt(PP[i_end, 0, 0]), np.sqrt(PP[i_end, 1, 1])
        on_cp = dist < CP_MAX_DIST
        flag = ("yes" if d2 <= CP_CHI2 else "NO") if on_cp else "(not on an X?)"
        print(f"  {imu_t[i_end] - t0:6.1f}  {best:10s}  {100 * dist:5.1f} cm"
              f"  {p[0]:6.3f}  {p[1]:6.3f}  {100 * e[0]:+6.1f}  {100 * e[1]:+6.1f}"
              f"  {100 * sx:5.1f}  {100 * sy:5.1f}  {d2:5.2f}   {flag}")
        if on_cp:
            cp_results.append((best, d2 <= CP_CHI2))
    if cp_results:
        n_in = sum(ok for _, ok in cp_results)
        print(f"  X stops inside the 95% ellipse (loose check, sigma {100 * CP_SIGMA:.0f} cm):"
              f" {n_in} of {len(cp_results)}")

    th_end = np.degrees(wrap(X[-1, 2] - HOME_POSE[2]))
    e_end = X[-1, :2] - np.array(HOME_POSE[:2])
    print(f"\nFinal pose vs home X: {100 * e_end[0]:+.1f} cm, {100 * e_end[1]:+.1f} cm;"
          f" heading {th_end:+.1f} deg from south"
          f" (sigma {np.degrees(np.sqrt(PP[-1, 2, 2])):.1f} deg)")

    # ------------------------------------------------------------------ plots
    tt = imu_t - t0
    fig, ax = plt.subplots(figsize=(6.5, 10))
    draw_room(ax)
    for mid, (mx, my, smx, smy) in MARKER_MAP.items():
        seen = (U[:, 1] == mid).any() if len(U) else False
        ax.plot(mx, my, "s", color="tab:orange" if seen else "0.6", ms=8)
        ax.annotate(str(mid), (mx, my), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.plot(X[:, 0], X[:, 1], color="tab:blue", lw=1.2, label="estimated path")
    step = 100  # one ellipse per second
    for i in range(0, n, step):
        ax.add_patch(cov_ellipse(X[i, :2], PP[i, :2, :2], CP_CHI2,
                                 color="tab:blue", lw=0.6, alpha=0.6))
    for i_start, i_end in pauses:
        ax.add_patch(cov_ellipse(X[i_end, :2], PP[i_end, :2, :2], CP_CHI2,
                                 color="tab:green", lw=1.4))
        ax.plot(*X[i_end, :2], ".", color="tab:green")
    for cname, (cx, cy, _) in CHECKPOINTS.items():
        ax.plot(cx, cy, "x", color="tab:red", ms=12, mew=2)
        ax.annotate(cname, (cx, cy), textcoords="offset points", xytext=(8, -12),
                    color="tab:red", fontsize=9)
    ax.plot(*X[0, :2], "o", color="k", ms=5, label="start")
    ax.plot(*X[-1, :2], "^", color="k", ms=6, label="end")
    ax.set_xlim(-0.3, ROOM_W + 0.3)
    ax.set_ylim(-0.3, ROOM_L + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title(f"{name}: EKF localization, 95% ellipses\n"
                 "blue = every 1 s, green = pauses, red X = checkpoints", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8)
    fig.tight_layout()
    map_png = os.path.join(out_dir, f"{name}_ekf_map.png")
    fig.savefig(map_png, dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axs[0].plot(tt, np.degrees(np.unwrap(X[:, 2])), lw=1)
    axs[0].set_ylabel("heading (deg)")
    axs[1].plot(tt, 100 * np.sqrt(PP[:, 0, 0]), lw=1, label="sigma x (cm)")
    axs[1].plot(tt, 100 * np.sqrt(PP[:, 1, 1]), lw=1, label="sigma y (cm)")
    axs[1].plot(tt, np.degrees(np.sqrt(PP[:, 2, 2])), lw=1, label="sigma heading (deg)")
    axs[1].legend(fontsize=8, loc="upper left")
    axs[1].set_ylabel("uncertainty")
    if len(U):
        ut = U[:, 0] - t0
        axs[2].plot(ut[acc], 100 * U[acc, 2], ".", ms=2, label="accepted")
        axs[2].plot(ut[~acc], 100 * U[~acc, 2], "x", ms=4, color="tab:red", label="gated out")
        axs[3].plot(ut[acc], np.degrees(U[acc, 3]), ".", ms=2)
        axs[3].plot(ut[~acc], np.degrees(U[~acc, 3]), "x", ms=4, color="tab:red")
        axs[2].legend(fontsize=8, loc="upper left")
    axs[2].set_ylabel("range innov (cm)")
    axs[3].set_ylabel("bearing innov (deg)")
    axs[3].set_xlabel("time (s)")
    for a in axs:
        for i_start, i_end in pauses:
            a.axvspan(tt[i_start], tt[i_end], color="tab:green", alpha=0.08)
        for tr in recoveries:
            a.axvline(tr - t0, color="tab:purple", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    axs[2].set_ylim(-30, 30)
    axs[3].set_ylim(-15, 15)
    fig.suptitle(f"{name}: filter over time (green bands = pauses, purple = recoveries)")
    fig.tight_layout()
    time_png = os.path.join(out_dir, f"{name}_ekf_time.png")
    fig.savefig(time_png, dpi=110)
    plt.close(fig)

    np.savez(os.path.join(out_dir, f"{name}_ekf.npz"), t=imu_t, x=X, P=PP, updates=U)
    print(f"\nSaved {map_png}\nSaved {time_png}")


if __name__ == "__main__":
    main()
