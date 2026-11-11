#!/usr/bin/env python3
"""EKF-SLAM on a recorded drive. Version 2 (Nov 11, 2026).

v2 changes: stalled-pivot slip noise (K_SLIP), and map scoring that propagates the
alignment's own uncertainty (the anchor marker is placed by the other markers).

Usage (on the Pi, with ekf_localize.py in the same folder):
    python3 ekf_slam.py ~/replay/m8_run4.npz
    python3 ekf_slam.py ~/replay/m8_run4.npz --frame room

Input: the .npz written by replay_extract.py.

State: x = (px, py, theta, m_a_x, m_a_y, m_b_x, m_b_y, ...)
       the car pose plus every marker seen so far, in the map frame. Size 3 + 2N.
  Predict (every gyro sample, 100 Hz): the car only, with the same motion model as
    ekf_localize.py (gyro rotation, 0.40 m/s forward, measured pivot points, Q x4).
    Markers do not move, so only the car block of P grows; the car-map cross block
    is carried along (P_rm <- F P_rm).
  New marker: once 3 sightings of a new ID within 1 s agree to 15 cm, it joins the
    state from the latest sighting (inverse measurement model). Its covariance
    inherits the car's uncertainty and it is correlated with the car and with
    every marker already in the map.
  Update (every sighting of a mapped marker): range and bearing, Jacobian with a car
    block and a marker block. The gain moves every correlated state, which is what
    makes loop closure shrink the whole map.
  The hand-measured map's positions are NOT used by the filter. It is only used
  afterwards, to score the result. Its ID list is used to accept only IDs 0 to 12,
  so spare marker prints in view cannot enter the map.

Frames:
  --frame start (default): the map frame is the car's start pose with no
    uncertainty (the textbook choice). It is drawn at the home X facing south.
    Scoring aligns the estimated map to the tape map with a rigid fit (x, y,
    rotation), leaving each scored marker out of its own alignment.
  --frame room: start pose uncertain (8 cm, 5 deg, placed by eye); positions are
    compared with the tape map directly, no alignment. Honest but the ellipses
    carry the start uncertainty.

Outputs:
  printed: settings, markers added, update and gate stats, recoveries, loop
           closures (re-sightings after a long gap), honesty test, timing, the
           map score per marker
  ~/replay/<name>_slam_map.png     estimated map and path; aligned map vs tape
  ~/replay/<name>_slam_time.png    car and marker uncertainty over time
  ~/replay/<name>_slam.npz         car pose at 100 Hz, full state and covariance
                                   at 10 Hz (for the replay video), map, score
"""

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ekf_localize as E  # noqa: E402  (motion model constants, helpers, tape map)

wrap, rot2, drot2 = E.wrap, E.rot2, E.drot2

CONFIRM_N = 3          # sightings of a new ID needed before it joins the map
CONFIRM_DT = 1.0       # s, within this window
CONFIRM_DIST = 0.15    # m, all implied positions within this of their mean
REVISIT_GAP = 20.0     # s without an update on a marker: next update is a re-sighting
# Stalled pivots (v2): a pivot command that turns the car less than SLIP_FRAC of the
# nominal rate is probably shoving it (rubbing, braked side sliding). The missing
# rotation adds position noise along the heading (a quarter of it sideways).
K_SLIP_DEFAULT = 0.10   # m^2 per rad of missing rotation: sigma about 30 cm for 0.94 rad
W_PIVOT_NOM = np.radians(23.0)  # rad/s, braked pivot at duty 0.7
SLIP_FRAC = 0.8
SLIP_SIDE = 0.25
SNAP_DT = 0.1          # s, full state saved at 10 Hz for the replay video
CHI2_95 = 5.99


# ----------------------------------------------------------------------------
# Filter steps
# ----------------------------------------------------------------------------
def car_motion(xr, dt, dth, v, mode, sgn, K, pivot_c):
    """Same motion model as ekf_localize.predict, for the car part of the state."""
    x = xr.copy()
    F = np.eye(3)
    Q = np.zeros((3, 3))
    th = xr[2]
    if mode == E.FORWARD:
        thm = th + 0.5 * dth
        ds = v * dt
        x[0] += ds * np.cos(thm)
        x[1] += ds * np.sin(thm)
        F[0, 2] = -ds * np.sin(thm)
        F[1, 2] = ds * np.cos(thm)
        d = abs(ds)
        R = rot2(thm)
        Q[:2, :2] = R @ np.diag([K["along"] * d, K["lat"] * d]) @ R.T
    elif mode == E.PIVOT:
        s = sgn if sgn != 0 else (1 if dth >= 0 else -1)
        c = np.array(pivot_c[s])
        x[:2] += rot2(th) @ c - rot2(th + dth) @ c
        F[:2, 2] = drot2(th) @ c - drot2(th + dth) @ c
        Q[:2, :2] = K["pivot"] * abs(dth) * np.eye(2)
        nom = W_PIVOT_NOM * dt
        if K["slip"] > 0 and abs(dth) < SLIP_FRAC * nom:
            miss = nom - abs(dth)
            Rh = rot2(th)
            Q[:2, :2] += Rh @ np.diag([K["slip"] * miss, SLIP_SIDE * K["slip"] * miss]) @ Rh.T
    x[2] = wrap(th + dth)
    Q[2, 2] += K["turn"] * abs(dth) + K["bias"] * dt
    return x, F, Q


def predict(x, P, F, Q):
    """P <- Fbig P Fbig^T + Qbig with Fbig = blockdiag(F, I): only car rows and columns change."""
    P[:3, :] = F @ P[:3, :]
    P[:, :3] = P[:, :3] @ F.T
    P[:3, :3] += Q
    return P


def implied_position(x, r, b):
    a = x[2] + b
    return np.array([x[0] + r * np.cos(a), x[1] + r * np.sin(a)])


def add_marker(x, P, r, b, sr, sb):
    """State augmentation with the inverse measurement model."""
    n = len(x)
    a = x[2] + b
    ca, sa = np.cos(a), np.sin(a)
    m = np.array([x[0] + r * ca, x[1] + r * sa])
    Gx = np.array([[1.0, 0.0, -r * sa],
                   [0.0, 1.0, r * ca]])
    Gz = np.array([[ca, -r * sa],
                   [sa, r * ca]])
    Rz = np.diag([sr ** 2, sb ** 2])
    Pmm = Gx @ P[:3, :3] @ Gx.T + Gz @ Rz @ Gz.T
    Pmx = Gx @ P[:3, :]                      # 2 x n: with the car and every marker
    Pn = np.zeros((n + 2, n + 2))
    Pn[:n, :n] = P
    Pn[n:, :n] = Pmx
    Pn[:n, n:] = Pmx.T
    Pn[n:, n:] = Pmm
    return np.concatenate([x, m]), Pn


def update(x, P, i, z_r, z_b, sr, sb, gate):
    """Range-bearing update on the marker at state index i (its x; y is i + 1)."""
    n = len(x)
    dx, dy = x[i] - x[0], x[i + 1] - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    b_hat = wrap(np.arctan2(dy, dx) - x[2])
    H = np.zeros((2, n))
    H[:, :3] = [[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]]
    H[:, i:i + 2] = [[dx / r, dy / r], [-dy / q, dx / q]]
    R = np.diag([sr ** 2, sb ** 2])
    y = np.array([z_r - r, wrap(z_b - b_hat)])
    PHt = P @ H.T
    S = H @ PHt + R
    S_inv = np.linalg.inv(S)
    d2 = float(y @ S_inv @ y)
    s_diag = np.sqrt(np.diag(S))
    if d2 > gate:
        return x, P, y, d2, False, s_diag
    Kg = PHt @ S_inv
    x = x + Kg @ y
    x[2] = wrap(x[2])
    IKH = np.eye(n) - Kg @ H
    P = IKH @ P @ IKH.T + Kg @ R @ Kg.T      # Joseph form
    P = 0.5 * (P + P.T)
    return x, P, y, d2, True, s_diag


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def rigid_fit(A, B):
    """Rotation phi and translation t minimizing sum |R(phi) a + t - b|^2 (2D Kabsch)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    num = np.sum(A0[:, 0] * B0[:, 1] - A0[:, 1] * B0[:, 0])
    den = np.sum(A0[:, 0] * B0[:, 0] + A0[:, 1] * B0[:, 1])
    phi = np.arctan2(num, den)
    t = cb - rot2(phi) @ ca
    return phi, t


def score_map(ids, est, cov, align, min_for_fit=3, Pmap=None):
    """v2: with align=True and the full map covariance Pmap (2N x 2N, in ids order),
    the leave-one-out placement's own uncertainty is propagated (numerical Jacobian
    with respect to every marker estimate and every tape position)."""
    if align and Pmap is not None:
        return score_map_v2(ids, est, Pmap, min_for_fit)
    return score_map_v1(ids, est, cov, align, min_for_fit)


def score_map_v2(ids, est, Pmap, min_for_fit=3, h=1e-5):
    tape = np.array([E.MARKER_MAP[i][:2] for i in ids])
    tsig = np.array([E.MARKER_MAP[i][2:] for i in ids])
    Pt = np.diag((tsig ** 2).ravel())
    N = len(ids)
    rows = []
    for k, mid in enumerate(ids):
        others = [j for j in range(N) if j != k]
        if len(others) < min_for_fit - 1:
            continue

        def err(ef, tf):
            e_, t_ = ef.reshape(-1, 2), tf.reshape(-1, 2)
            phi, tr = rigid_fit(e_[others], t_[others])
            return rot2(phi) @ e_[k] + tr - t_[k]

        ef, tf = est.ravel().copy(), tape.ravel().copy()
        e = err(ef, tf)
        Je = np.zeros((2, 2 * N))
        Jt = np.zeros((2, 2 * N))
        for c in range(2 * N):
            d = np.zeros(2 * N)
            d[c] = h
            Je[:, c] = (err(ef + d, tf) - err(ef - d, tf)) / (2 * h)
            Jt[:, c] = (err(ef, tf + d) - err(ef, tf - d)) / (2 * h)
        C = Je @ Pmap @ Je.T + Jt @ Pt @ Jt.T
        d2 = float(e @ np.linalg.inv(C) @ e)
        sig = np.sqrt(np.linalg.eigvalsh(C)[::-1])
        rows.append((mid, e + tape[k], e, sig, d2))
    return rows


def score_map_v1(ids, est, cov, align, min_for_fit=3):
    """Per marker: error vs tape, Mahalanobis d2 with the marker's covariance plus the
    tape sigma. align=True: rigid fit on the other markers (leave one out)."""
    tape = np.array([E.MARKER_MAP[i][:2] for i in ids])
    tsig = np.array([E.MARKER_MAP[i][2:] for i in ids])
    rows = []
    for k, mid in enumerate(ids):
        if align:
            others = [j for j in range(len(ids)) if j != k]
            if len(others) < min_for_fit - 1:
                continue
            phi, t = rigid_fit(est[others], tape[others])
        else:
            phi, t = 0.0, np.zeros(2)
        Rr = rot2(phi)
        p = Rr @ est[k] + t
        C = Rr @ cov[k] @ Rr.T + np.diag(tsig[k] ** 2)
        e = p - tape[k]
        d2 = float(e @ np.linalg.inv(C) @ e)
        sig = np.sqrt(np.linalg.eigvalsh(C)[::-1])
        rows.append((mid, p, e, sig, d2))
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="output of replay_extract.py, e.g. ~/replay/m8_run4.npz")
    ap.add_argument("--frame", choices=["start", "room"], default="start",
                    help="start: map frame = start pose, scored after a rigid alignment"
                         " (default); room: start uncertain (8 cm, 5 deg), scored directly")
    ap.add_argument("--gyro-scale", type=float, default=E.GYRO_SCALE_DEFAULT)
    ap.add_argument("--speed", type=float, default=E.SPEED_DEFAULT)
    ap.add_argument("--q-scale", type=float, default=E.Q_SCALE_DEFAULT)
    ap.add_argument("--gate", type=float, default=E.GATE)
    ap.add_argument("--max-range", type=float, default=E.MAX_RANGE)
    ap.add_argument("--pivot-model", choices=["measured", "old"], default="measured")
    ap.add_argument("--k-slip", type=float, default=K_SLIP_DEFAULT,
                    help="stalled-pivot slip noise, m^2 per rad of missing rotation (default 0.10; 0 = v1)")
    ap.add_argument("--no-recover", action="store_true", help="switch the recovery rule off")
    args = ap.parse_args()

    qs = args.q_scale
    K = dict(along=E.K_ALONG * qs, lat=E.K_LAT * qs, turn=E.K_TURN * qs,
             pivot=E.K_PIVOT * qs, bias=E.Q_BIAS * qs, slip=args.k_slip)
    pivot_c = dict(E.PIVOT_C)
    if args.pivot_model == "old":
        pivot_c = {1: (0.0, E.PIVOT_HALF_TRACK), -1: (0.0, -E.PIVOT_HALF_TRACK)}

    path = os.path.expanduser(args.npz)
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path)
    d = np.load(path)
    imu_t, imu_wz = d["imu_t"], d["imu_wz"]
    t0 = imu_t[0]

    v_s, w_s, mode = E.command_per_sample(imu_t, d["cmd_t"], d["cmd_v"], d["cmd_w"])
    v_s = np.where(mode == E.FORWARD, np.sign(v_s) * args.speed, 0.0)
    b0, b1, t_first, t_last, n0, n1 = E.gyro_biases(imu_t, imu_wz, mode)
    print(f"Run: {name}   (EKF-SLAM: the hand-measured map is used only for scoring)")
    print(f"Gyro bias, start still ({n0 / 100:.1f} s): {np.degrees(b0):+.4f} deg/s")
    print(f"Frame: {args.frame}, pivot model: {args.pivot_model}, gyro scale {args.gyro_scale},"
          f" speed {args.speed} m/s, q-scale {qs}, gate {args.gate},"
          f" max range {args.max_range} m, recovery {'off' if args.no_recover else 'on'}")
    print(f"ekf_slam.py v2: stalled-pivot slip K_SLIP {args.k_slip} m^2/rad (below"
          f" {SLIP_FRAC} x {np.degrees(W_PIVOT_NOM):.0f} deg/s), scoring with alignment uncertainty")
    print(f"New markers join after {CONFIRM_N} agreeing sightings within {CONFIRM_DT} s"
          f" (to {100 * CONFIRM_DIST:.0f} cm); only IDs {min(E.MARKER_MAP)} to {max(E.MARKER_MAP)}")

    o_t, o_id = d["obs_t"], d["obs_id"]
    o_r, o_b = d["obs_range"], d["obs_bearing"]
    o_sr, o_sb = d["obs_sigma_r"], d["obs_sigma_b"]
    known = np.array([i in E.MARKER_MAP for i in o_id], dtype=bool)
    in_range = (o_r >= E.MIN_RANGE) & (o_r <= args.max_range)
    use = known & in_range & (o_t >= imu_t[0])
    print(f"\nSightings: {len(o_t)} total, {int((~known).sum())} unknown IDs ignored,"
          f" {int((known & ~in_range).sum())} outside {E.MIN_RANGE} to {args.max_range} m,"
          f" {int(use.sum())} used")
    order = np.argsort(o_t[use], kind="stable")
    O = {k: arr[use][order] for k, arr in
         (("t", o_t), ("id", o_id), ("r", o_r), ("b", o_b), ("sr", o_sr), ("sb", o_sb))}
    n_obs = len(O["t"])

    # ------------------------------------------------------------------ filter
    x = np.array(E.HOME_POSE, dtype=float)
    if args.frame == "room":
        P = np.diag([E.P0_POS ** 2, E.P0_POS ** 2, E.P0_TH ** 2])
    else:
        P = np.diag([1e-8, 1e-8, 1e-8])
    slot = {}                 # marker id -> state index of its x
    pending = {}              # marker id -> recent implied positions (t, xy)
    added = []                # (t, id, sigma at init)
    n = len(imu_t)
    Xc = np.zeros((n, 3))
    Pc = np.zeros((n, 3, 3))
    snaps_t, snaps_x, snaps_P = [], [], []
    DMAX = 3 + 2 * len(E.MARKER_MAP)
    next_snap = imu_t[0]
    upd_rows = []             # t, id, y_r, y_b, d2, ok, graded, nr_r, nr_b, blind
    revisits = []             # (t, id, gap, car sigma before/after, mean marker sigma before/after)
    marker_sig = {}           # id -> list of (t, sigma major axis)
    recoveries = []
    last_upd = {}
    t_last_seen = imu_t[0]
    reject_run, reject_t0 = 0, None
    t_stop_start = imu_t[0]
    n_thinned = 0
    t_pred = 0.0
    upd_time = {}             # state size -> [total s, count]
    j = 0

    def mean_marker_sigma(P_):
        if not slot:
            return 0.0
        return float(np.mean([np.sqrt(np.linalg.eigvalsh(P_[i:i + 2, i:i + 2])[-1])
                              for i in slot.values()]))

    for k in range(1, n):
        dt = imu_t[k] - imu_t[k - 1]
        dth = args.gyro_scale * (imu_wz[k] - b0) * dt
        if mode[k] != E.STOP:
            t_stop_start = imu_t[k]
        still = imu_t[k] - t_stop_start > E.STILL_AFTER
        tp = time.perf_counter()
        xr, F, Q = car_motion(x[:3], dt, dth, v_s[k], mode[k], int(np.sign(w_s[k])), K, pivot_c)
        x[:3] = xr
        P = predict(x, P, F, Q)
        t_pred += time.perf_counter() - tp

        while j < n_obs and O["t"][j] <= imu_t[k]:
            tj, mid = O["t"][j], int(O["id"][j])
            zr, zb, sr, sb = O["r"][j], O["b"][j], O["sr"][j], O["sb"][j]
            j += 1
            if mid not in slot:
                # new marker: wait for agreeing sightings, then add it to the state
                lst = [(tt, p) for tt, p in pending.get(mid, []) if tj - tt <= CONFIRM_DT]
                lst.append((tj, implied_position(x, zr, zb)))
                pending[mid] = lst
                pts = np.array([p for _, p in lst])
                if len(lst) >= CONFIRM_N and np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) <= CONFIRM_DIST:
                    slot[mid] = len(x)
                    x, P = add_marker(x, P, zr, zb, sr, sb)
                    i = slot[mid]
                    s_init = np.sqrt(np.linalg.eigvalsh(P[i:i + 2, i:i + 2])[-1])
                    added.append((tj, mid, s_init, np.sqrt(max(P[0, 0], P[1, 1]))))
                    marker_sig[mid] = [(tj, s_init)]
                    last_upd[mid] = tj
                    t_last_seen = tj
                    pending.pop(mid, None)
                continue
            if still and tj - last_upd.get(mid, -1e9) < E.STILL_DT:
                n_thinned += 1
                t_last_seen = tj
                continue
            graded = tj - t_last_seen >= E.BLIND_DT
            blind = tj - t_last_seen
            gap = tj - last_upd.get(mid, tj)
            car_before = np.sqrt(max(P[0, 0], P[1, 1]))
            map_before = mean_marker_sigma(P)
            tu = time.perf_counter()
            x, P, y, d2, ok, sd = update(x, P, slot[mid], zr, zb, sr, sb, args.gate)
            if not ok and not args.no_recover:
                if reject_run == 0:
                    reject_t0 = tj
                reject_run += 1
                if reject_run >= E.RECOVER_N and tj - reject_t0 >= E.RECOVER_DT:
                    # the car is probably not where the filter thinks: treat it as an
                    # unmodelled slip (extra process noise on the car only), then retry
                    P[:3, :3] += np.diag([E.RECOVER_POS ** 2, E.RECOVER_POS ** 2, E.RECOVER_TH ** 2])
                    recoveries.append(tj)
                    reject_run = 0
                    # note: the logged row keeps the first (rejected) innovation
                    x, P, _, _, ok, _ = update(x, P, slot[mid], zr, zb, sr, sb, args.gate)
            el = time.perf_counter() - tu
            u = upd_time.setdefault(len(x), [0.0, 0])
            u[0] += el
            u[1] += 1
            if ok:
                reject_run = 0
                if gap >= REVISIT_GAP:
                    revisits.append((tj, mid, gap, car_before, np.sqrt(max(P[0, 0], P[1, 1])),
                                     map_before, mean_marker_sigma(P)))
                t_last_seen = tj
                last_upd[mid] = tj
                i = slot[mid]
                marker_sig[mid].append((tj, np.sqrt(np.linalg.eigvalsh(P[i:i + 2, i:i + 2])[-1])))
            upd_rows.append((tj, mid, y[0], y[1], d2, ok, graded, y[0] / sd[0], y[1] / sd[1], blind))

        Xc[k], Pc[k] = x[:3], P[:3, :3]
        if imu_t[k] >= next_snap:
            xs = np.full(DMAX, np.nan)
            Ps = np.full((DMAX, DMAX), np.nan)
            xs[:len(x)] = x
            Ps[:len(x), :len(x)] = P
            snaps_t.append(imu_t[k])
            snaps_x.append(xs)
            snaps_P.append(Ps)
            next_snap += SNAP_DT
    Xc[0], Pc[0] = Xc[1], Pc[1]

    # ------------------------------------------------------------------ report
    U = np.array(upd_rows, dtype=float).reshape(-1, 10)
    acc = U[:, 5] == 1
    print(f"\nMarkers added to the map ({len(added)}): time s, ID, marker sigma at entry, car sigma then")
    for tj, mid, s_init, s_car in added:
        print(f"  {tj - t0:6.1f}  ID {mid:2d}   {100 * s_init:5.1f} cm   car {100 * s_car:5.1f} cm")
    never = sorted(set(E.MARKER_MAP) - set(slot))
    if never:
        print(f"  never added: {never}")
    print(f"\nThinned (repeat sightings while stopped): {n_thinned}")
    print(f"Updates: {len(U)} tried, {int(acc.sum())} accepted, {int((~acc).sum())} gated out"
          f" ({100.0 * (~acc).mean() if len(U) else 0:.1f}%)")
    print(f"Recoveries: {len(recoveries)}" + (" at " + ", ".join(f"{t - t0:.1f}" for t in recoveries) + " s"
                                               if recoveries else ""))
    print(f"Timing on this machine: predict {1e6 * t_pred / max(n - 1, 1):.0f} us; update by state size: "
          + ", ".join(f"{sz}: {1e6 * tot / c:.0f} us" for sz, (tot, c) in sorted(upd_time.items())))

    print(f"\nLoop closures (a marker updated again after {REVISIT_GAP:.0f} s or more):")
    if revisits:
        print("    time   ID    gap   car sigma before/after   mean marker sigma before/after")
        for tj, mid, gap, cb, ca, mb, ma in revisits:
            print(f"  {tj - t0:6.1f}  {mid:3d}  {gap:5.1f} s    {100 * cb:5.1f} -> {100 * ca:5.1f} cm"
                  f"            {100 * mb:5.1f} -> {100 * ma:5.1f} cm")
    else:
        print("  none")

    g = U[:, 6] == 1
    print(f"\nHonesty test (first sighting after {E.BLIND_DT:.0f} s or more without an update):")
    if g.any():
        nis = U[g, 4]
        print(f"  graded: {int(g.sum())}, mean NIS {nis.mean():.2f} (honest: about 2),"
              f" inside 95%: {100.0 * np.mean(nis <= CHI2_95):.0f}% (honest: about 95%);"
              f" normalized RMS range {np.sqrt(np.mean(U[g, 7] ** 2)):.2f},"
              f" bearing {np.sqrt(np.mean(U[g, 8] ** 2)):.2f}")
    else:
        print("  none")

    print("\nInnovations per ID (accepted; measured minus predicted):")
    print("   ID     n   range mean   range sd   bearing mean   bearing sd   gated")
    for mid in sorted(slot):
        sel = U[:, 1] == mid
        a = sel & acc
        if a.any():
            print(f"  {mid:3d}  {int(a.sum()):4d}   {100 * U[a, 2].mean():+7.1f} cm"
                  f"   {100 * U[a, 2].std():6.1f} cm   {np.degrees(U[a, 3].mean()):+8.2f} deg"
                  f"   {np.degrees(U[a, 3].std()):7.2f} deg   {int((sel & ~acc).sum()):4d}")
        else:
            print(f"  {mid:3d}     0   (only the entry sighting{'s' if sel.sum() else ''};"
                  f" {int(sel.sum())} gated)")

    # map score
    ids = sorted(slot)
    est = np.array([x[slot[i]:slot[i] + 2] for i in ids])
    cov = np.array([P[slot[i]:slot[i] + 2, slot[i]:slot[i] + 2] for i in ids])
    align = args.frame == "start"
    idx = np.concatenate([[slot[i], slot[i] + 1] for i in ids]) if ids else np.array([], int)
    Pmap = P[np.ix_(idx, idx)]
    rows = score_map(ids, est, cov, align, Pmap=Pmap)
    print("\nMap score vs the tape map" + (" (rigid alignment on the other markers, leave one out,"
                                          " alignment uncertainty included):"
                                          if align else " (room frame, no alignment):"))
    print("   ID   est x   est y   err x   err y   |err|   95% ellipse axes      d2   inside")
    inside = []
    for mid, p, e, sig, d2 in rows:
        ok = d2 <= CHI2_95
        inside.append(ok)
        print(f"  {mid:3d}  {p[0]:6.3f}  {p[1]:6.3f}  {100 * e[0]:+6.1f}  {100 * e[1]:+6.1f}"
              f"  {100 * np.hypot(*e):5.1f}   {100 * 2.45 * sig[0]:5.1f} x {100 * 2.45 * sig[1]:4.1f} cm"
              f"  {d2:6.2f}   {'yes' if ok else 'NO'}")
    if rows:
        errs = np.array([np.hypot(*r[2]) for r in rows])
        print(f"  markers scored: {len(rows)}, inside their 95% ellipse: {sum(inside)} of {len(rows)}"
              f" (honest: about 95%); error mean {100 * errs.mean():.1f} cm, RMS"
              f" {100 * np.sqrt(np.mean(errs ** 2)):.1f} cm, max {100 * errs.max():.1f} cm")
        if align:
            phi, t = rigid_fit(est, np.array([E.MARKER_MAP[i][:2] for i in ids]))
            print(f"  alignment on all markers: rotate {np.degrees(phi):+.1f} deg, shift"
                  f" ({100 * t[0]:+.1f}, {100 * t[1]:+.1f}) cm about the room origin"
                  " (mostly the by-eye start placement)")
            print("  note: each marker's test includes the uncertainty of its own alignment"
                  " (v2), so the anchor marker 0 is graded fairly")

    # ------------------------------------------------------------------ plots
    tt = imu_t - t0
    phi_all, t_all = (rigid_fit(est, np.array([E.MARKER_MAP[i][:2] for i in ids]))
                      if align and len(ids) >= 2 else (0.0, np.zeros(2)))
    Ra = rot2(phi_all)
    fig, axs = plt.subplots(1, 2, figsize=(13, 10))
    for ax, aligned in ((axs[0], False), (axs[1], True)):
        E.draw_room(ax)
        for mid, (mx, my, _, _) in E.MARKER_MAP.items():
            ax.plot(mx, my, "s", color="0.7", ms=9, zorder=1)
        path_xy = Xc[:, :2] @ Ra.T + t_all if aligned else Xc[:, :2]
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="tab:blue", lw=1.0, alpha=0.8, label="car path")
        if not aligned:
            for k in range(0, n, 200):
                ax.add_patch(E.cov_ellipse(Xc[k, :2], Pc[k, :2, :2], CHI2_95,
                                           color="tab:blue", lw=0.5, alpha=0.5))
        for kk, mid in enumerate(ids):
            m, C = est[kk], cov[kk]
            if aligned:
                m, C = Ra @ m + t_all, Ra @ C @ Ra.T
                tx, ty = E.MARKER_MAP[mid][:2]
                ax.plot([m[0], tx], [m[1], ty], color="tab:red", lw=1)
            ax.plot(*m, "o", color="tab:orange", ms=5, zorder=3)
            ax.add_patch(E.cov_ellipse(m, C, CHI2_95, color="tab:orange", lw=1.3))
            ax.annotate(str(mid), m, textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.plot([], [], "s", color="0.7", label="tape-measured marker")
        ax.plot([], [], "o", color="tab:orange", label="SLAM marker, 95% ellipse")
        ax.set_xlim(-0.3, E.ROOM_W + 0.3)
        ax.set_ylim(-0.3, E.ROOM_L + 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8)
    axs[0].set_title(f"{name}: EKF-SLAM map as estimated (start frame, not aligned)\n(start pose drawn at the home X;"
                     " blue = car 95% every 2 s)", fontsize=10)
    axs[1].set_title("aligned to the tape map (red = error)" if align else "room frame vs tape (red = error)",
                     fontsize=10)
    if not align:
        for kk, mid in enumerate(ids):
            tx, ty = E.MARKER_MAP[mid][:2]
            axs[1].plot([est[kk][0], tx], [est[kk][1], ty], color="tab:red", lw=1)
    fig.tight_layout()
    map_png = os.path.join(out_dir, f"{name}_slam_map.png")
    fig.savefig(map_png, dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axs[0].plot(tt, 100 * np.sqrt(Pc[:, 0, 0]), lw=1, label="car sigma x (cm)")
    axs[0].plot(tt, 100 * np.sqrt(Pc[:, 1, 1]), lw=1, label="car sigma y (cm)")
    axs[0].plot(tt, np.degrees(np.sqrt(Pc[:, 2, 2])), lw=1, label="car sigma heading (deg)")
    axs[0].legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("car")
    # every marker's uncertainty from the 10 Hz snapshots of the full covariance, so the
    # loop-closure drop shows on markers that were not seen at that moment too
    cmap = plt.get_cmap("tab20")
    st = np.array(snaps_t) - t0
    sP = np.array(snaps_P)
    for c_i, mid in enumerate(ids):
        i = slot[mid]
        blk = sP[:, i:i + 2, i:i + 2]
        ok = np.isfinite(blk).all(axis=(1, 2))
        sig = np.full(len(st), np.nan)
        sig[ok] = np.sqrt(np.linalg.eigvalsh(blk[ok])[:, -1])
        axs[1].plot(st, 100 * sig, lw=1.2, color=cmap(c_i % 20), label=f"{mid}")
    axs[1].set_ylabel("marker sigma, major axis (cm)")
    axs[1].legend(fontsize=7, ncol=7, loc="upper left", title="marker", title_fontsize=7)
    if len(U):
        ut = U[:, 0] - t0
        axs[2].plot(ut[acc], U[acc, 4], ".", ms=2, label="accepted")
        axs[2].plot(ut[~acc], np.minimum(U[~acc, 4], 50), "x", ms=4, color="tab:red", label="gated out")
        axs[2].axhline(args.gate, color="tab:red", lw=0.6, ls=":")
        axs[2].set_ylim(0, 50)
        axs[2].legend(fontsize=8, loc="upper left")
    axs[2].set_ylabel("NIS per sighting")
    axs[2].set_xlabel("time (s)")
    for a in axs:
        for tj, *_ in revisits:
            a.axvline(tj - t0, color="tab:green", lw=0.8, ls="--")
        for tr in recoveries:
            a.axvline(tr - t0, color="tab:purple", lw=0.8, ls="--")
        a.grid(alpha=0.3)
    fig.suptitle(f"{name}: EKF-SLAM over time (green = loop closures, purple = recoveries)")
    fig.tight_layout()
    time_png = os.path.join(out_dir, f"{name}_slam_time.png")
    fig.savefig(time_png, dpi=110)
    plt.close(fig)

    slot_ids = np.full(len(E.MARKER_MAP), -1)
    for mid, i in slot.items():
        slot_ids[(i - 3) // 2] = mid
    np.savez_compressed(os.path.join(out_dir, f"{name}_slam.npz"),
                        t=imu_t, car_x=Xc, car_P=Pc, snap_t=np.array(snaps_t),
                        snap_x=np.array(snaps_x), snap_P=np.array(snaps_P), slot_ids=slot_ids,
                        map_ids=np.array(ids), map_est=est, map_cov=cov, map_P=Pmap, updates=U,
                        recoveries=np.array(recoveries), revisits=np.array([r[:3] for r in revisits]).reshape(-1, 3),
                        score_inside=np.array([r[4] <= CHI2_95 for r in rows]),
                        score_err=np.array([np.hypot(*r[2]) for r in rows]),
                        frame=args.frame)
    print(f"\nSaved {map_png}\nSaved {time_png}\nSaved {os.path.join(out_dir, name + '_slam.npz')}")


if __name__ == "__main__":
    main()
