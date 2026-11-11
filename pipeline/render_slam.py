#!/usr/bin/env python3
"""Render an EKF-SLAM replay video from <run>_slam.npz (the demo video).

Usage (Pi or laptop, with ekf_localize.py and ekf_slam.py in the same folder):
    python3 render_slam.py ~/replay/m8_run4_slam.npz
    python3 render_slam.py ~/replay/m8_run4_slam.npz --fps 10 --score "11 of 11 markers inside"

Output: <run>_slam_video.mp4 next to the input, 960 x 1080 (the right half of a
1920 x 1080 side-by-side with the phone footage).

What it draws, per frame:
  map (top):  the room and the tape-measured markers (grey squares, truth, NOT used
              by the filter); the car's path, pose and 95% ellipse (blue); every
              mapped marker with its 95% ellipse (orange); the sightings of the last
              0.3 s (green rays; red if gated out); a banner at each loop closure.
  strip (bottom): car and map uncertainty over time with a moving cursor.
The whole estimate is drawn after one rigid alignment of the final map to the tape
map (start frame), so it sits on the room drawing. The last frame is held with the
score.
Printed: the video time of the first motion, to line up with the phone footage.
"""

import argparse
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Ellipse, Polygon  # noqa: E402

import ekf_localize as E  # noqa: E402
from ekf_slam import rigid_fit, CHI2_95  # noqa: E402

RAY_DT = 0.3        # s, show sightings this recent
BANNER_DT = 3.0     # s, loop closure banner duration
REVISIT_GAP = 20.0  # s, same rule as ekf_slam.py
HOLD_S = 4.0        # s, final frame held with the score
BLUE, ORANGE, GREEN, RED = "#1f6fb4", "#e8780c", "#2ca02c", "#d62728"


def ell_params(cov):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return 2 * np.sqrt(CHI2_95 * vals[1]), 2 * np.sqrt(CHI2_95 * vals[0]), ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="<run>_slam.npz written by ekf_slam.py")
    ap.add_argument("--fps", type=float, default=30.0, help="video frame rate (10 renders 3x faster)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--score", default=None, help="score text for the final frame (default: from the npz)")
    ap.add_argument("--title", default="EKF-SLAM: mapping the room from scratch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = os.path.expanduser(args.npz)
    d = np.load(path, allow_pickle=False)
    name = os.path.basename(path).replace("_slam.npz", "")
    out = args.out or os.path.join(os.path.dirname(path), f"{name}_slam_video.mp4")

    t, car_x, car_P = d["t"], d["car_x"], d["car_P"]
    st, sx, sP = d["snap_t"], d["snap_x"], d["snap_P"]
    slot_ids = d["slot_ids"]
    U = d["updates"]
    map_ids, map_est = d["map_ids"], d["map_est"]
    frame = str(d["frame"])
    t0 = t[0]

    # one rigid alignment of the final map to the tape map, applied to everything
    if frame == "start" and len(map_ids) >= 2:
        tape = np.array([E.MARKER_MAP[int(i)][:2] for i in map_ids])
        phi, tr = rigid_fit(map_est, tape)
    else:
        phi, tr = 0.0, np.zeros(2)
    Ra = E.rot2(phi)

    def to_room(p):
        return p @ Ra.T + tr

    car_xy = to_room(car_x[:, :2])
    car_th = car_x[:, 2] + phi

    # loop closures: accepted update on a marker after REVISIT_GAP s without one
    revisits = []
    last = {}
    for k in range(len(st)):  # entry time of each marker from the snapshots
        for s_i, mid in enumerate(slot_ids):
            if mid >= 0 and mid not in last and np.isfinite(sx[k, 3 + 2 * s_i]):
                last[int(mid)] = st[k]
    for row in U:
        tj, mid, ok = row[0], int(row[1]), row[5] == 1
        if ok:
            if tj - last.get(mid, tj) >= REVISIT_GAP:
                revisits.append((tj, mid, tj - last[mid]))
            last[mid] = tj

    # uncertainty curves from the snapshots
    car_sig = np.sqrt(np.maximum(sP[:, 0, 0], sP[:, 1, 1]))
    map_sig = np.full(len(st), np.nan)
    for k in range(len(st)):
        sig = []
        for s_i, mid in enumerate(slot_ids):
            i = 3 + 2 * s_i
            blk = sP[k, i:i + 2, i:i + 2]
            if mid >= 0 and np.isfinite(blk).all():
                sig.append(np.sqrt(np.linalg.eigvalsh(blk)[-1]))
        if sig:
            map_sig[k] = np.mean(sig)
    stt = st - t0

    moved = np.flatnonzero(np.hypot(*(car_x[:, :2] - car_x[0, :2]).T) > 0.002)
    t_move = t[moved[0]] - t0 if len(moved) else 0.0

    if args.score is None and "score_inside" in d.files:
        si, se = d["score_inside"], d["score_err"]
        args.score = (f"{int(si.sum())} of {len(si)} markers inside their 95% ellipse\n"
                      f"map error RMS {100 * np.sqrt(np.mean(se ** 2)):.1f} cm")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(9.6, 10.8), dpi=100)
    ax = fig.add_axes([0.07, 0.25, 0.90, 0.68])
    ax2 = fig.add_axes([0.10, 0.05, 0.86, 0.14])
    fig.text(0.5, 0.975, args.title, ha="center", va="top", fontsize=17, weight="bold")
    E.draw_room(ax)
    for mid, (mx, my, _, _) in E.MARKER_MAP.items():
        ax.plot(mx, my, "s", color="0.75", ms=11, zorder=1)
    ax.set_xlim(-0.3, E.ROOM_W + 0.3)
    ax.set_ylim(-0.3, E.ROOM_L + 0.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.plot([], [], "s", color="0.75", ms=9, label="tape-measured marker (not used by the filter)")
    ax.plot([], [], "o", color=ORANGE, label="SLAM marker estimate, 95% ellipse")
    ax.plot([], [], "-", color=BLUE, label="car path and 95% ellipse")
    ax.plot([], [], "-", color=GREEN, label="camera sighting")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=2, fontsize=9, frameon=False)

    path_ln, = ax.plot([], [], color=BLUE, lw=1.6, alpha=0.8, zorder=2)
    car_ell = Ellipse((0, 0), 0, 0, fill=True, fc=BLUE, alpha=0.18, ec=BLUE, lw=1.5, zorder=4)
    ax.add_patch(car_ell)
    car_tri = Polygon(np.zeros((3, 2)), closed=True, fc=BLUE, ec="k", lw=0.8, zorder=6)
    ax.add_patch(car_tri)
    rays = LineCollection([], lw=1.2, zorder=3)
    ax.add_collection(rays)
    m_ell, m_dot, m_lbl = {}, {}, {}
    for mid in E.MARKER_MAP:
        e = Ellipse((0, 0), 0, 0, fill=True, fc=ORANGE, alpha=0.25, ec=ORANGE, lw=1.5, zorder=4,
                    visible=False)
        ax.add_patch(e)
        m_ell[mid] = e
        m_dot[mid], = ax.plot([], [], "o", color=ORANGE, ms=5, zorder=5)
        m_lbl[mid] = ax.text(0, 0, str(mid), fontsize=10, color="0.2", visible=False, zorder=7)
    clock = ax.text(0.02, 0.015, "", transform=ax.transAxes, fontsize=12, family="monospace")
    count = ax.text(0.98, 0.015, "", transform=ax.transAxes, fontsize=12, ha="right")
    banner = ax.text(0.5, 0.5, "", transform=ax.transAxes, fontsize=16, ha="center", va="center",
                     weight="bold", color="white", zorder=10,
                     bbox=dict(boxstyle="round,pad=0.5", fc=GREEN, ec="none", alpha=0.92))
    banner.set_visible(False)
    final = ax.text(0.5, 0.42, "", transform=ax.transAxes, fontsize=16, ha="center", va="center",
                    weight="bold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="k", alpha=0.95))
    final.set_visible(False)

    ax2.plot(stt, 100 * car_sig, color=BLUE, lw=0.8, alpha=0.2)
    ax2.plot(stt, 100 * map_sig, color=ORANGE, lw=0.8, alpha=0.2)
    c_ln, = ax2.plot([], [], color=BLUE, lw=1.8, label="car position")
    m_ln, = ax2.plot([], [], color=ORANGE, lw=1.8, label="map (mean over markers)")
    cursor = ax2.axvline(0, color="k", lw=1)
    for tj, _, _ in revisits:
        ax2.axvline(tj - t0, color=GREEN, lw=1, ls="--")
    ax2.set_xlim(0, stt[-1])
    top = np.nanpercentile(np.concatenate([car_sig, map_sig[np.isfinite(map_sig)]]), 99) * 100
    ax2.set_ylim(0, max(5.0, 1.1 * top))
    ax2.set_ylabel("uncertainty\n1 sigma (cm)", fontsize=10)
    ax2.set_xlabel("time (s)   green dashes = loop closures", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)
    ax2.grid(alpha=0.3)

    tri = np.array([[0.13, 0.0], [-0.07, 0.07], [-0.07, -0.07]])

    def draw(tv):
        k = min(np.searchsorted(t, tv), len(t) - 1)
        ks = max(np.searchsorted(st, tv, side="right") - 1, 0)
        path_ln.set_data(car_xy[:k + 1, 0], car_xy[:k + 1, 1])
        p, th = car_xy[k], car_th[k]
        car_tri.set_xy(p + tri @ E.rot2(th).T)
        w, h, a = ell_params(Ra @ car_P[k, :2, :2] @ Ra.T)
        car_ell.set_center(p)
        car_ell.set_width(w)
        car_ell.set_height(h)
        car_ell.set_angle(a + 0.0)
        n_map = 0
        for s_i, mid in enumerate(slot_ids):
            if mid < 0:
                continue
            i = 3 + 2 * s_i
            m = sx[ks, i:i + 2]
            if not np.isfinite(m).all():
                continue
            n_map += 1
            mr = to_room(m)
            w, h, a = ell_params(Ra @ sP[ks, i:i + 2, i:i + 2] @ Ra.T)
            e = m_ell[int(mid)]
            e.set_center(mr)
            e.set_width(w)
            e.set_height(h)
            e.set_angle(a)
            e.set_visible(True)
            m_dot[int(mid)].set_data([mr[0]], [mr[1]])
            m_lbl[int(mid)].set_position((mr[0] + 0.05, mr[1] + 0.05))
            m_lbl[int(mid)].set_visible(True)
        # recent sightings: rays from the car to the marker estimate
        sel = (U[:, 0] <= tv) & (U[:, 0] > tv - RAY_DT)
        segs, cols = [], []
        seen = set()
        for row in U[sel][::-1]:
            mid = int(row[1])
            if mid in seen:
                continue
            seen.add(mid)
            s_i = np.flatnonzero(slot_ids == mid)
            if not len(s_i):
                continue
            m = sx[ks, 3 + 2 * s_i[0]:5 + 2 * s_i[0]]
            if np.isfinite(m).all():
                segs.append([p, to_room(m)])
                cols.append(GREEN if row[5] == 1 else RED)
        rays.set_segments(segs)
        rays.set_color(cols if cols else GREEN)
        clock.set_text(f"t = {tv - t0:5.1f} s")
        count.set_text(f"markers mapped: {n_map}")
        act = [r for r in revisits if 0 <= tv - r[0] < BANNER_DT]
        if act:
            tj, mid, gap = act[-1]
            banner.set_text(f"Loop closure: marker {mid} seen again\nafter {gap:.0f} s,"
                            " uncertainty shrinks")
            banner.set_visible(True)
        else:
            banner.set_visible(False)
        c_ln.set_data(stt[:ks + 1], 100 * car_sig[:ks + 1])
        m_ln.set_data(stt[:ks + 1], 100 * map_sig[:ks + 1])
        cursor.set_xdata([tv - t0, tv - t0])

    times = np.arange(t[0], t[-1], args.speed / args.fps)
    n_hold = int(HOLD_S * args.fps)
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast"])
    tic = time.perf_counter()
    print(f"Rendering {len(times)} frames + {n_hold} held at {args.fps:g} fps to {out}")
    with writer.saving(fig, out, dpi=100):
        for f_i, tv in enumerate(times):
            draw(tv)
            writer.grab_frame()
            if f_i % 300 == 0 and f_i:
                el = time.perf_counter() - tic
                print(f"  {f_i}/{len(times)} frames, {el:.0f} s, about"
                      f" {el / f_i * (len(times) - f_i):.0f} s left")
        draw(t[-1])
        banner.set_visible(False)
        rays.set_segments([])
        if args.score:
            final.set_text(args.score)
            final.set_visible(True)
        for _ in range(n_hold):
            writer.grab_frame()
    print(f"Done in {time.perf_counter() - tic:.0f} s")
    print(f"Loop closures: " + (", ".join(f"marker {m} at {tj - t0:.1f} s (gap {g:.0f} s)"
                                           for tj, m, g in revisits) or "none"))
    print(f"SYNC: the car first moves at {t_move / args.speed:.2f} s into this video."
          " Line that up with the first motion in the phone footage.")


if __name__ == "__main__":
    main()
