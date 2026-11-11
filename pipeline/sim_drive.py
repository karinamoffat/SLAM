#!/usr/bin/env python3
"""Simulated drive with known ground truth, for testing the filters before real data.

Writes a .npz in the same format as replay_extract.py, so ekf_localize.py and
ekf_slam.py run on it unchanged. The true marker positions are the tape map
(ekf_localize.MARKER_MAP), so the map score that ekf_slam.py prints is a score
against the truth. The true car path is saved too, and --check grades the
filter's car pose against it.

Usage:
    python3 sim_drive.py sim1.npz --seed 1        # write a simulated drive
    python3 ekf_slam.py sim1.npz                  # run SLAM on it as usual
    python3 sim_drive.py sim1.npz --check         # car pose vs truth (needs sim1_slam.npz)

What is simulated (the U route of the real demo drives, out and back):
  Motion   forward at 0.40 m/s (5% speed error per segment) with a left pull of
           about 14 deg/m; braked pivots at about 23 deg/s (10% per pivot) about the
           measured pivot points; random along-track, sideways and pivot slip drawn
           from the filter's base process noise (before the x4 real-drive scale).
  Gyro     true yaw rate read 3% low, plus a 1.4 deg/s bias and 0.026 deg/s noise
           per sample at 100 Hz.
  Camera   10 Hz; a marker is seen if it is 0.6 to 2.5 m away, within 33 deg of the
           optical axis, and the car is in front of the marker's face (within
           70 deg of its normal); 90% detection rate; range and bearing noise from
           the measurement model (sigma_r = 0.005 + 0.0035 r^2 m, sigma_b = 1 deg).

Not simulated: stalls, rubbing, hand corrections, occlusion by furniture, misreads,
correlated (biased) sightings. Real drives are harder; that gap is the point of
scoring on real data afterwards.
"""

import argparse
import os

import numpy as np

import ekf_localize as E

DT = 0.01                       # s, IMU period
CAM_DT = 0.1                    # s, camera period
SPEED = 0.40                    # m/s
W_PIVOT = np.radians(23.0)      # rad/s, braked pivot
PULL = np.radians(14.0)         # rad per m driven (left pull)
GYRO_SCALE = 1.03               # true rate = 1.03 x measured
GYRO_BIAS = np.radians(1.40)
GYRO_SIGMA = np.radians(0.026)
HALF_FOV = np.radians(33.0)
MAX_VIEW = np.radians(70.0)
P_DETECT = 0.9
AIM_EVERY = 0.5             # m driven before the driver re-aims
REACHED = 0.30              # m, close enough to a waypoint to move on

# Direction each marker faces (deg from +x, counterclockwise), from the room survey.
FACES = {0: 90, 1: 0, 2: -90, 3: -90, 4: 180, 5: -45, 6: 135, 7: 180,
         8: 90, 9: 90, 10: 0, 11: -90, 12: -90}

# U route: north zone, east corridor, south zone, turn around, and back home.
WAYPOINTS = [(2.45, 4.30), (2.50, 3.40), (2.45, 1.80), (1.50, 1.10), (0.70, 1.10),
             (1.50, 1.10), (2.45, 1.80), (2.50, 3.40), (2.30, 4.40), (1.603, 4.762)]


def visible(x, mid, rng_gen):
    mx, my = E.MARKER_MAP[mid][:2]
    dx, dy = mx - x[0], my - x[1]
    r = np.hypot(dx, dy)
    b = E.wrap(np.arctan2(dy, dx) - x[2])
    if not (E.MIN_RANGE <= r <= E.MAX_RANGE) or abs(b) > HALF_FOV:
        return None
    n = np.radians(FACES[mid])
    to_car = np.array([-dx, -dy]) / r
    if to_car @ np.array([np.cos(n), np.sin(n)]) < np.cos(MAX_VIEW):
        return None
    if rng_gen.random() > P_DETECT:
        return None
    return r, b


def simulate(seed):
    """Drive the U route like a person would: pivot toward the next waypoint, drive at
    most AIM_EVERY metres, re-aim (this corrects the left pull), repeat."""
    g = np.random.default_rng(seed)
    x = np.array(E.HOME_POSE, dtype=float)
    st = dict(t=100.0, next_cam=100.0)
    imu_t, imu_wz, cmd_t, cmd_v, cmd_w, truth = [], [], [], [], [], []
    obs = []

    def run(kind, sgn, dur):
        speed = SPEED * (1 + 0.05 * g.standard_normal()) if kind == "fwd" else 0.0
        w_rate = W_PIVOT * (1 + 0.10 * g.standard_normal()) if kind == "pivot" else 0.0
        for i in range(max(1, int(round(dur / DT)))):
            st["t"] += DT
            t = st["t"]
            if kind == "fwd":
                ds = speed * DT
                dth = PULL * ds
                thm = x[2] + 0.5 * dth
                along = ds + np.sqrt(E.K_ALONG * ds) * g.standard_normal()
                side = np.sqrt(E.K_LAT * ds) * g.standard_normal()
                x[0] += along * np.cos(thm) - side * np.sin(thm)
                x[1] += along * np.sin(thm) + side * np.cos(thm)
                v_cmd, w_cmd = 0.42, 0.0
            elif kind == "pivot":
                dth = sgn * w_rate * DT
                c = np.array(E.PIVOT_C[int(sgn)])
                x[:2] += E.rot2(x[2]) @ c - E.rot2(x[2] + dth) @ c
                x[:2] += np.sqrt(E.K_PIVOT * abs(dth)) * g.standard_normal(2)
                v_cmd, w_cmd = 0.0, sgn * 0.40
            else:
                dth = 0.0
                v_cmd, w_cmd = 0.0, 0.0
            dth += np.sqrt(E.K_TURN * abs(dth)) * g.standard_normal()
            x[2] = E.wrap(x[2] + dth)
            imu_t.append(t)
            imu_wz.append(dth / DT / GYRO_SCALE + GYRO_BIAS + GYRO_SIGMA * g.standard_normal())
            truth.append(x.copy())
            if i == 0 or i % 10 == 0:   # /cmd_applied at the key press, then 10 Hz
                cmd_t.append(t)
                cmd_v.append(v_cmd)
                cmd_w.append(w_cmd)
            if t >= st["next_cam"]:
                st["next_cam"] += CAM_DT
                for mid in E.MARKER_MAP:
                    vis = visible(x, mid, g)
                    if vis is None:
                        continue
                    r, b = vis
                    sr = 0.005 + 0.0035 * r * r
                    sb = np.radians(1.0)
                    obs.append((t, mid, r + sr * g.standard_normal(),
                                E.wrap(b + sb * g.standard_normal()), sr, sb))

    def pivot_to(heading):
        dth = E.wrap(heading - x[2])
        if abs(dth) > np.radians(5.0):
            run("pivot", np.sign(dth), abs(dth) / W_PIVOT)
            run("stop", 0, 1.0)

    run("stop", 0, 12.0)
    for w in WAYPOINTS:
        for _ in range(6):    # a person stops correcting after a few tries
            d = np.array(w) - x[:2]
            dist = np.hypot(*d)
            if dist < REACHED:
                break
            pivot_to(np.arctan2(d[1], d[0]))
            run("fwd", 1, min(dist, AIM_EVERY) / SPEED)
            run("stop", 0, 0.5)
    pivot_to(E.HOME_POSE[2])
    run("stop", 0, 12.0)

    obs = np.array(obs, dtype=float).reshape(-1, 6)
    return dict(
        imu_t=np.array(imu_t), imu_wz=np.array(imu_wz),
        cmd_t=np.array(cmd_t), cmd_v=np.array(cmd_v), cmd_w=np.array(cmd_w),
        obs_t=obs[:, 0], obs_id=obs[:, 1].astype(int), obs_range=obs[:, 2],
        obs_bearing=obs[:, 3], obs_sigma_r=obs[:, 4], obs_sigma_b=obs[:, 5],
        obs_in_model=np.ones(len(obs), bool),
        truth_t=np.array(imu_t), truth_x=np.array(truth), seed=seed,
    )


def check(path):
    """Car pose from <name>_slam.npz vs the simulated truth: error and 95% coverage."""
    d = np.load(path)
    s = np.load(path.replace(".npz", "_slam.npz"))
    if str(s["frame"]) != "start":
        print("note: run ekf_slam.py with --frame start (the default) for this check")
    tt, xt = d["truth_t"], d["truth_x"]
    xe, Pe = s["car_x"], s["car_P"]
    e = xe[:, :2] - xt[:, :2]
    eth = E.wrap(xe[:, 2] - xt[:, 2])
    d2 = np.einsum("ni,nij,nj->n", e, np.linalg.inv(Pe[:, :2, :2] + 1e-12 * np.eye(2)), e)
    k = slice(None, None, 100)   # once per second
    print(f"Car position error vs truth: RMS {100 * np.sqrt(np.mean(np.sum(e ** 2, 1))):.1f} cm,"
          f" max {100 * np.max(np.hypot(*e.T)):.1f} cm; heading RMS"
          f" {np.degrees(np.sqrt(np.mean(eth ** 2))):.1f} deg")
    print(f"Car position inside its 95% ellipse (sampled 1 Hz): {100 * np.mean(d2[k] <= 5.99):.0f}%"
          " (honest: about 95%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .npz (replay format), e.g. sim1.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="grade the car pose in <out>_slam.npz against the truth")
    a = ap.parse_args()
    path = os.path.expanduser(a.out)
    if a.check:
        check(path)
        return
    d = simulate(a.seed)
    np.savez(path, **d)
    print(f"Saved {path}: {d['imu_t'][-1] - d['imu_t'][0]:.0f} s, {len(d['obs_t'])} sightings,"
          f" IDs seen {sorted(set(d['obs_id'].tolist()))}")


if __name__ == "__main__":
    main()
