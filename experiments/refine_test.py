#!/usr/bin/env python3
"""Compare ArUco corner refinement methods on the same live frames.

Kept as run on Oct 25, 2026 (its own copies of the helpers and the camera transform
of that day), because it is how the edge correction in pipeline/marker_measure.py
was found. Results: logs/cov_stops.csv.

Hold the car still in front of a marker. The script grabs N frames, runs
detection with several corner refinement methods on identical frames,
and reports the planar range and bearing in base_link for each, so a
distance dependent range bias can be traced to corner localisation.

The extra row "apriltag+edge" pushes each AprilTag-refined edge outward
by --edge-px pixels (along its normal) before solvePnP, to correct a
constant inward bias in where the detector places the marker's edges.

Run on the Pi with the camera node running:
    python3 refine_test.py --size-cm 14.6 --expected 2.218
"""
import argparse
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

# Camera-to-car transform as of Oct 24, 2026 (parent base_link, child camera).
Q_XYZW = (0.56030, -0.54745, 0.42537, -0.45324)
T_BASE_CAM = np.array([0.0877, -0.0030, 0.1519])


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


R_BASE_CAM = quat_to_R(*Q_XYZW)

# name: (refinement method, subpix window half-size or None for default)
A = cv2.aruco
METHODS = {
    "none": (A.CORNER_REFINE_NONE, None),
    "subpix_w5": (A.CORNER_REFINE_SUBPIX, 5),
    "subpix_w2": (A.CORNER_REFINE_SUBPIX, 2),
    "contour": (A.CORNER_REFINE_CONTOUR, None),
    "apriltag": (A.CORNER_REFINE_APRILTAG, None),
}


def to_gray(msg):
    """Convert a sensor_msgs/Image to grayscale without cv_bridge."""
    ch = max(1, msg.step // msg.width)
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if ch == 1:
        return buf[:, :msg.width].copy()
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    rgb_order = msg.encoding.lower().startswith("rgb")
    if ch == 4:
        code = cv2.COLOR_RGBA2GRAY if rgb_order else cv2.COLOR_BGRA2GRAY
    else:
        code = cv2.COLOR_RGB2GRAY if rgb_order else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(img, code)


def expand_quad(c, d):
    """Move each edge of a 4-corner quad outward by d pixels, re-intersect.

    Edge i runs from corner i to corner i+1, so corner i is where edges
    i-1 and i meet. Works for any convex quad, including oblique views.
    """
    centroid = c.mean(axis=0)
    lines = []
    for i in range(4):
        p, q = c[i], c[(i + 1) % 4]
        t = q - p
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        if np.dot(centroid - p, n) > 0:  # make n point outward
            n = -n
        lines.append((n, np.dot(n, p) + d))  # line: n . x = offset
    out = np.empty_like(c)
    for i in range(4):
        (n1, o1), (n2, o2) = lines[i - 1], lines[i]
        out[i] = np.linalg.solve(np.array([n1, n2]), np.array([o1, o2]))
    return out


class Grabber(Node):
    def __init__(self, n_frames):
        super().__init__("refine_test")
        self.n_frames = n_frames
        self.frames = []
        self.K = None
        self.D = None
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw",
                                 self.on_image, qos_profile_sensor_data)

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)

    def on_image(self, msg):
        if self.K is not None and len(self.frames) < self.n_frames:
            self.frames.append(to_gray(msg))


def measure(corners, obj, K, D):
    """Corners (4x2, raw pixels) to planar range (m), bearing (deg), side (px)."""
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    p = R_BASE_CAM @ tvec.ravel() + T_BASE_CAM
    rng = float(np.hypot(p[0], p[1]))
    brg = float(np.degrees(np.arctan2(p[1], p[0])))
    side = float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
    return rng, brg, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-cm", type=float, required=True,
                    help="measured side of the black square in cm")
    ap.add_argument("--id", type=int, default=0, help="marker ID to measure")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--expected", type=float, default=None,
                    help="tape range in m (front line distance + 0.118)")
    ap.add_argument("--edge-px", type=float, default=0.25,
                    help="outward edge correction for the apriltag+edge row")
    ap.add_argument("--label", default="",
                    help="stop name, e.g. A-centre, saved in the log")
    ap.add_argument("--log", default=None,
                    help="CSV file to append one row per method, e.g. cov_stops.csv")
    args = ap.parse_args()

    s = args.size_cm / 100.0
    obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0],
                    [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], dtype=np.float64)

    dictionary = A.getPredefinedDictionary(A.DICT_4X4_50)
    detectors = {}
    for name, (method, win) in METHODS.items():
        p = A.DetectorParameters()
        p.cornerRefinementMethod = method
        if win is not None:
            p.cornerRefinementWinSize = win
        detectors[name] = A.ArucoDetector(dictionary, p)

    rclpy.init()
    node = Grabber(args.frames)
    print(f"Grabbing {args.frames} frames. Keep the car still.")
    t_end = time.time() + 30.0
    while len(node.frames) < args.frames and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.5)
    K, D, frames = node.K, node.D, node.frames
    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print("No frames received. Is the camera node running?")
        return
    print(f"Got {len(frames)} frames.\n")

    rows = {}
    for name, det in detectors.items():
        times = []
        rows[name] = {"res": [], "times": times}
        if name == "apriltag":
            rows["apriltag+edge"] = {"res": [], "times": times}
        for g in frames:
            t0 = time.perf_counter()
            corners, ids, _ = det.detectMarkers(g)
            times.append((time.perf_counter() - t0) * 1000.0)
            if ids is None:
                continue
            for c, i in zip(corners, ids.ravel()):
                if i != args.id:
                    continue
                c4 = c.reshape(4, 2).astype(np.float64)
                m = measure(c4, obj, K, D)
                if m is not None:
                    rows[name]["res"].append(m)
                if name == "apriltag":
                    m2 = measure(expand_quad(c4, args.edge_px), obj, K, D)
                    if m2 is not None:
                        rows["apriltag+edge"]["res"].append(m2)

    header = f"{'method':<14}{'n':>4}{'side px':>9}{'range m':>10}{'std mm':>8}{'bearing':>9}{'ms':>7}"
    if args.expected is not None:
        header += f"{'error mm':>10}"
    print(header)
    log_rows = []
    for name, row in rows.items():
        if not row["res"]:
            print(f"{name:<14}{0:>4}   marker not detected")
            continue
        r = np.array(row["res"])
        log_rows.append([args.label, name, len(r), f"{r[:, 2].mean():.2f}",
                         f"{r[:, 0].mean():.4f}", f"{r[:, 0].std() * 1000:.2f}",
                         f"{r[:, 1].mean():.3f}", f"{r[:, 1].std():.3f}",
                         args.edge_px, time.strftime("%Y-%m-%d %H:%M:%S")])
        line = (f"{name:<14}{len(r):>4}{r[:, 2].mean():>9.1f}{r[:, 0].mean():>10.3f}"
                f"{r[:, 0].std() * 1000:>8.1f}{r[:, 1].mean():>+8.1f}d"
                f"{np.mean(row['times']):>7.1f}")
        if args.expected is not None:
            line += f"{(r[:, 0].mean() - args.expected) * 1000:>+10.0f}"
        print(line)

    if args.log:
        import csv
        import os
        new_file = not os.path.exists(args.log)
        with open(args.log, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["label", "method", "n", "side_px", "range_m",
                            "range_std_mm", "bearing_deg", "bearing_std_deg",
                            "edge_px", "time"])
            w.writerows(log_rows)
        print(f"\nLogged to {args.log} as '{args.label}'.")


if __name__ == "__main__":
    main()
