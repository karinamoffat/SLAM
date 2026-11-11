#!/usr/bin/env python3
"""Camera-to-car transform from a checkerboard flat on the floor.

The camera node must be running (with the focus flags). Then on the Pi:
    python3 cam_extrinsics.py --front-line 9.5 --gap-left 12.0 --gap-right 12.0 \
        --over-left 3.0 --over-right 3.0

Board placement (all distances in cm, measured to the PRINTED PATTERN's outer
edge, not the paper or cardboard edge):
- Board flat on the floor in front of the car, long side (10 squares,
  23.75 cm) left to right, short side (7 squares, 16.625 cm) pointing forward.
- A straight edge on the floor touching the front of both front tires is the
  "front line". --front-line = distance from the base_link origin forward to
  that line (half the wheelbase plus the front wheel radius).
- --gap-left / --gap-right: front line to the pattern's near edge, measured
  at the pattern's left and right ends. Equal gaps = board square to the car.
- --over-left / --over-right: how far each end of the pattern sticks out past
  the outer face of the front tire on that side (negative if it falls short).

What it computes, per frame:
  solvePnP -> board pose in the camera frame (raw image, so K and D are used)
  floor normal -> camera tilt, roll, height
  pattern centre and edge direction + measured placement -> yaw, x, y
Only quantities that survive a 180 degree flip of the corner order are used.
"""
import argparse
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

PATTERN = (9, 6)            # inner corners
SQUARE = 0.02375            # m, measured (printed at 95%)
PATTERN_W = 10 * SQUARE     # 0.2375 m, left to right
PATTERN_D = 7 * SQUARE      # 0.16625 m, forward depth

# Hand measurements of the lens in base_link (m) with 1-sigma, Oct 24 2026.
HAND = {"x": (0.090, 0.007), "y": (0.002, 0.005), "z": (0.137, 0.003)}
PLACEMENT_SIGMA = 0.005     # m, 1-sigma for measuring where the board sits

OBJ = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
OBJ[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE
CENTER = np.array([(PATTERN[0] - 1) / 2 * SQUARE, (PATTERN[1] - 1) / 2 * SQUARE, 0.0])


def image_to_gray(msg):
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    rows = buf.reshape(msg.height, msg.step)
    if enc in ("mono8", "8uc1"):
        return rows[:, :msg.width].copy()
    if enc in ("bgr8", "rgb8"):
        img = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if enc == "bgr8" else cv2.COLOR_RGB2GRAY)
    if enc in ("bgra8", "rgba8"):
        img = rows[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if enc == "bgra8" else cv2.COLOR_RGBA2GRAY)
    raise RuntimeError(f"unsupported image encoding '{msg.encoding}'")


def solve_frame(gray, K, D, board_center_base):
    """Returns (R_bc, t_bc, reprojection_error_px) or None.

    R_bc maps vectors in the camera frame to base_link. t_bc is the camera
    origin (lens) in base_link.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
    ok, rvec, tvec = cv2.solvePnP(OBJ, corners, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(OBJ, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj - corners) ** 2, axis=2))))

    R, _ = cv2.Rodrigues(rvec)
    center_cam = R @ CENTER + tvec.ravel()

    # Floor normal, pointing up (toward the camera).
    up = R[:, 2].copy()
    if up @ center_cam > 0:
        up = -up
    # The board edge direction closest to the car's left (optical -x).
    cands = [R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]]
    left = max(cands, key=lambda v: float(v @ np.array([-1.0, 0.0, 0.0])))
    left = left - (left @ up) * up
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)

    R_bc = np.vstack([fwd, left, up])   # rows: base_link axes in camera coords
    t_bc = board_center_base - R_bc @ center_cam
    return R_bc, t_bc, err


def camera_angles(R_bc):
    """Tilt down, yaw left, roll (right side of image lower), in degrees."""
    optical = R_bc @ np.array([0.0, 0.0, 1.0])
    right = R_bc @ np.array([1.0, 0.0, 0.0])
    tilt = math.degrees(math.asin(-optical[2]))
    yaw = math.degrees(math.atan2(optical[1], optical[0]))
    roll = math.degrees(math.asin(-right[2]))
    return tilt, yaw, roll


def rot_to_quat(R):
    """Rotation matrix to quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return x, y, z, w


class Grabber(Node):
    def __init__(self, n_frames, board_center_base):
        super().__init__("cam_extrinsics")
        self.n = n_frames
        self.board = board_center_base
        self.K = self.D = None
        self.frame_id = None
        self.results = []
        self.seen = 0
        self.last_print = time.monotonic()
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw", self.on_image,
                                 qos_profile_sensor_data)

    @property
    def done(self):
        return len(self.results) >= self.n

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.d, dtype=np.float64)
            self.frame_id = msg.header.frame_id
            self.get_logger().info(
                f"camera_info: fx={self.K[0, 0]:.1f} fy={self.K[1, 1]:.1f} "
                f"frame_id='{self.frame_id}'")

    def on_image(self, msg):
        if self.K is None or self.done:
            return
        self.seen += 1
        res = solve_frame(image_to_gray(msg), self.K, self.D, self.board)
        if res is not None:
            self.results.append(res)
        now = time.monotonic()
        if now - self.last_print > 1.0:
            self.last_print = now
            status = "board FOUND" if res is not None else "board not found"
            print(f"  {status}: {len(self.results)}/{self.n} good frames "
                  f"({self.seen} processed)")


def report(results, frame_id):
    ts = np.array([r[1] for r in results])
    errs = np.array([r[2] for r in results])
    angs = np.array([camera_angles(r[0]) for r in results])

    U, _, Vt = np.linalg.svd(sum(r[0] for r in results))
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    t_mean = ts.mean(axis=0)
    tilt, yaw, roll = camera_angles(R_mean)

    print(f"\n{len(results)} frames, reprojection error {errs.mean():.2f} px mean "
          f"({errs.max():.2f} max)")
    print("\nCamera (lens) position in base_link, checkerboard vs hand:")
    print(f"  {'':3}{'board (cm)':>18}{'hand (cm)':>16}{'diff':>8}{'diff/sigma':>12}")
    placement = {"x": PLACEMENT_SIGMA, "y": PLACEMENT_SIGMA, "z": 0.0}
    for i, ax in enumerate("xyz"):
        hv, hs = HAND[ax]
        bv, bs = t_mean[i], ts[:, i].std()
        diff = bv - hv
        sigma = math.sqrt(hs ** 2 + placement[ax] ** 2)
        print(f"  {ax}: {bv * 100:8.2f} +- {bs * 100:4.2f}   {hv * 100:6.2f} +- {hs * 100:3.1f}"
              f"{diff * 100:+8.2f}{diff / sigma:+12.1f}")
    print("  (+- after the board value = frame-to-frame spread, pixel noise only)")

    print("\nCamera orientation:")
    print(f"  tilt down {tilt:6.2f} deg  (spread {angs[:, 0].std():.2f})")
    print(f"  yaw left  {yaw:6.2f} deg  (spread {angs[:, 1].std():.2f})")
    print(f"  roll      {roll:6.2f} deg  (spread {angs[:, 2].std():.2f})")

    qx, qy, qz, qw = rot_to_quat(R_mean)
    child = frame_id or "camera"
    print("\nStatic transform for tf2 (base_link -> camera frame):")
    print(f"  ros2 run tf2_ros static_transform_publisher "
          f"--x {t_mean[0]:.4f} --y {t_mean[1]:.4f} --z {t_mean[2]:.4f} "
          f"--qx {qx:.5f} --qy {qy:.5f} --qz {qz:.5f} --qw {qw:.5f} "
          f"--frame-id base_link --child-frame-id {child}")


def main():
    p = argparse.ArgumentParser(description="Camera-to-car transform from a floor checkerboard")
    p.add_argument("--front-line", type=float, required=True,
                   help="cm, base_link origin to the line touching the front tires")
    p.add_argument("--gap-left", type=float, required=True, help="cm, front line to pattern, left end")
    p.add_argument("--gap-right", type=float, required=True, help="cm, front line to pattern, right end")
    p.add_argument("--over-left", type=float, required=True, help="cm, pattern overhang past left tire")
    p.add_argument("--over-right", type=float, required=True, help="cm, pattern overhang past right tire")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--timeout", type=float, default=60.0)
    a = p.parse_args()

    gap = (a.gap_left + a.gap_right) / 2 / 100
    x_b = a.front_line / 100 + gap + PATTERN_D / 2
    y_b = (a.over_left - a.over_right) / 2 / 100
    skew = math.degrees(math.atan2((a.gap_right - a.gap_left) / 100, PATTERN_W))
    print(f"Board pattern centre in base_link: x={x_b * 100:.1f} cm, y={y_b * 100:+.1f} cm")
    print(f"Board squareness: {skew:+.2f} deg (this error goes straight into the camera yaw)")
    if abs(skew) > 1.0:
        print("  WARNING: the board is more than 1 degree off square. Adjust until the "
              "left and right gaps match within about 3 mm.")

    rclpy.init()
    node = Grabber(a.frames, np.array([x_b, y_b, 0.0]))
    start = time.monotonic()
    try:
        while rclpy.ok() and not node.done and time.monotonic() - start < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    results, frame_id, got_info = node.results, node.frame_id, node.K is not None
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if not got_info:
        print("\nNo camera_info received. Is the camera node running?")
    elif len(results) < 5:
        print(f"\nOnly {len(results)} good frames. Check that the whole pattern is in view.")
    else:
        report(results, frame_id)


if __name__ == "__main__":
    main()
