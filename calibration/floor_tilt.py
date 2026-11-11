#!/usr/bin/env python3
"""Measure the camera tilt and roll from the floor-to-baseboard line.

Setup: car centred on the home X, facing east, square to the cabinets,
picar.launch.py running. The floor line under the cabinet fronts is then a
straight horizontal line 1.397 m (4 ft 7 in) ahead of base_link.

The script grabs one raw frame, finds the floor line in every column (bright
baseboard above, dark floor below), undistorts the points with K and D, and
fits camera tilt and roll so the projected floor line matches. Yaw and the
lens position are kept from Oct 24 (yaw is checked separately with markers).

Prints the new tilt and roll, the full rotation as a quaternion (x, y, z, w)
and as tf2 roll/pitch/yaw, and saves ~/floor_tilt.png showing the detected
points (green) and the fitted line (red).

Angle convention:
  R_base<-cam = Rz(yaw) . Ry(tilt) . R0 . Rz(roll)
  R0 = level camera looking along base x (optical z = base x, x = -base y, y = -base z)
  yaw > 0: camera points left; tilt > 0: camera points down; roll about the optical axis.
  Oct 24: yaw 2.48, tilt 13.14, roll 1.15 deg. After the Nov 10 re-tilt this gave tilt
  14.76 and roll 1.75 deg; yaw was then re-measured as 1.38 deg with a marker in front
  of the car squared to a straight edge. K and D below are copied from the calibration.

Usage: python3 floor_tilt.py [--dist 1.397] [--band 85 160] [--image file.png]
"""
import argparse
import os

import cv2
import numpy as np

# Intrinsics (Oct 23 calibration, 640x360)
K = np.array([[492.67, 0.0, 299.81], [0.0, 491.21, 182.03], [0.0, 0.0, 1.0]])
D = np.array([0.09484, -0.07446, 0.00338, -0.01645, 0.0])

# Kept from the Oct 24 transform
YAW_DEG = 2.48
T_CAM = np.array([0.0877, -0.0030, 0.1519])  # lens in base_link (m)
OCT24 = {"tilt": 13.14, "roll": 1.15}

R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_of(yaw, tilt, roll):
    return Rz(yaw) @ Ry(tilt) @ R0 @ Rz(roll)


def quat_xyzw(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(3)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rpy_fixed(R):
    """tf2 / static_transform_publisher convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def grab_raw(topic, timeout=5.0):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class G(Node):
        def __init__(self):
            super().__init__("floor_tilt")
            self.msg = None
            self.create_subscription(Image, topic, self.cb, qos_profile_sensor_data)

        def cb(self, m):
            if self.msg is None:
                self.msg = m

    rclpy.init()
    n = G()
    end = n.get_clock().now().nanoseconds + int(timeout * 1e9)
    while n.msg is None and n.get_clock().now().nanoseconds < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    m = n.msg
    n.destroy_node()
    rclpy.shutdown()
    if m is None:
        raise SystemExit(f"No image on {topic}. Is picar.launch.py running?")
    buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
    enc = m.encoding.lower()
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}[enc]
    img = buf[:, : m.width * ch].reshape(m.height, m.width, ch) if ch > 1 else buf[:, : m.width]
    conv = {"bgra8": cv2.COLOR_BGRA2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
            "rgb8": cv2.COLOR_RGB2BGR, "mono8": cv2.COLOR_GRAY2BGR}
    return cv2.cvtColor(img, conv[enc]) if enc in conv else img.copy()


def find_line(gray, band):
    """Per column: row of the strongest bright-to-dark step going down."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    dv = np.zeros_like(g)
    dv[2:-2] = g[4:] - g[:-4]  # positive = brighter below
    v0, v1 = band
    us, vs, strength = [], [], []
    for u in range(4, g.shape[1] - 4, 2):
        col = dv[v0:v1, u]
        i = int(np.argmin(col))  # strongest darkening going down
        if 0 < i < len(col) - 1:
            a, b, c = col[i - 1], col[i], col[i + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if den != 0 else 0.0
            us.append(u)
            vs.append(v0 + i + off)
            strength.append(-b)
    us, vs, strength = map(np.array, (us, vs, strength))
    keep = strength > 0.4 * np.median(strength)  # drop weak columns
    us, vs = us[keep], vs[keep]
    # robust straight-line fit in pixels to throw out stray columns
    inl = np.ones(len(us), bool)
    for _ in range(5):
        p = np.polyfit(us[inl], vs[inl], 1)
        r = np.abs(vs - np.polyval(p, us))
        thr = max(2.0, 3.0 * 1.4826 * np.median(r[inl]))
        inl = r < thr
    return us, vs, inl


def predicted_line(tilt, roll, dist):
    """Two normalized image points on the projected floor line."""
    R = R_of(np.radians(YAW_DEG), tilt, roll)
    pts = []
    for yb in (-1.0, 1.0):
        p_base = np.array([dist, yb, 0.0])
        pc = R.T @ (p_base - T_CAM)
        pts.append(pc[:2] / pc[2])
    return np.array(pts)


def residuals(params, xn, yn, dist):
    (x1, y1), (x2, y2) = predicted_line(params[0], params[1], dist)
    y_pred = y1 + (y2 - y1) * (xn - x1) / (x2 - x1)
    return yn - y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=1.397,
                    help="base_link to the floor line along the car's x (m); home X to cabinet fronts = 1.397")
    ap.add_argument("--band", type=int, nargs=2, default=[85, 160],
                    help="rows to search for the floor line")
    ap.add_argument("--image", help="use a saved raw frame instead of the camera topic")
    ap.add_argument("--topic", default="/camera/image_raw")
    args = ap.parse_args()

    img = cv2.imread(args.image) if args.image else grab_raw(args.topic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    us, vs, inl = find_line(gray, args.band)
    n_in = int(inl.sum())
    if n_in < 30:
        raise SystemExit(f"Only {n_in} floor-line points found. Check the view (see ~/floor_tilt.png).")

    und = cv2.undistortPoints(np.stack([us[inl], vs[inl]], 1).astype(np.float64).reshape(-1, 1, 2), K, D)
    xn, yn = und[:, 0, 0], und[:, 0, 1]

    p = np.radians([OCT24["tilt"], OCT24["roll"]])
    for _ in range(30):
        r = residuals(p, xn, yn, args.dist)
        J = np.stack([(residuals(p + d, xn, yn, args.dist) - r) / 1e-7 for d in np.eye(2) * 1e-7], 1)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p += step
        if np.abs(step).max() < 1e-10:
            break
    r = residuals(p, xn, yn, args.dist)
    rms_px = float(np.sqrt(np.mean(r ** 2)) * K[1, 1])
    tilt_d, roll_d = np.degrees(p)

    R = R_of(np.radians(YAW_DEG), p[0], p[1])
    q = quat_xyzw(R)
    rr, pp, yy = rpy_fixed(R)

    # debug image
    dbg = img.copy()
    for u, v, ok in zip(us, vs, inl):
        cv2.circle(dbg, (int(u), int(round(v))), 1, (0, 255, 0) if ok else (0, 140, 255), -1)
    (x1, y1), (x2, y2) = predicted_line(p[0], p[1], args.dist)
    xs = np.linspace(-0.75, 0.75, 200)
    ys = y1 + (y2 - y1) * (xs - x1) / (x2 - x1)
    pix, _ = cv2.projectPoints(np.stack([xs, ys, np.ones_like(xs)], 1), np.zeros(3), np.zeros(3), K, D)
    cv2.polylines(dbg, [pix.reshape(-1, 1, 2).astype(np.int32)], False, (0, 0, 255), 1)
    out = os.path.expanduser("~/floor_tilt.png")
    cv2.imwrite(out, dbg)

    v_mid = np.polyval(np.polyfit(us[inl], vs[inl], 1), 320)
    print(f"Floor line: {n_in} of {len(us)} columns used, row {v_mid:.1f} at the image centre, "
          f"fit RMS {rms_px:.2f} px")
    print(f"Tilt down: {tilt_d:6.2f} deg   (Oct 24: {OCT24['tilt']:.2f})")
    print(f"Roll:      {roll_d:6.2f} deg   (Oct 24: {OCT24['roll']:.2f})")
    print(f"Yaw left:  {YAW_DEG:6.2f} deg   (kept from Oct 24; check with markers)")
    print(f"Quaternion x y z w: {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}")
    print(f"tf2 roll pitch yaw (rad): {rr:.5f} {pp:.5f} {yy:.5f}")
    print(f"tf2 roll pitch yaw (deg): {np.degrees(rr):.2f} {np.degrees(pp):.2f} {np.degrees(yy):.2f}")
    print("Rotation matrix R_base<-cam:")
    print(np.array2string(R, precision=5, suppress_small=True))
    print(f"Saved {out} (green: detected floor line, orange: rejected, red: fitted line)")


if __name__ == "__main__":
    main()
