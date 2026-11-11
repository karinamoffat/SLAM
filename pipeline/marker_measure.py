#!/usr/bin/env python3
"""Marker measurement model for the Pi car.

Turns one grayscale camera frame into planar marker measurements in base_link:

    z = (range, bearing)        range in m, bearing in rad (positive = left, REP-103)
    R = diag(sigma_r^2, sigma_b^2)

Settings:
    DICT_4X4_50, CORNER_REFINE_APRILTAG
    each detected edge pushed EDGE_PX outward along its normal before solvePnP
    SOLVEPNP_IPPE_SQUARE on the raw image, so K and D (not P)

No ROS imports: the live test, the EKF nodes and offline bag replay can all
import this file and get exactly the same measurements.
"""
import math
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Camera pose in base_link: translation measured Oct 24, 2026 (checkerboard and tape);
# rotation re-measured Nov 10, 2026 after the mount was re-tilted.
T_BASE_CAM = np.array([0.0877, -0.0030, 0.1519])
Q_BASE_CAM = (0.55827, -0.56189, 0.41968, -0.44326)  # x, y, z, w (Nov 10: tilt 14.76, roll 1.75 from floor_tilt.py, yaw 1.38 from the marker 4 straight-edge test)

# Corner bias correction: AprilTag places each edge slightly inside the true
# edge. Uncertain between 0.25 and 0.55 px; 0.4 px used (Oct 25, 2026).
EDGE_PX = 0.4

# Measured side of the black square, per marker ID, in metres.
# Printers rescale pages: the 15 cm squares printed at 14.6 cm, and the measured size
# is what goes into solvePnP. IDs not listed use DEFAULT_SIZE_M.
DEFAULT_SIZE_M = 0.146
MARKER_SIZES_M = {
    **{i: 0.146 for i in range(18)},  # IDs 0 to 17, all measured 14.6 cm (Oct 25, 2026)
}

# Measurement model (Oct 25, 2026): built from errors against tape ground
# truth plus the unresolved range bias, not from frame-to-frame jitter.
SIGMA_R0 = 0.005           # m, constant term (tape-level errors)
SIGMA_R2 = 0.0035          # m per m^2, constant-pixel-error term
SIGMA_B = math.radians(1.0)
RANGE_MIN_M = 0.6          # closer: marker top leaves the frame
RANGE_MAX_M = 2.5          # farther: untested, treat as outside the model


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


R_BASE_CAM = quat_to_rot(Q_BASE_CAM)


def marker_object_points(size_m):
    """Corner order expected by SOLVEPNP_IPPE_SQUARE (matches ArUco: TL, TR, BR, BL)."""
    h = size_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


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


def measurement_cov(rng):
    """Return (sigma_r [m], sigma_b [rad], R 2x2) for a sighting at range rng."""
    sigma_r = SIGMA_R0 + SIGMA_R2 * rng * rng
    R = np.diag([sigma_r * sigma_r, SIGMA_B * SIGMA_B])
    return sigma_r, SIGMA_B, R


def image_to_gray(msg):
    """Convert a sensor_msgs/Image to grayscale without cv_bridge.

    Only reads msg fields, so it needs no ROS import.
    """
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


# ---------------------------------------------------------------------------
# The measurement function
# ---------------------------------------------------------------------------

class MarkerMeasurer:
    """Detect markers in a frame and return range, bearing and R for each."""

    def __init__(self, K, D, edge_px=EDGE_PX, default_size_m=DEFAULT_SIZE_M,
                 sizes_m=None):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64)
        self.edge_px = edge_px
        self.default_size_m = default_size_m
        self.sizes_m = dict(MARKER_SIZES_M if sizes_m is None else sizes_m)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
        self._obj = {}
        self.last_ms = 0.0  # time of the last measure() call, for budget checks

    def size_for(self, marker_id):
        return self.sizes_m.get(marker_id, self.default_size_m)

    def _object_points(self, marker_id):
        if marker_id not in self._obj:
            self._obj[marker_id] = marker_object_points(self.size_for(marker_id))
        return self._obj[marker_id]

    def measure(self, gray):
        """Return a list of dicts, one per detected marker.

        Keys: id, range (m), bearing (rad), R (2x2), sigma_r (m), sigma_b (rad),
        in_model (bool, range inside RANGE_MIN_M..RANGE_MAX_M),
        side_px (raw detected side, the detection margin), p_base (x, y, z in m).
        """
        t0 = time.perf_counter()
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is not None:
            for c, mid in zip(corners, ids.ravel()):
                mid = int(mid)
                raw = c.reshape(4, 2).astype(np.float64)
                side_px = float(np.mean(np.linalg.norm(raw - np.roll(raw, -1, axis=0),
                                                       axis=1)))
                pts = expand_quad(raw, self.edge_px) if self.edge_px else raw
                ok, _rvec, tvec = cv2.solvePnP(self._object_points(mid), pts,
                                               self.K, self.D,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                p_base = R_BASE_CAM @ tvec.ravel() + T_BASE_CAM
                rng = float(math.hypot(p_base[0], p_base[1]))
                brg = float(math.atan2(p_base[1], p_base[0]))
                sigma_r, sigma_b, R = measurement_cov(rng)
                results.append({
                    "id": mid,
                    "range": rng,
                    "bearing": brg,
                    "R": R,
                    "sigma_r": sigma_r,
                    "sigma_b": sigma_b,
                    "in_model": RANGE_MIN_M <= rng <= RANGE_MAX_M,
                    "side_px": side_px,
                    "p_base": p_base,
                })
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return results
