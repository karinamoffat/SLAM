#!/usr/bin/env python3
"""Render the onboard camera panel for the demo video.

Usage (the robot does not need to be running):
    python3 render_camera.py ~/bags/m8_run4 ~/replay/m8_run4_slam.npz

Reads the camera frames from the bag, draws every detected ArUco marker (box, ID and
measured range), and writes ~/replay/<run>_camera_video.mp4, 960 x 1080. It uses the
same time grid, frame rate and final hold as render_slam.py (read from the SLAM file),
so the two videos line up frame for frame: put them side by side with no syncing.
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

W, H = 960, 1080
CAM_W, CAM_H = 960, 540
CAM_Y = 250
FONT = cv2.FONT_HERSHEY_DUPLEX
GREEN = (60, 180, 60)      # BGR
GREY = (150, 150, 150)
DARK = (40, 40, 40)
MAX_ID = 12                # ekf_slam.py maps IDs 0 to 12 only


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def to_bgr(msg):
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf[:, :w].copy(), cv2.COLOR_GRAY2BGR)
    if enc in ("bgr8", "rgb8", "8uc3"):
        img = buf[:, : w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
    if enc in ("bgra8", "rgba8", "xrgb8888", "bgrx8", "8uc4"):
        img = buf[:, : w * 4].reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(d, p)


def put(img, text, org, scale, color=DARK, thick=1, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
        org = (int(org[0] - tw / 2), org[1])
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def detect(det, measurer, bgr):
    """List of (id, corners 4x2 in the camera image, range or None)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    ranges = {}
    if measurer is not None:
        for m in measurer.measure(gray):
            g = (lambda k: m[k]) if isinstance(m, dict) else (lambda k: getattr(m, k))
            if g("in_model"):
                ranges[int(g("id"))] = float(g("range"))
    out = []
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            out.append((int(i), c.reshape(4, 2), ranges.get(int(i))))
    return out


def compose(bgr, dets, t_rel):
    """One 960 x 1080 panel: header, the camera image with detections, a caption."""
    panel = np.full((H, W, 3), 255, np.uint8)
    put(panel, "What the car sees", (W // 2, 60), 1.3, DARK, 2, center=True)
    put(panel, "Raspberry Pi Camera 3, 10 Hz, ArUco markers measured as range + bearing",
        (W // 2, 105), 0.6, (90, 90, 90), 1, center=True)
    sx, sy = CAM_W / bgr.shape[1], CAM_H / bgr.shape[0]
    cam = cv2.resize(bgr, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    labels = []
    for mid, c, r in dets:
        pts = (c * [sx, sy]).astype(np.int32)
        used = mid <= MAX_ID
        col = GREEN if used else GREY
        cv2.polylines(cam, [pts], True, col, 3, cv2.LINE_AA)
        txt = f"ID {mid}" + (f"  {r:.2f} m" if (r is not None and used) else "")
        x, y = pts[:, 0].min(), pts[:, 1].min() - 10
        y = max(y, 28)
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.8, 2)
        cv2.rectangle(cam, (x - 4, y - th - 6), (x + tw + 4, y + 6), col, -1)
        put(cam, txt, (x, y), 0.8, (255, 255, 255), 2)
        if used:
            labels.append(txt)
    panel[CAM_Y:CAM_Y + CAM_H] = cam
    y0 = CAM_Y + CAM_H + 70
    if labels:
        put(panel, "In view: " + ",   ".join(labels[:4]), (W // 2, y0), 0.8, GREEN, 2, center=True)
    else:
        put(panel, "No marker in view: dead reckoning (gyro + commands), uncertainty grows",
            (W // 2, y0), 0.65, (60, 60, 200), 1, center=True)
    put(panel, f"t = {t_rel:5.1f} s", (40, H - 50), 0.9, DARK, 1)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag directory, e.g. ~/bags/m8_run4")
    ap.add_argument("slam_npz", help="<run>_slam.npz (for the time grid)")
    ap.add_argument("--fps", type=float, default=30.0, help="must match render_slam.py")
    ap.add_argument("--speed", type=float, default=1.0, help="must match render_slam.py")
    ap.add_argument("--hold", type=float, default=4.0, help="final hold, s (render_slam HOLD_S)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    try:
        from marker_measure import MarkerMeasurer
    except Exception as e:  # ranges are a nice-to-have
        print(f"marker_measure not importable ({e}): drawing IDs without ranges")
        MarkerMeasurer = None

    bag = os.path.expanduser(args.bag).rstrip("/")
    name = os.path.basename(bag)
    s = np.load(os.path.expanduser(args.slam_npz))
    t_imu = s["t"]
    times = np.arange(t_imu[0], t_imu[-1], args.speed / args.fps)
    n_hold = int(args.hold * args.fps)
    out = args.out or os.path.expanduser(f"~/replay/{name}_camera_video.mp4")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    try:
        reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/image_10hz",
                                                           "/camera/camera_info"]))
    except Exception:
        pass
    img_cls = get_message(types["/camera/image_10hz"])
    info_cls = get_message(types["/camera/camera_info"])

    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE)

    det = make_detector()
    measurer = None
    last_panel = compose(np.zeros((360, 640, 3), np.uint8), [], 0.0)
    k = 0  # next output frame index
    n_img = 0
    tic = time.perf_counter()
    print(f"Writing {len(times)} frames + {n_hold} held at {args.fps:g} fps to {out}")
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/camera/camera_info":
            if measurer is None and MarkerMeasurer is not None:
                msg = deserialize_message(data, info_cls)
                measurer = MarkerMeasurer(np.array(msg.k, float).reshape(3, 3),
                                          np.array(msg.d, float))
            continue
        if topic != "/camera/image_10hz":
            continue
        msg = deserialize_message(data, img_cls)
        t_img = stamp_s(msg.header.stamp)
        # output frames before this image show the previous one
        while k < len(times) and times[k] < t_img:
            ff.stdin.write(last_panel.tobytes())
            k += 1
        if k >= len(times):
            break
        bgr = to_bgr(msg)
        last_panel = compose(bgr, detect(det, measurer, bgr), t_img - t_imu[0])
        n_img += 1
        if n_img % 300 == 0:
            print(f"  {n_img} camera frames, {k}/{len(times)} output frames,"
                  f" {time.perf_counter() - tic:.0f} s")
    while k < len(times):
        ff.stdin.write(last_panel.tobytes())
        k += 1
    for _ in range(n_hold):
        ff.stdin.write(last_panel.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"Done in {time.perf_counter() - tic:.0f} s: {n_img} camera frames used. Saved {out}")


if __name__ == "__main__":
    main()
