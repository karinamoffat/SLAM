#!/usr/bin/env python3
"""Extract a recorded drive (ros2 bag) into a compact .npz for offline filtering.

Usage (on the Pi, home directory):
    python3 replay_extract.py ~/bags/m7_run4 --max-frames 100     # quick test
    python3 replay_extract.py ~/bags/m7_run4                      # full run

Reads the bag once, runs marker_measure.py on every camera frame, and saves
~/replay/<bag name>.npz with:
    imu_t, imu_t_rx, imu_wz      gyro z (rad/s, RAW: bias not subtracted)
    temp_t, temp_c               IMU temperature
    cmd_t, cmd_v, cmd_w          /cmd_applied (nominal applied motion)
    frame_t, frame_t_rx, frame_n one row per camera frame (n = markers seen)
    obs_t, obs_t_rx, obs_id, obs_range, obs_bearing, obs_sigma_r,
    obs_sigma_b, obs_in_model, obs_side_px
                                 one row per marker sighting (base_link frame,
                                 bearing positive = left)
    K, D                         camera intrinsics from /camera/camera_info
Times are in seconds. *_t are header stamps, *_t_rx are bag receive times.

Also prints a clock check (header stamp minus receive time per topic) and a
coverage summary (sightings per ID, longest stretches with no marker).
"""

import argparse
import os
import sys
import time
import warnings

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

warnings.filterwarnings("ignore", category=DeprecationWarning)

import marker_measure  # noqa: E402
from marker_measure import MarkerMeasurer  # noqa: E402

IMU_TOPIC = "/imu/data"
TEMP_TOPIC = "/imu/temperature"
CMD_TOPIC = "/cmd_applied"
IMG_TOPIC = "/camera/image_10hz"
INFO_TOPIC = "/camera/camera_info"


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def open_reader(path):
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=path, storage_id="mcap")
    conv = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader.open(storage, conv)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


def to_gray(msg):
    """sensor_msgs/Image to a grayscale uint8 array, without cv_bridge."""
    h, w, step = msg.height, msg.width, msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        return buf[:, :w].copy()
    if enc in ("bgr8", "rgb8", "8uc3"):
        img = buf[:, : w * 3].reshape(h, w, 3)
        code = cv2.COLOR_RGB2GRAY if enc == "rgb8" else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)
    if enc in ("bgra8", "rgba8", "xrgb8888", "bgrx8", "8uc4"):
        img = buf[:, : w * 4].reshape(h, w, 4)
        code = cv2.COLOR_RGBA2GRAY if enc == "rgba8" else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(img, code)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def field(m, name):
    """marker_measure results may be dicts or objects."""
    return m[name] if isinstance(m, dict) else getattr(m, name)


def offset_report(name, hdr, rx):
    if len(hdr) == 0:
        print(f"  {name:22s} no messages")
        return
    off = (np.asarray(hdr) - np.asarray(rx)) * 1000.0
    print(
        f"  {name:22s} header - receive: median {np.median(off):10.1f} ms,"
        f" min {off.min():10.1f}, max {off.max():10.1f}"
    )


def zero_runs(frame_t, frame_n, min_len_s):
    """Stretches of consecutive frames with no marker, longer than min_len_s."""
    runs = []
    start = None
    for t, n in zip(frame_t, frame_n):
        if n == 0 and start is None:
            start = t
        elif n > 0 and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, frame_t[-1]))
    return [(a, b) for a, b in runs if b - a >= min_len_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag directory, e.g. ~/bags/m7_run4")
    ap.add_argument("--out", default=None, help="output .npz path")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (test)")
    args = ap.parse_args()

    bag = os.path.expanduser(args.bag).rstrip("/")
    name = os.path.basename(bag)
    out = args.out or os.path.expanduser(f"~/replay/{name}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"marker_measure: EDGE_PX = {getattr(marker_measure, 'EDGE_PX', '?')}")
    sizes = getattr(marker_measure, "MARKER_SIZES_M", None)
    print(f"marker_measure: MARKER_SIZES_M has {len(sizes) if sizes else 0} IDs")
    if not sizes:
        print("ERROR: MARKER_SIZES_M is empty on this copy of marker_measure.py")
        sys.exit(1)

    reader, types = open_reader(bag)
    msg_cls = {topic: get_message(typ) for topic, typ in types.items()}
    print(f"/cmd_applied type: {types.get(CMD_TOPIC)}")

    imu_t, imu_rx, imu_wz = [], [], []
    temp_t, temp_rx, temp_c = [], [], []
    cmd_t, cmd_rx, cmd_v, cmd_w = [], [], [], []
    frame_t, frame_rx, frame_n = [], [], []
    obs = []
    K = D = None
    measurer = None
    encoding = None
    frame_ms = []
    t_wall0 = time.perf_counter()

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t_rx = t_ns * 1e-9
        if topic not in (IMU_TOPIC, TEMP_TOPIC, CMD_TOPIC, IMG_TOPIC, INFO_TOPIC):
            continue
        msg = deserialize_message(data, msg_cls[topic])

        if topic == IMU_TOPIC:
            imu_t.append(stamp_s(msg.header.stamp))
            imu_rx.append(t_rx)
            imu_wz.append(msg.angular_velocity.z)

        elif topic == TEMP_TOPIC:
            temp_t.append(stamp_s(msg.header.stamp))
            temp_rx.append(t_rx)
            temp_c.append(msg.temperature)

        elif topic == CMD_TOPIC:
            if hasattr(msg, "twist"):  # TwistStamped
                cmd_t.append(stamp_s(msg.header.stamp))
                tw = msg.twist
            else:  # plain Twist: no header, use receive time
                cmd_t.append(t_rx)
                tw = msg
            cmd_rx.append(t_rx)
            cmd_v.append(tw.linear.x)
            cmd_w.append(tw.angular.z)

        elif topic == INFO_TOPIC:
            if K is None:
                K = np.array(msg.k, dtype=float).reshape(3, 3)
                D = np.array(msg.d, dtype=float)
                measurer = MarkerMeasurer(K, D)

        elif topic == IMG_TOPIC:
            if measurer is None:
                continue  # no intrinsics yet; camera_info arrives within the first frames
            if encoding is None:
                encoding = msg.encoding
                print(f"image encoding: {encoding}, {msg.width}x{msg.height}")
            t_img = stamp_s(msg.header.stamp)
            gray = to_gray(msg)
            t0 = time.perf_counter()
            results = measurer.measure(gray)
            frame_ms.append((time.perf_counter() - t0) * 1000.0)
            frame_t.append(t_img)
            frame_rx.append(t_rx)
            frame_n.append(len(results))
            for m in results:
                obs.append((
                    t_img, t_rx, int(field(m, "id")),
                    float(field(m, "range")), float(field(m, "bearing")),
                    float(field(m, "sigma_r")), float(field(m, "sigma_b")),
                    bool(field(m, "in_model")), float(field(m, "side_px")),
                ))
            if len(frame_t) % 200 == 0:
                print(f"  {len(frame_t)} frames, {len(obs)} sightings ...")
            if args.max_frames and len(frame_t) >= args.max_frames:
                print(f"stopping after {args.max_frames} frames (--max-frames)")
                break

    elapsed = time.perf_counter() - t_wall0
    if K is None:
        print("ERROR: no /camera/camera_info in the bag")
        sys.exit(1)

    obs = np.array(obs, dtype=float).reshape(-1, 9)
    imu_t = np.array(imu_t)
    imu_wz = np.array(imu_wz)

    print("\nClock check (a large or drifting offset means a different clock):")
    offset_report(IMU_TOPIC, imu_t, imu_rx)
    offset_report(IMG_TOPIC, frame_t, frame_rx)
    offset_report(TEMP_TOPIC, temp_t, temp_rx)
    if types.get(CMD_TOPIC, "").endswith("TwistStamped"):
        offset_report(CMD_TOPIC, cmd_t, cmd_rx)
    else:
        print(f"  {CMD_TOPIC:22s} no header, receive time used")

    if len(imu_t):
        first = imu_t < imu_t[0] + 5.0
        print(
            f"\nGyro z mean over the first 5 s: {np.degrees(imu_wz[first].mean()):+.3f} deg/s"
            " (should be about +1.4, the still bias)"
        )

    n_frames = len(frame_t)
    n_seen = int(np.sum(np.array(frame_n) > 0))
    print(f"\nFrames: {n_frames}, with at least one marker: {n_seen}"
          f" ({100.0 * n_seen / max(n_frames, 1):.0f}%)")
    print(f"Sightings: {len(obs)}, out of model range (>2.5 m): "
          f"{int(np.sum(obs[:, 7] == 0)) if len(obs) else 0}")
    if frame_ms:
        print(f"marker_measure: {np.mean(frame_ms):.1f} ms per frame (max {np.max(frame_ms):.1f});"
              f" total {elapsed:.0f} s")

    if len(obs):
        print("\nSightings per ID:")
        ids, counts = np.unique(obs[:, 2].astype(int), return_counts=True)
        for i, c in zip(ids, counts):
            sel = obs[:, 2] == i
            print(f"  ID {i:2d}: {c:4d} sightings, range {obs[sel, 3].min():.2f} to"
                  f" {obs[sel, 3].max():.2f} m")

    if n_frames:
        t0 = frame_t[0]
        gaps = zero_runs(frame_t, frame_n, 3.0)
        print("\nStretches with no marker for 3 s or more (seconds from first frame):")
        if not gaps:
            print("  none")
        for a, b in gaps:
            print(f"  {a - t0:6.1f} to {b - t0:6.1f} s  ({b - a:4.1f} s)")

    np.savez(
        out,
        imu_t=imu_t, imu_t_rx=np.array(imu_rx), imu_wz=imu_wz,
        temp_t=np.array(temp_t), temp_c=np.array(temp_c),
        cmd_t=np.array(cmd_t), cmd_t_rx=np.array(cmd_rx),
        cmd_v=np.array(cmd_v), cmd_w=np.array(cmd_w),
        frame_t=np.array(frame_t), frame_t_rx=np.array(frame_rx), frame_n=np.array(frame_n),
        obs_t=obs[:, 0], obs_t_rx=obs[:, 1], obs_id=obs[:, 2].astype(int),
        obs_range=obs[:, 3], obs_bearing=obs[:, 4],
        obs_sigma_r=obs[:, 5], obs_sigma_b=obs[:, 6],
        obs_in_model=obs[:, 7].astype(bool), obs_side_px=obs[:, 8],
        K=K, D=D,
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
