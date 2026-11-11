#!/usr/bin/env python3
"""Check a recorded drive: topic rates, gyro bias, and heading per command.

Run on the Pi:
    python3 bag_check.py ~/bags/test1

What it does:
1. Reads the bag message by message with rosbag2_py (no live ROS needed).
2. Estimates the gyro z bias from the still period at the start: everything
   before the first nonzero /cmd_applied, minus a 0.5 s safety margin.
3. Integrates (gyro z - bias) over time to get heading, and splits the drive
   into segments by /cmd_applied mode (stop, straight, pivot left/right),
   printing how much the heading changed in each segment.
"""
import math
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag(path, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def mode_of(v, w):
    if abs(w) > 1e-6:
        return "pivot left" if w > 0 else "pivot right"
    if abs(v) > 1e-6:
        return "forward" if v > 0 else "back"
    return "stop"


def main():
    path = sys.argv[1]
    d = read_bag(path, ["/imu/data", "/cmd_applied", "/camera/image_10hz", "/imu/temperature"])
    imu = d["/imu/data"]
    t0 = imu[0][0]

    # Rates
    for topic, msgs in d.items():
        if len(msgs) > 1:
            span = msgs[-1][0] - msgs[0][0]
            print(f"{topic:22s} {len(msgs):6d} msgs  {(len(msgs) - 1) / span:6.1f} Hz")

    # Gyro z and stamps (header stamps: time of measurement)
    t = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for _, m in imu])
    wz = np.array([m.angular_velocity.z for _, m in imu])
    dts = np.diff(t)
    print(f"IMU sample gaps: median {np.median(dts) * 1000:.1f} ms, max {dts.max() * 1000:.1f} ms")

    cmds = [(tt, mode_of(m.linear.x, m.angular.z)) for tt, m in d["/cmd_applied"]]
    first_move = next((tt for tt, mode in cmds if mode != "stop"), None)
    if first_move is None:
        print("No motion commands in this bag.")
        return
    still_end = first_move - 0.5
    still = t < still_end
    bias = wz[still].mean()
    print(f"Still period: {still_end - t[0]:.1f} s, {still.sum()} samples")
    print(f"Gyro z bias {math.degrees(bias):+.3f} deg/s, noise sigma "
          f"{math.degrees(wz[still].std(ddof=1)):.3f} deg/s")
    temps = [m.temperature for _, m in d["/imu/temperature"]]
    if temps:
        print(f"IMU temperature {temps[0]:.1f} to {temps[-1]:.1f} C")

    # Heading by trapezoid integration of bias-corrected rate
    w = wz - bias
    heading = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * dts)])

    def heading_at(tt):
        return float(np.interp(tt, t, heading))

    # Segments: a new segment starts when the mode changes
    segs = []
    for tt, mode in cmds:
        if not segs or segs[-1][1] != mode:
            segs.append([tt, mode])
    print("\nSegments (heading change in deg, positive = left):")
    for i, (start, mode) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else t[-1]
        dh = math.degrees(heading_at(end) - heading_at(start))
        print(f"  {start - t0:6.1f} to {end - t0:6.1f} s  {mode:12s} {dh:+7.1f}")
    print(f"\nTotal heading change: {math.degrees(heading[-1]):+.1f} deg")


if __name__ == "__main__":
    main()
