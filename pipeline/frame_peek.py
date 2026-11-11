#!/usr/bin/env python3
"""Save a few camera frames from a bag and run raw ArUco detection on them.

Usage (on the Pi):
    python3 frame_peek.py ~/bags/m7_run4 --frames 50 300 700

For each requested frame (index counted on /camera/image_10hz):
  - saves ~/peek_<index>.png (colour, as recorded)
  - prints image stats (to spot black or garbled frames)
  - runs plain cv2.aruco DICT_4X4_50 detection on the frame as is,
    mirrored left-right, and flipped upside down, and prints the IDs found
"""

import argparse
import os
import warnings

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

warnings.filterwarnings("ignore", category=DeprecationWarning)

IMG_TOPIC = "/camera/image_10hz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--frames", type=int, nargs="+", default=[50, 300, 700])
    args = ap.parse_args()

    bag = os.path.expanduser(args.bag).rstrip("/")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    cls = get_message(types[IMG_TOPIC])

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )

    def ids_of(gray):
        _, ids, rejected = detector.detectMarkers(gray)
        found = [] if ids is None else sorted(int(i) for i in ids.flatten())
        return found, len(rejected)

    wanted = set(args.frames)
    last = max(wanted)
    idx = -1
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != IMG_TOPIC:
            continue
        idx += 1
        if idx not in wanted:
            if idx > last:
                break
            continue
        msg = deserialize_message(data, cls)
        h, w, step = msg.height, msg.width, msg.step
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
        bgra = buf[:, : w * 4].reshape(h, w, 4)
        bgr = np.ascontiguousarray(bgra[:, :, :3])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        path = os.path.expanduser(f"~/peek_{idx}.png")
        cv2.imwrite(path, bgr)
        print(f"frame {idx}: {msg.encoding} {w}x{h}, step {step} (w*4 = {w * 4}),"
              f" data {len(msg.data)} bytes")
        print(f"  gray mean {gray.mean():.0f}, min {gray.min()}, max {gray.max()};"
              f" 4th channel min {bgra[:, :, 3].min()}, max {bgra[:, :, 3].max()}")
        for label, img in (
            ("as recorded   ", gray),
            ("mirrored L-R  ", cv2.flip(gray, 1)),
            ("upside down   ", cv2.flip(gray, 0)),
        ):
            found, n_rej = ids_of(img)
            print(f"  {label} IDs {found}  (rejected candidates: {n_rej})")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
