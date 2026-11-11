#!/usr/bin/env python3
"""Live marker measurement test (the robot's camera must be running).

Subscribes to /camera/image_raw and /camera/camera_info (camera_ros) and runs
marker_measure.MarkerMeasurer on the frames: AprilTag corner refinement,
0.4 px outward edge correction, solvePnP (IPPE_SQUARE) on the raw image with
K and D, then the camera-to-car transform.

For each detected marker it prints, about twice a second:
  side_px    detected marker side in the image (detection margin)
  range      planar distance from the base_link origin to the marker centre,
             with sigma_r from the measurement model
  bearing    angle to the marker, positive = left (REP-103), with sigma_b
  z          marker centre height above the floor (sanity check only)
  flag       "ok" inside 0.6 to 2.5 m, otherwise "OUTSIDE MODEL"
and the detection time per frame.

Needs marker_measure.py in the same folder.

Run on the Pi (camera node running in another terminal):
  python3 aruco_test.py
  python3 aruco_test.py --expected 1.620     # also prints the error vs tape
"""

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

import numpy as np

from marker_measure import (DEFAULT_SIZE_M, EDGE_PX, RANGE_MAX_M, RANGE_MIN_M,
                            MarkerMeasurer, image_to_gray)


class ArucoTest(Node):
    def __init__(self, size_m, edge_px, expected):
        super().__init__("aruco_test")
        self.size_m = size_m
        self.edge_px = edge_px
        self.expected = expected
        self.measurer = None
        self.last_print = 0.0
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/image_raw", self.on_image,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f"default marker size {size_m * 100:.2f} cm, edge correction "
            f"{edge_px:.2f} px, valid range {RANGE_MIN_M}-{RANGE_MAX_M} m; "
            f"waiting for camera_info")

    def on_info(self, msg):
        if self.measurer is None:
            self.measurer = MarkerMeasurer(msg.k, msg.d, edge_px=self.edge_px,
                                           default_size_m=self.size_m)
            K = self.measurer.K
            self.get_logger().info(f"camera_info: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}")

    def on_image(self, msg):
        if self.measurer is None:
            return
        now = time.monotonic()
        if now - self.last_print < 0.5:
            return
        self.last_print = now
        res = self.measurer.measure(image_to_gray(msg))
        ms = self.measurer.last_ms
        if not res:
            print(f"  no marker  ({ms:.0f} ms)")
            return
        for r in res:
            z = r["p_base"][2]
            flag = "ok" if r["in_model"] else "OUTSIDE MODEL"
            line = (f"  ID {r['id']:2d}  side {r['side_px']:5.1f} px  "
                    f"range {r['range']:.3f} m +- {r['sigma_r'] * 100:.1f} cm  "
                    f"bearing {math.degrees(r['bearing']):+5.1f} "
                    f"+- {math.degrees(r['sigma_b']):.1f} deg  "
                    f"z {z:.3f}  {flag}  ({ms:.0f} ms)")
            if self.expected is not None:
                err = r["range"] - self.expected
                line += f"  err {err * 1000:+.0f} mm ({err / r['sigma_r']:+.1f} sigma)"
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-cm", type=float, default=DEFAULT_SIZE_M * 100,
                    help="black square side in cm for IDs not in MARKER_SIZES_M "
                         "(default 14.6)")
    ap.add_argument("--edge-px", type=float, default=EDGE_PX,
                    help="outward edge correction in px (default 0.4)")
    ap.add_argument("--expected", type=float, default=None,
                    help="tape range in m; prints the error and its size in sigma")
    args = ap.parse_args()
    rclpy.init()
    node = ArucoTest(args.size_cm / 100.0, args.edge_px, args.expected)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
