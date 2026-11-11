#!/usr/bin/env python3
"""Capture checkerboard images for intrinsic calibration (roadmap step 3, C2).

Subscribes to /camera/image_raw, detects the 9x6 checkerboard, and saves a
frame automatically when the board has been held still and the view is
different enough from every view saved so far.

Also serves a live preview with detected corners drawn on it. Open
http://picar.local:8080 in a browser on the laptop.

Run on the Pi while the camera node is running:
  python3 calib_capture.py --check    # focus check only, saves nothing
  python3 calib_capture.py            # capture mode, saves to ~/calib_images
"""
import argparse
import glob
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

PATTERN = (9, 6)      # inner corners (columns, rows)
STILL_PX = 1.0        # mean corner motion between frames below this counts as still
STILL_S = 0.6         # board must stay still this long before a save
NOVELTY_PX = 30.0     # mean corner distance to every saved view must exceed this
STATUS_PERIOD_S = 0.5

# ROS encoding -> (channels, conversion to grayscale)
ENCODINGS = {
    "bgra8": (4, cv2.COLOR_BGRA2GRAY),
    "rgba8": (4, cv2.COLOR_RGBA2GRAY),
    "bgr8": (3, cv2.COLOR_BGR2GRAY),
    "rgb8": (3, cv2.COLOR_RGB2GRAY),
    "mono8": (1, None),
}

PAGE = b"""<!doctype html><html><head><title>Calibration preview</title></head>
<body style="margin:0;background:#111">
<img id="v" style="width:100%;max-width:1280px;display:block;margin:auto">
<script>
const v = document.getElementById('v');
function next() { v.src = '/preview.jpg?t=' + Date.now(); }
v.onload = () => setTimeout(next, 100);
v.onerror = () => setTimeout(next, 1000);
next();
</script></body></html>"""


class Preview:
    """Latest annotated frame as JPEG bytes, shared with the HTTP thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None

    def set(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()

    def get(self):
        with self.lock:
            return self.jpeg


def start_preview_server(preview, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/preview.jpg"):
                data = preview.get()
                if data is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                ctype = "image/jpeg"
            else:
                data, ctype = PAGE, "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # keep the terminal quiet

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class CalibCapture(Node):
    def __init__(self, out_dir, check_only, target, preview):
        super().__init__("calib_capture")
        self.out_dir = out_dir
        self.check_only = check_only
        self.target = target
        self.preview = preview
        self.saved = []           # (54, 2) corner arrays of saved views
        self.prev = None          # corners from the previous frame
        self.still_since = None
        self.last_status = 0.0
        self.coverage = np.zeros((3, 3), dtype=int)
        self.warned_encoding = False
        self.done = False
        self.create_subscription(
            Image, "/camera/image_raw", self.on_image, qos_profile_sensor_data)
        mode = "FOCUS CHECK (nothing is saved)" if check_only else f"CAPTURE to {out_dir}"
        print(f"Mode: {mode}. Waiting for /camera/image_raw ...", flush=True)

    def to_gray(self, msg):
        if msg.encoding not in ENCODINGS:
            if not self.warned_encoding:
                print(f"Unsupported encoding '{msg.encoding}'. Add it to ENCODINGS.", flush=True)
                self.warned_encoding = True
            return None
        ch, code = ENCODINGS[msg.encoding]
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = rows[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        if code is None:
            return np.ascontiguousarray(img[:, :, 0])
        return cv2.cvtColor(np.ascontiguousarray(img), code)

    def report(self, now, gray, found, corners, text):
        """Update the browser preview every frame; print to the terminal at most every 0.5 s."""
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            cv2.drawChessboardCorners(vis, PATTERN, corners, found)
        color = (0, 200, 0) if found else (0, 0, 255)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(vis, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        self.preview.set(vis)
        if now - self.last_status >= STATUS_PERIOD_S:
            print(text, flush=True)
            self.last_status = now

    def novelty(self, pts):
        """Smallest mean corner distance to any saved view (either corner ordering)."""
        if not self.saved:
            return np.inf
        best = np.inf
        for s in self.saved:
            d_same = np.mean(np.linalg.norm(pts - s, axis=1))
            d_flip = np.mean(np.linalg.norm(pts[::-1] - s, axis=1))
            best = min(best, d_same, d_flip)
        return best

    def on_image(self, msg):
        if self.done:
            return
        gray = self.to_gray(msg)
        if gray is None:
            return
        now = time.monotonic()

        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            self.prev = None
            self.still_since = None
            self.report(now, gray, False, None, "no board detected")
            return

        pts = corners.reshape(-1, 2).astype(np.float64)
        # Returns (values, per-edge data). values[0] is the mean width of the
        # black/white edges in pixels (10% to 90% rise). Lower is sharper.
        values, _ = cv2.estimateChessboardSharpness(gray, PATTERN, corners)
        sharp = float(values[0])
        board_w = float(np.ptp(pts[:, 0]))

        motion = np.inf if self.prev is None else float(
            np.mean(np.linalg.norm(pts - self.prev, axis=1)))
        self.prev = pts
        if motion < STILL_PX:
            if self.still_since is None:
                self.still_since = now
        else:
            self.still_since = None

        if self.check_only:
            self.report(now, gray, True, corners,
                        f"board found | edge width {sharp:.2f} px | board {board_w:.0f} px wide")
            return

        still = self.still_since is not None and now - self.still_since >= STILL_S
        nov = self.novelty(pts)
        if not still:
            self.report(now, gray, True, corners,
                        f"{len(self.saved)}/{self.target} | hold still ... (edge width {sharp:.2f} px)")
            return
        if nov < NOVELTY_PX:
            self.report(now, gray, True, corners,
                        f"{len(self.saved)}/{self.target} | too similar to a saved view, move or tilt")
            return

        idx = len(self.saved) + 1
        path = os.path.join(self.out_dir, f"calib_{idx:02d}.png")
        cv2.imwrite(path, gray)   # lossless; calibration re-detects corners from these
        self.saved.append(pts)
        col = np.clip((pts[:, 0] / msg.width * 3).astype(int), 0, 2)
        row = np.clip((pts[:, 1] / msg.height * 3).astype(int), 0, 2)
        np.add.at(self.coverage, (row, col), 1)
        self.still_since = None
        self.last_status = 0.0
        self.report(now, gray, True, corners, f"SAVED {idx}/{self.target}")

        print(f"\nSAVED {idx}/{self.target}: {path} | edge width {sharp:.2f} px | "
              f"board {board_w:.0f} px wide", flush=True)
        print("Corner coverage (3x3 grid of the frame, aim for no low cells):", flush=True)
        for r in self.coverage:
            print("   " + "  ".join(f"{c:4d}" for c in r), flush=True)
        print(flush=True)

        if idx >= self.target:
            print(f"Done: {idx} images saved in {self.out_dir}", flush=True)
            self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="focus check only, save nothing")
    parser.add_argument("--out", default=os.path.expanduser("~/calib_images"))
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--port", type=int, default=8080, help="browser preview port")
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    if not args.check:
        os.makedirs(args.out, exist_ok=True)
        if glob.glob(os.path.join(args.out, "calib_*.png")):
            print(f"{args.out} already has calib_*.png files. Move or delete them first.")
            return

    preview = Preview()
    server = start_preview_server(preview, args.port)
    print(f"Live preview: open http://picar.local:{args.port} on the laptop", flush=True)

    rclpy.init()
    node = CalibCapture(args.out, args.check, args.target, preview)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
