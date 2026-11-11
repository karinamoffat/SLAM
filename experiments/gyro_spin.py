#!/usr/bin/env python3
"""Turn tests measured by the MPU-6050 gyro (no ROS).

Each trial: the car sits still while the gyro bias is measured, then one
motor tap (spin or pivot) is applied, the car is left to coast to a stop,
and the integrated gyro z angle is the measured turn.

Uses car/picar_drive.py and car/mpu6050.py.
Do not run while motor_node.py is running (only one program can own the GPIO).

Styles:
  spin   left and right sides driven opposite (turn in place)
  pivot  inner side stopped, outer side forward

Examples:
  python3 gyro_spin.py --style spin --direction both --duties 1.0 --durations 0.15 0.25 --repeats 3
  python3 gyro_spin.py --style pivot --direction left --duties 0.7 0.8 0.9 1.0 --durations 0.4 --repeats 1
  python3 gyro_spin.py --style pivot --brake --direction both --duties 0.7 --durations 0.8 --repeats 3

Sign: positive angle = left (counterclockwise from above), REP-103.

Note: rows of logs/gyro_spins.csv logged before 21:18 on Nov 9, 2026 predate the
left/right channel fix in picar_drive.py, so their directions are mirrored.
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car"))
from mpu6050 import SMBus, s16, ADDR, BUS, WHO_AM_I, PWR_MGMT_1, CONFIG, SMPLRT_DIV, GYRO_CONFIG  # noqa: E402
from picar_drive import Drive  # noqa: E402

GYRO_ZOUT_H = 0x47
RANGES = {250: (0x00, 131.0), 500: (0x08, 65.5), 1000: (0x10, 32.8), 2000: (0x18, 16.4)}
LOG = os.path.expanduser("~/gyro_spins.csv")


class GyroZ:
    """Reads gyro z at about 200 Hz in a background thread and integrates it."""

    def __init__(self, gyro_range=500):
        self.bus = SMBus(BUS)
        who = self.bus.read_byte_data(ADDR, WHO_AM_I)
        print(f"WHO_AM_I = 0x{who:02x}")
        cfg, self.lsb = RANGES[gyro_range]
        self.range = gyro_range
        self.bus.write_byte_data(ADDR, PWR_MGMT_1, 0x01)
        time.sleep(0.1)
        self.bus.write_byte_data(ADDR, CONFIG, 0x02)       # DLPF about 94 Hz
        self.bus.write_byte_data(ADDR, SMPLRT_DIV, 4)      # 200 Hz output
        self.bus.write_byte_data(ADDR, GYRO_CONFIG, cfg)
        time.sleep(0.1)
        self.lock = threading.Lock()
        self.bias = 0.0
        self.angle = 0.0
        self.peak = 0.0
        self.clipped = False
        self.rec = False
        self.samples = []
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def raw_z(self):
        d = self.bus.read_i2c_block_data(ADDR, GYRO_ZOUT_H, 2)
        return s16(d[0], d[1])

    def _loop(self):
        t_prev = time.perf_counter()
        while self.running:
            r = self.raw_z()
            t = time.perf_counter()
            dt = t - t_prev
            t_prev = t
            w = r / self.lsb
            with self.lock:
                self.samples.append(w)
                if self.rec:
                    wc = w - self.bias
                    self.angle += wc * dt
                    self.peak = max(self.peak, abs(wc))
                    if abs(r) >= 32700:
                        self.clipped = True
            time.sleep(0.004)

    def measure_bias(self, seconds):
        with self.lock:
            self.samples = []
        time.sleep(seconds)
        with self.lock:
            s = np.array(self.samples)
        self.bias = float(s.mean())
        return self.bias, float(s.std(ddof=1)), len(s)

    def start(self):
        with self.lock:
            self.angle, self.peak, self.clipped, self.rec = 0.0, 0.0, False, True

    def stop(self):
        with self.lock:
            self.rec = False
            return self.angle, self.peak, self.clipped

    def close(self):
        self.running = False
        self.t.join(timeout=1)


def sides(style, direction, duty):
    """(left, right) duty for a turn. Left turn = right side forward."""
    if style == "spin":
        lr = (-duty, duty)
    else:  # pivot: inner side stopped
        lr = (0.0, duty)
    return lr if direction == "left" else (lr[1], lr[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["spin", "pivot"], default="spin")
    p.add_argument("--direction", choices=["left", "right", "both"], default="both")
    p.add_argument("--duties", type=float, nargs="+", default=[1.0])
    p.add_argument("--durations", type=float, nargs="+", default=[0.15, 0.25])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--range", type=int, choices=sorted(RANGES), default=500, help="gyro range, deg/s")
    p.add_argument("--bias-seconds", type=float, default=2.0, help="still time before each tap")
    p.add_argument("--coast", type=float, default=1.0, help="wait after the tap for the car to stop")
    p.add_argument("--brake", action="store_true", help="brake the stopped side (pivots)")
    p.add_argument("--label", default="")
    a = p.parse_args()

    dirs = ["left", "right"] if a.direction == "both" else [a.direction]
    trials = [(d, duty, dur) for duty in a.duties for dur in a.durations
              for _ in range(a.repeats) for d in dirs]

    gyro = GyroZ(a.range)
    results = []
    new = not os.path.exists(LOG)
    f = open(LOG, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["time", "label", "style", "direction", "duty", "duration_s",
                    "angle_deg", "peak_dps", "clipped", "bias_dps", "range_dps"])

    print(f"{len(trials)} trials. Before each: Enter = run, s = skip, q = quit.")
    print("Keep about 50 cm clear around the car. Don't touch it during a trial.")
    try:
        with Drive() as drive:
            for i, (d, duty, dur) in enumerate(trials, 1):
                cmd = input(f"\n[{i}/{len(trials)}] {a.style} {d} duty {duty} for {dur} s: ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "s":
                    continue
                time.sleep(0.5)  # let the car settle after you step back
                bias, bsd, n = gyro.measure_bias(a.bias_seconds)
                if bsd > 0.2:
                    print(f"  warning: car not still during bias (sigma {bsd:.2f} deg/s)")
                gyro.start()
                drive.set(*sides(a.style, d, duty), brake=a.brake)
                time.sleep(dur)
                drive.stop()
                time.sleep(a.coast)
                ang, peak, clip = gyro.stop()
                flag = "  CLIPPED: rerun with --range 1000" if clip else ""
                print(f"  turned {ang:+7.1f} deg   peak {peak:5.0f} deg/s   bias {bias:+.3f}{flag}")
                results.append((d, duty, dur, ang, clip))
                w.writerow([datetime.now().isoformat(timespec="seconds"), a.label, a.style, d,
                            duty, dur, f"{ang:.2f}", f"{peak:.1f}", int(clip), f"{bias:.4f}", a.range])
                f.flush()
    finally:
        f.close()
        gyro.close()

    if results:
        print("\nSummary (deg; positive = left)")
        print(f"{'dir':6} {'duty':>5} {'dur s':>6} {'n':>3} {'mean':>8} {'std':>6}  per tap")
        keys = sorted({(d, duty, dur) for d, duty, dur, _, _ in results})
        for k in keys:
            v = np.array([r[3] for r in results if r[:3] == k])
            sd = f"{v.std(ddof=1):6.1f}" if len(v) > 1 else "     -"
            print(f"{k[0]:6} {k[1]:5.2f} {k[2]:6.2f} {len(v):3d} {v.mean():+8.1f} {sd}  "
                  + " ".join(f"{x:+.0f}" for x in v))
        if any(r[4] for r in results):
            print("Some trials clipped: those angles are too small. Rerun with --range 1000.")
    print(f"Logged to {LOG}")


if __name__ == "__main__":
    main()
