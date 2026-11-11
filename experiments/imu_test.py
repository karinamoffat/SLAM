#!/usr/bin/env python3
"""MPU-6050 gyro tests for the Pi car (no ROS).

Modes:
  still   Car (or board) sitting still: measures gyro bias and noise on all
          three axes, plus accel (to confirm z is vertical) and temperature.
  rotate  Measures bias for a few seconds, then integrates gyro z while you
          rotate the car by hand. Press Ctrl+C to stop and print the angle.

Examples:
  python3 imu_test.py --mode still --seconds 30
  python3 imu_test.py --mode rotate

Sign convention: chip side up, gyro z positive = counterclockwise seen from
above = a LEFT turn (REP-103). So a left 90 deg turn should read about +90.
"""

import argparse
import csv
import math
import os
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car"))
from mpu6050 import (SMBus, s16, BUS, ADDR, SMPLRT_DIV, CONFIG, GYRO_CONFIG,  # noqa: E402
                     ACCEL_CONFIG, ACCEL_XOUT_H, PWR_MGMT_1, WHO_AM_I)

GYRO_LSB_PER_DPS = 131.0     # +-250 deg/s range
ACCEL_LSB_PER_G = 16384.0    # +-2 g range
RATE_HZ = 100.0              # 1 kHz internal (DLPF on) / (1 + 9)

LOG_STILL = os.path.expanduser("~/imu_still.csv")


class MPU6050:
    def __init__(self, bus=BUS, addr=ADDR):
        self.bus = SMBus(bus)
        self.addr = addr
        who = self.bus.read_byte_data(addr, WHO_AM_I)
        print(f"WHO_AM_I = 0x{who:02x}", end="")
        if who == 0x68:
            print(" (genuine MPU-6050 ID)")
        else:
            print(" (not 0x68: likely a clone or compatible chip; continuing)")
        self.bus.write_byte_data(addr, PWR_MGMT_1, 0x01)   # wake, clock = gyro X PLL
        time.sleep(0.1)
        self.bus.write_byte_data(addr, CONFIG, 0x03)       # DLPF about 44 Hz
        self.bus.write_byte_data(addr, SMPLRT_DIV, 9)      # 100 Hz output
        self.bus.write_byte_data(addr, GYRO_CONFIG, 0x00)  # +-250 deg/s
        self.bus.write_byte_data(addr, ACCEL_CONFIG, 0x00) # +-2 g
        time.sleep(0.1)

    def read(self):
        """Returns accel (g, 3), gyro (deg/s, 3), temperature (C)."""
        d = self.bus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14)
        acc = np.array([s16(d[0], d[1]), s16(d[2], d[3]), s16(d[4], d[5])]) / ACCEL_LSB_PER_G
        temp = s16(d[6], d[7]) / 340.0 + 36.53
        gyr = np.array([s16(d[8], d[9]), s16(d[10], d[11]), s16(d[12], d[13])]) / GYRO_LSB_PER_DPS
        return acc, gyr, temp

    def sample(self, seconds, label=""):
        """Polls at about RATE_HZ for `seconds`. Returns arrays and times."""
        n = int(seconds * RATE_HZ)
        accs, gyrs, temps, ts = [], [], [], []
        period = 1.0 / RATE_HZ
        t0 = time.perf_counter()
        next_t = t0
        for i in range(n):
            a, g, tc = self.read()
            accs.append(a); gyrs.append(g); temps.append(tc)
            ts.append(time.perf_counter() - t0)
            if label and i % int(RATE_HZ) == 0:
                print(f"  {label} {i / RATE_HZ:4.0f} s", end="\r", flush=True)
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))
        if label:
            print()
        return np.array(accs), np.array(gyrs), np.array(temps), np.array(ts)


def mode_still(imu, seconds, label):
    print(f"Keep the car completely still for {seconds:.0f} s (don't touch the table).")
    acc, gyr, temp, ts = imu.sample(seconds, label="sampling")
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    bias = gyr.mean(axis=0)
    sigma = gyr.std(axis=0, ddof=1)
    a_mean = acc.mean(axis=0)

    # Integrated heading error after T seconds (z axis), for intuition:
    #   from white noise:  sigma_theta = sigma_w * sqrt(T / rate)   (grows as sqrt T)
    #   from a bias left uncorrected:  bias * T                     (grows as T)
    def rw(T):
        return sigma[2] * math.sqrt(T / rate)

    # How well the bias itself is known from this run (standard error)
    bias_se = sigma[2] / math.sqrt(len(ts))

    # Drift check: bias in the first vs last third of the run
    k = len(ts) // 3
    b_first, b_last = gyr[:k, 2].mean(), gyr[-k:, 2].mean()

    print()
    print(f"Samples: {len(ts)}  at {rate:.1f} Hz   temperature {temp.mean():.1f} C "
          f"(start {temp[:k].mean():.1f}, end {temp[-k:].mean():.1f})")
    print("Gyro (deg/s)        x         y         z")
    print(f"  bias         {bias[0]:+8.3f}  {bias[1]:+8.3f}  {bias[2]:+8.3f}")
    print(f"  sigma        {sigma[0]:8.3f}  {sigma[1]:8.3f}  {sigma[2]:8.3f}")
    print(f"  z bias std error {bias_se:.4f} deg/s;  first third {b_first:+.3f}, "
          f"last third {b_last:+.3f} (drift {b_last - b_first:+.3f})")
    print("Accel (g)           x         y         z")
    print(f"  mean         {a_mean[0]:+8.3f}  {a_mean[1]:+8.3f}  {a_mean[2]:+8.3f}"
          f"   |a| = {np.linalg.norm(a_mean):.3f}")
    tilt = math.degrees(math.acos(min(1.0, abs(a_mean[2]) / np.linalg.norm(a_mean))))
    print(f"  z axis tilt from vertical: {tilt:.1f} deg "
          f"({'chip up' if a_mean[2] > 0 else 'chip DOWN: z sign flips'})")
    print("Heading error if gyro z is integrated (z axis):")
    for T in (10, 60):
        print(f"  after {T:3d} s: noise only {rw(T):.2f} deg;  "
              f"bias uncorrected {abs(bias[2]) * T:.1f} deg")

    new = not os.path.exists(LOG_STILL)
    with open(LOG_STILL, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "label", "seconds", "rate_hz", "temp_c",
                        "bias_x", "bias_y", "bias_z", "sigma_x", "sigma_y", "sigma_z",
                        "bias_z_first", "bias_z_last", "acc_x", "acc_y", "acc_z"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), label, seconds,
                    f"{rate:.2f}", f"{temp.mean():.2f}",
                    *[f"{v:.4f}" for v in bias], *[f"{v:.4f}" for v in sigma],
                    f"{b_first:.4f}", f"{b_last:.4f}", *[f"{v:.4f}" for v in a_mean]])
    print(f"Logged to {LOG_STILL}")


def mode_rotate(imu, bias_seconds):
    print(f"Hold still: measuring bias for {bias_seconds:.0f} s ...")
    _, gyr, _, _ = imu.sample(bias_seconds, label="bias")
    bz = gyr[:, 2].mean()
    print(f"Bias z = {bz:+.3f} deg/s (subtracted).")
    print("Now rotate the car by hand. Press Ctrl+C when it has stopped.")
    angle = 0.0
    peak = 0.0
    t_prev = time.perf_counter()
    last_print = 0.0
    try:
        while True:
            _, g, _ = imu.read()
            t = time.perf_counter()
            dt = t - t_prev
            t_prev = t
            w = g[2] - bz
            angle += w * dt
            peak = max(peak, abs(w))
            if t - last_print > 0.2:
                print(f"  angle {angle:+8.2f} deg   rate {w:+8.1f} deg/s", end="\r", flush=True)
                last_print = t
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    print()
    print(f"Total rotation: {angle:+.2f} deg (positive = left / counterclockwise from above)")
    print(f"Peak rate seen: {peak:.0f} deg/s (range limit is 250)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["still", "rotate"], default="still")
    p.add_argument("--seconds", type=float, default=30.0, help="still: sampling time")
    p.add_argument("--bias-seconds", type=float, default=5.0, help="rotate: bias time")
    p.add_argument("--label", default="", help="still: label for the log")
    a = p.parse_args()
    imu = MPU6050()
    if a.mode == "still":
        mode_still(imu, a.seconds, a.label)
    else:
        mode_rotate(imu, a.bias_seconds)


if __name__ == "__main__":
    main()
