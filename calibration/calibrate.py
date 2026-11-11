#!/usr/bin/env python3
"""Intrinsic calibration from saved checkerboard images (roadmap step 3, C2).

Reads ~/calib_images/calib_*.png (saved by calib_capture.py), runs OpenCV's
calibrateCamera (Zhang's method), reports K, D, the uncertainty of each
parameter and the per-image reprojection error, and optionally writes the
camera_info YAML that camera_ros loads.

Run on the Pi:
  python3 calibrate.py            # review the results only
  python3 calibrate.py --write    # also write the camera_info YAML
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

PATTERN = (9, 6)              # inner corners (columns, rows)
DEFAULT_SQUARE_M = 0.02375    # measured: 4 squares = 95 mm
CAMINFO_NAME = ("imx708__base_axi_pcie_1000120000_rp1_i2c_88000_imx708_1a_"
                "640x360_2304x1296_BGGR_PISP_COMP1_RAW")
DEFAULT_YAML = os.path.expanduser(f"~/.ros/camera_info/{CAMINFO_NAME}.yaml")
EXPECTED_F = 490.0            # from the 66 degree horizontal FOV spec at 640 px


def fmt(arr):
    return "[" + ", ".join(f"{v:.8g}" for v in np.asarray(arr).ravel()) + "]"


def write_yaml(path, w, h, K, D, P):
    text = (
        f"image_width: {w}\n"
        f"image_height: {h}\n"
        f"camera_name: {CAMINFO_NAME}\n"
        "camera_matrix:\n  rows: 3\n  cols: 3\n"
        f"  data: {fmt(K)}\n"
        "distortion_model: plumb_bob\n"
        "distortion_coefficients:\n  rows: 1\n  cols: 5\n"
        f"  data: {fmt(D)}\n"
        "rectification_matrix:\n  rows: 3\n  cols: 3\n"
        "  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
        "projection_matrix:\n  rows: 3\n  cols: 4\n"
        f"  data: {fmt(P)}\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
        print(f"Existing file backed up to {path}.bak")
    with open(path, "w") as f:
        f.write(text)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.expanduser("~/calib_images"))
    ap.add_argument("--square", type=float, default=DEFAULT_SQUARE_M, help="square size in metres")
    ap.add_argument("--fix-k3", action="store_true", help="hold k3 at 0 (simpler distortion model)")
    ap.add_argument("--write", action="store_true", help="write the camera_info YAML")
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "calib_*.png")))
    if not paths:
        sys.exit(f"No calib_*.png images in {args.images}")

    # Board frame: origin at the first inner corner, Z = 0 for every corner (flat board).
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * args.square

    obj_pts, img_pts, names, size = [], [], [], None
    flags_det = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=flags_det)
        if not found:
            print(f"  skip {os.path.basename(p)}: board not re-detected")
            continue
        obj_pts.append(objp)
        img_pts.append(corners.astype(np.float32))
        names.append(os.path.basename(p))
    print(f"Using {len(names)} of {len(paths)} images at {size[0]}x{size[1]}")
    if len(names) < 10:
        sys.exit("Fewer than 10 usable images. Capture more before calibrating.")

    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    rms, K, D, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, size, None, None, flags=flags)
    D = D.ravel()[:5]
    std = std_int.ravel()

    print(f"\nRMS reprojection error: {rms:.3f} px  (rough guide: under 0.5 good, over 1 suspicious)")
    print("\nIntrinsics (value +- 1 standard deviation):")
    for name, val, s in zip(["fx", "fy", "cx", "cy"],
                            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], std[:4]):
        print(f"  {name:3s} = {val:8.2f} +- {s:.2f} px")
    print("Distortion [k1, k2, t1, t2, k3]:")
    for name, val, s in zip(["k1", "k2", "t1", "t2", "k3"], D, std[4:9]):
        print(f"  {name:3s} = {val:+.5f} +- {s:.5f}")

    print("\nPer-image RMS error (worst first):")
    errs = per_view.ravel()
    med = float(np.median(errs))
    for i in np.argsort(errs)[::-1]:
        flag = "   <-- outlier? (over 2x median)" if errs[i] > 2 * med else ""
        print(f"  {names[i]}  {errs[i]:.3f} px{flag}")

    print("\nSanity check against the spec:")
    ok = True
    for name, val in [("fx", K[0, 0]), ("fy", K[1, 1])]:
        good = abs(val - EXPECTED_F) / EXPECTED_F < 0.15
        ok &= good
        print(f"  {name} {val:.1f} vs about {EXPECTED_F:.0f}: {'ok' if good else 'CHECK'}")
    for name, val, ref in [("cx", K[0, 2], size[0] / 2), ("cy", K[1, 2], size[1] / 2)]:
        good = abs(val - ref) < 40
        ok &= good
        print(f"  {name} {val:.1f} vs about {ref:.0f}: {'ok' if good else 'CHECK'}")

    # P for the rectified image (alpha = 0 keeps only valid pixels, the ROS default).
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0)
    P = np.hstack([newK, np.zeros((3, 1))])

    if args.write:
        if not ok:
            print("\nSanity check failed; not writing. Review the numbers first.")
            return
        write_yaml(args.yaml, size[0], size[1], K, D, P)
    else:
        print("\nReview only. Rerun with --write to save the camera_info YAML.")


if __name__ == "__main__":
    main()
