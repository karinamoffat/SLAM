#!/usr/bin/env python3
"""Make printable ArUco markers (DICT_4X4_50), one per page, as PNG files.

Usage:
    python3 make_markers.py                     # IDs 0 to 17, 15 cm, letter paper
    python3 make_markers.py --ids 0 1 2 --size-cm 15 --paper a4 --out markers

Each page has the marker (black square of --size-cm at 300 dpi), a white quiet zone
around it, and the ID printed underneath. Print at "actual size", then MEASURE the
printed black square: printers rescale pages (here 15 cm came out as 14.6 cm), and
the measured size is what goes into pipeline/marker_measure.py (MARKER_SIZES_M).
"""

import argparse
import os

import cv2
import numpy as np

DPI = 300
PAPER_IN = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", default=list(range(18)))
    ap.add_argument("--size-cm", type=float, default=15.0, help="black square side")
    ap.add_argument("--paper", choices=sorted(PAPER_IN), default="letter")
    ap.add_argument("--out", default="markers")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = int(round(a.size_cm / 2.54 * DPI))
    pw, ph = (int(v * DPI) for v in PAPER_IN[a.paper])
    if side > pw - 2 * DPI // 2:
        raise SystemExit("marker too large for the page with a 0.5 in quiet zone")
    for mid in a.ids:
        page = np.full((ph, pw), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(dictionary, mid, side)
        x0, y0 = (pw - side) // 2, DPI  # 1 in from the top
        page[y0:y0 + side, x0:x0 + side] = marker
        label = f"ID {mid}   DICT_4X4_50   black square {a.size_cm:g} cm (measure after printing)"
        cv2.putText(page, label, (x0, y0 + side + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3,
                    cv2.LINE_AA)
        path = os.path.join(a.out, f"marker_{mid:02d}.png")
        cv2.imwrite(path, page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Wrote {len(a.ids)} pages to {a.out}/ ({side} px = {a.size_cm:g} cm at {DPI} dpi)")


if __name__ == "__main__":
    main()
