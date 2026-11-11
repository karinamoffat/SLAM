#!/usr/bin/env python3
"""Straight-run test: duty to speed on the floor (no ROS).

Each trial drives both sides at the same duty for a fixed time, then asks for
three tape measurements (inches, any of: 47.5 / 47 1/2 / 3' 11 1/2):
  1. left rear tire: start edge to the back of the tire
  2. right rear tire: start edge to the back of the tire
  3. cabinet fronts to the right rear tire's outer face
From these it logs forward distance, speed, heading change and sideways drift.

Start pose for every trial: car north of the start edge, facing north (away
from the edge), backs of both rear tires touching the edge, right rear tire's
outer face on the tape mark (the --start-side-in distance from the cabinet
fronts). Cabinets on the car's right.

Runway guard: after each run the script learns speed per unit duty and, if a
later trial is predicted to travel more than --max-travel-m, shortens it and
says so. The logged duration_s is the one actually commanded.

Trim: --trim-right scales the right side's duty (right = duty x trim) to
cancel a steady pull; the trim is appended to the label in the log.
(The trimR rows in logs/straight_runs.csv were logged before the Nov 9 channel
fix, so they actually slowed the physical left side.)
--repeats N runs the whole trial list N times (for noise statistics).

Two durations per duty separate speed (slope of distance vs time) from the
startup loss (intercept). The reference duty runs first and last to expose
battery drain.

Usage (Pi, home directory, picar_drive.py beside it):
    python3 straight_run.py --start-side-in "2' 0"
Ctrl+C at any time stops the motors.
"""
import argparse
import csv
import math
import os
import time
from datetime import datetime

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car"))
from picar_drive import Drive  # noqa: E402

IN_TO_M = 0.0254
TIRE_SPACING_M = 0.140  # tire centrelines, 14.0 cm apart (rear assumed same as front)

DEFAULT_DUTIES = [0.35, 0.4, 0.5, 0.65, 0.8, 1.0]
DEFAULT_DURATIONS = [0.75, 1.5]
DEFAULT_MAX_TRAVEL_M = 1.5   # runway available in the test room, minus the car's length and a margin
SAFETY = 1.3                 # margin on the predicted distance (average speed underestimates top speed)
DEFAULT_REF = 0.65


def parse_inches(text):
    """'47.5', '47 1/2', "3' 11 1/2", '-1 1/4' -> inches as a float."""
    t = text.strip().replace('"', "")
    if not t:
        raise ValueError("empty")
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-").strip()
    feet = 0.0
    if "'" in t:
        f, t = t.split("'", 1)
        feet = float(f) if f.strip() else 0.0
        t = t.strip()
    inches = 0.0
    for tok in t.split():
        if "/" in tok:
            n, d = tok.split("/")
            inches += float(n) / float(d)
        else:
            inches += float(tok)
    return sign * (feet * 12.0 + inches)


def ask_inches(prompt):
    """Ask until a valid length is typed. Returns inches, or None for skip."""
    while True:
        s = input(prompt)
        if s.strip().lower() in ("s", "skip"):
            return None
        try:
            return parse_inches(s)
        except (ValueError, ZeroDivisionError):
            print("  Could not read that. Examples: 47.5   47 1/2   3' 11 1/2   (s to skip)")


def build_trials(duties, durations, ref):
    """Reference duty first and last, everything else in between."""
    trials = [(ref, d) for d in durations]
    for duty in duties:
        if abs(duty - ref) < 1e-9:
            continue
        trials += [(duty, d) for d in durations]
    trials += [(ref, d) for d in durations]
    return trials


FIELDS = ["time", "label", "duty", "duration_s", "ran_s", "d_left_m", "d_right_m",
          "forward_m", "speed_mps", "heading_deg", "lateral_m", "note"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-side-in", required=True,
                    help="cabinet fronts to right rear tire outer face at the start (inches or ft' in)")
    ap.add_argument("--duties", type=float, nargs="+", default=DEFAULT_DUTIES)
    ap.add_argument("--durations", type=float, nargs="+", default=DEFAULT_DURATIONS)
    ap.add_argument("--ref", type=float, default=DEFAULT_REF, help="reference duty run first and last")
    ap.add_argument("--trim-right", type=float, default=1.0,
                    help="right duty = duty x trim (below 1 slows the right side, cancels a left pull)")
    ap.add_argument("--repeats", type=int, default=1, help="run the whole trial list this many times")
    ap.add_argument("--max-travel-m", type=float, default=DEFAULT_MAX_TRAVEL_M,
                    help="longest allowed run (m) before the car would reach an obstacle")
    ap.add_argument("--label", default="straight", help="free text stored in every row")
    ap.add_argument("--log", default=os.path.expanduser("~/straight_runs.csv"))
    args = ap.parse_args()

    start_side_in = parse_inches(args.start_side_in)
    trials = build_trials(args.duties, args.durations, args.ref)
    if args.ref in args.duties and len(args.duties) == 1:
        trials = [(args.ref, d) for d in args.durations]  # one duty: no extra reference runs
    trials = trials * max(1, args.repeats)
    label = args.label if args.trim_right == 1.0 else f"{args.label} trimR={args.trim_right:.3f}"
    n = len(trials)
    print(f"Right side trim {args.trim_right:.3f}.")
    print(f"{n} trials. Start side offset {start_side_in:.2f} in. Logging to {args.log}")
    print(f"Runway guard: trials predicted to go past {args.max_travel_m:.2f} m are shortened.")
    print("Keep a hand near the car. Ctrl+C stops the motors.")
    k_speed = None  # highest observed (average speed / duty), m/s per unit duty

    new_file = not os.path.exists(args.log)
    with open(args.log, "a", newline="") as fh, Drive() as drv:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        try:
            for i, (duty, dur) in enumerate(trials, 1):
                if k_speed is not None and k_speed > 0:
                    predicted = SAFETY * k_speed * duty * dur
                    if predicted > args.max_travel_m:
                        new_dur = max(0.25, round(args.max_travel_m / (SAFETY * k_speed * duty) / 0.05) * 0.05)
                        print(f"  Runway guard: {dur:.2f} s predicted {predicted:.2f} m, shortened to {new_dur:.2f} s")
                        dur = new_dur
                print(f"\n[{i}/{n}] duty {duty:.2f} for {dur:.2f} s")
                s = input("  Rear tires on the edge, right rear on the mark, then Enter (s skip, q quit): ")
                s = s.strip().lower()
                if s == "q":
                    break
                if s == "s":
                    continue
                for k in (2, 1):
                    print(f"  {k}...")
                    time.sleep(1.0)

                t0 = time.perf_counter()
                drv.set(duty, duty * args.trim_right)
                while time.perf_counter() - t0 < dur:
                    time.sleep(0.005)
                drv.stop()
                ran = time.perf_counter() - t0

                dl = ask_inches("  Left rear tire, start edge to back of tire (in): ")
                if dl is None:
                    continue
                dr = ask_inches("  Right rear tire, start edge to back of tire (in): ")
                if dr is None:
                    continue
                side = ask_inches("  Cabinet fronts to right rear tire outer face (in): ")
                if side is None:
                    continue
                note = input("  Note (Enter for none): ").strip()

                dl_m, dr_m = dl * IN_TO_M, dr * IN_TO_M
                fwd = 0.5 * (dl_m + dr_m)
                heading = math.degrees(math.atan2(dr_m - dl_m, TIRE_SPACING_M))  # + = turned left
                lateral = (side - start_side_in) * IN_TO_M                       # + = drifted left
                speed = fwd / ran if ran > 0 else float("nan")
                if ran > 0 and duty > 0 and fwd > 0:
                    k_now = speed / duty
                    k_speed = k_now if k_speed is None else max(k_speed, k_now)

                w.writerow({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "label": label, "duty": f"{duty:.3f}", "duration_s": f"{dur:.2f}",
                    "ran_s": f"{ran:.3f}", "d_left_m": f"{dl_m:.4f}", "d_right_m": f"{dr_m:.4f}",
                    "forward_m": f"{fwd:.4f}", "speed_mps": f"{speed:.4f}",
                    "heading_deg": f"{heading:.2f}", "lateral_m": f"{lateral:.4f}", "note": note,
                })
                fh.flush()
                print(f"  forward {fwd:.3f} m, average speed {speed:.3f} m/s, "
                      f"heading {heading:+.1f} deg, drift {lateral * 100:+.1f} cm")
        except KeyboardInterrupt:
            drv.stop()
            print("\nStopped.")
    print(f"Done. Log: {args.log}")


if __name__ == "__main__":
    main()
