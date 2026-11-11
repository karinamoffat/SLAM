# Results

All plots come from the scripts in `pipeline/` run on recorded drives (replayed offline). Plots committed here were made on the car before the final cosmetic change to the legends (they cover marker 3 at the top of the maps); re-running the scripts gives the same numbers with the legend below the map.

## `m8_slam/`: EKF-SLAM, drives 4 to 6 (`ekf_slam.py`)

| Drive | Markers inside 95% ellipse | Map error RMS | Recoveries | Map rotation removed by alignment |
|---|---|---|---|---|
| run 4 | 11 of 12 | 10.2 cm | 6 | -1.1 deg |
| run 5 | 10 of 11 | 8.4 cm | 3 | -3.8 deg |
| run 6 | 11 of 11 | 9.8 cm | 1 | -13.7 deg |

- `*_slam_map.png`: left, the map as estimated in the start frame (not aligned; drive 6's 13.7 deg rotation is visible here); right, aligned to the tape map, with errors in red.
- `*_slam_time.png`: car uncertainty, every marker's uncertainty from the full covariance (loop closures in green), and NIS per sighting (gated-out sightings in red, recoveries in purple).
- Markers 9 and 12 were never added in runs 5 and 6, and 9 not in run 4: the drives rarely came close enough.

## `m8_localization/`: EKF localization against the tape map, drives 4 to 6 (`ekf_localize.py`)

Measured pivot model. Honesty over the three drives: 33 graded sightings, mean NIS 2.00, 94% inside the 95% region (1.54 / 90%, 3.11 / 92%, 1.01 / 100% per drive). With the earlier pivot model, which was not fitted on these drives: NIS 2.62, 94%.

## `m7_localization/`: the first localization runs

Drives 1 and 2, used to tune the process-noise scale (x4). The large ellipses in run 1 come from a stall (a forward command that did not move the car) and a recovery.

## `sim/`: simulated drive with known truth (`sim_drive.py --seed 1`)

12 of 12 markers inside, 4.4 cm map RMS; car position inside its 95% ellipse 100% of the time.

## `calibration/`

`camera_retilt_floor_line.png`: the floor line under the cabinets as detected per column (green) and the fitted camera model (red), giving tilt 14.76 deg and roll 1.75 deg (fit RMS 0.61 px).
