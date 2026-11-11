# Experiments

Controlled tests that measured the models used by the filters. The logs are kept raw, as written by the scripts; known problems in them are documented here rather than edited out.

| Script | Log | What it measured | Result used |
|---|---|---|---|
| `straight_run.py` | `logs/straight_runs.csv` | Distance and heading after timed straight runs, tape-measured | Breakaway at duty 0.55 to 0.6; cruise 0.42 m/s at duty 0.8 (slope between 0.75 s and 1.5 s runs); left pull about 14 deg/m |
| `gyro_spin.py` | `logs/gyro_spins.csv` | Turn angle per motor tap, from the integrated gyro | Spins about 70 deg/s with 0.05 to 0.06 s spin-up; braked pivots 18.1 deg per 0.8 s tap (22.7 deg/s), 11% spread |
| `imu_test.py` | `logs/imu_still.csv` | Gyro bias and noise with the car still | Bias +1.51 deg/s, noise 0.026 deg/s per sample at 100 Hz (one 30 s run) |
| `refine_test.py` | `logs/cov_stops.csv` | Marker range and bearing from 5 corner refinement methods on identical frames | AprilTag refinement plus a 0.4 px outward edge correction (see the main README) |
| (no script kept) | `logs/spin_runs.csv` | Whether a full spin breaks loose at duty 0.6 to 0.9 | It does not: spins only break loose at duty 1.0, so turns became braked pivots |

## Known issues in the logs

- **`gyro_spins.csv`, rows 2 to 13** ("spin taps nov9"): logged before the left/right motor channel swap was found, so the `direction` labels are mirrored (a "left" tap turned right; see the negative angles). Row 14 ("after side swap", 21:18) is the first correct row.
- **`straight_runs.csv`:**
  - Row 33's note ("left rear is actually 21.5") corrects its left-tire distance from 21.0 to 21.5 in; the logged numbers were not updated.
  - The `trimR=0.950` rows were logged before the channel fix, so the trim actually slowed the physical left side.
  - Headings come from two tape readings 14 cm apart, so they are quantized to about 1.3 deg.
- **`cov_stops.csv`:** labels are stops (A about 0.9 m, B about 1.65 m, C about 2.5 m; c/l/r = centre, left, right of the image). The tape-measured ranges for the stops were recorded by hand and are not in this file. All rows use the test's 0.25 px edge correction, not the final 0.4 px. A and B were each measured twice, at different placements.

The hardware scripts expect the car's code in `../car` (they add it to the import path) and must not run while `car/motor_node.py` holds the motor pins.
