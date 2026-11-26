# Niti Vahan vehicle tests

From the checkout root, activate the same Python environment used for the competition tasks. The Mac/Linux launcher also accepts a Python executable through `VEHICLE_PYTHON`. Set `COPPELIASIM_BIN` if CoppeliaSim is outside the runner's default search path.

| Command | Input and output |
| --- | --- |
| `./test-vehicle` | Run the current Task 1B controller on a fresh scene copy. Record initial pose, time-stamped trajectory, paired steering targets and actual joint angles, and criterion results under `vehicle-test-runs/sim/`. |
| `./test-vehicle steering` | Analyze the bundled Task 1B trajectory. To include measured joint positions from a fresh simulation, append its run directory. |
| `./test-vehicle vision` | Replay all 20 Task 1C clips. Save overlays, predictions and a summary under `vehicle-test-runs/vision/`. Append clip paths to replay a subset. |
| `./test-vehicle vision --baseline <earlier-run>/predictions.csv` | Identify the first changed output in each clip. Use reviewed frame labels with `--labels labels.csv` to check correctness. |

Task 1B follows the documented 120 s example with lane changes at 15, 60 and 95 s. The local regression check accepts a maximum lane-center error of 0.02 m during the last second of each segment. The competition's official score is separate.

The hardware fault diagnosis skill works with the robot's board revision, wiring, firmware and serial evidence when they are available.
