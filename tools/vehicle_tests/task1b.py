"""Run the current Task 1B controller against a fixed lane-change schedule."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import queue
import shutil

from steering_recorder import SteeringRecorder

PROJECT = Path(__file__).resolve().parents[2]
SCENE = PROJECT / "task1b" / "Task_1B.ttt"
VEHICLE = "/Niti_Vahan"
# Task 1B README example and lane centers, expressed in simulated seconds and meters.
SEGMENTS = [(0, 15, 0.0), (15, 60, 0.2), (60, 95, 0.0), (95, 120, 0.2)]
SCHEDULE = [(15, "R"), (60, "L"), (95, "R")]
# Local regression threshold for holding the commanded lane during the last second.
HOLD_ERROR_M = 0.02


def run(sim, folder):
    controller = folder / "path_tracking.py"
    shutil.copy2(SCENE.with_name(controller.name), controller)
    (folder / "controller-input.json").write_text(json.dumps({
        "sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "source": str(SCENE.with_name(controller.name)), "schedule": SCHEDULE,
        "duration_s": 120, "log_hz": 20, "hold_error_m": HOLD_ERROR_M
    }, indent=2))
    spec = importlib.util.spec_from_file_location("nv_path_tracking", controller)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with (folder / "steering.csv").open("w", newline="") as log:
        recorder = SteeringRecorder(sim, driver, log)
        driver.run_session(recorder, queue.Queue(), str(folder / "trajectory.csv"), 20, SCHEDULE)


def evaluate(folder):
    with (folder / "trajectory.csv").open(newline="") as handle:
        rows = [{k: float(row[k]) for k in ("t", "x", "y", "target_y")} for row in csv.DictReader(handle)]
    if not rows or not all(math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("Empty or nonfinite trajectory")
    times = [row["t"] for row in rows]
    gaps = [b - a for a, b in zip(times, times[1:])]
    # The controller logs at 20 Hz. Allow one missed sample at the run edges.
    complete = (0 <= times[0] <= 0.101 and 119.899 <= times[-1] <= 120.101
                and all(0 < gap <= 0.101 for gap in gaps))
    checks = [dict(name="120 s continuous trajectory", passed=complete,
                   first_s=times[0], last_s=times[-1], samples=len(rows))]
    progress = rows[0]["x"] - rows[-1]["x"]
    checks.append(dict(name="Forward progress along world -x", passed=progress > 0,
                       travel_m=round(progress, 4)))
    for start, end, target in SEGMENTS:
        # Exclude the boundary sample; CSV times are rounded by the controller.
        segment = [row for row in rows if start < row["t"] < end]
        hold = [row for row in segment if row["t"] >= end - 1]
        commanded = bool(segment) and all(abs(row["target_y"] - target) < 1e-6 for row in segment)
        error = max((abs(row["y"] - target) for row in hold), default=None)
        checks.append(dict(name=f"{start}–{end} s: hold y={target:.2f} m in the final second",
                           passed=commanded and error is not None and error <= HOLD_ERROR_M + 1e-10,
                           expected_target_seen=commanded, max_hold_error_m=error,
                           local_tolerance_m=HOLD_ERROR_M))
    return checks
