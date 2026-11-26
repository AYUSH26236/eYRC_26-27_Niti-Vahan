"""Analyze Task 1B steering commands, lane response, and optional joint feedback."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def records(file, fields):
    with file.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{file}: missing columns {sorted(missing)}")
        data = [{k: float(row[k]) for k in fields} for row in reader]
    if not data or not all(math.isfinite(x) for row in data for x in row.values()):
        raise ValueError(f"{file}: missing or nonfinite samples")
    return data


def ordered(rows, key):
    if any(a[key] >= b[key] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{key} must increase throughout the recording")


def commands(rows):
    issues = {}
    segments = []
    target = None
    direction = 0
    for row in rows:
        delta, left, right = (row[k] for k in ("steering", "left_wheel", "right_wheel"))
        def issue(name):
            entry = issues.setdefault(name, dict(first_t_s=row["t"], samples=0))
            entry["samples"] += 1
        if abs(delta) > math.radians(30) + 0.0001:
            issue("virtual_angle_exceeds_30_degree_limit_check_units")
        if abs(delta) > 0.01:
            if left*delta <= 0 or right*delta <= 0:
                issue("wheel_command_sign")
            inner, outer = (left, right) if delta > 0 else (right, left)
            if abs(inner) + 0.0001 < abs(outer):
                issue("inner_outer_reversed")
        # Cross-check against the documented wheelbase, track and kingpin offset.
        curvature = math.tan(delta) / 0.120
        half_track = 0.110 / 2 - 0.0275
        expected = [math.atan2(0.120*curvature, 1 - side*half_track*curvature)
                    for side in (1, -1)]
        if max(abs(a-b) for a, b in zip((left, right), expected)) > 0.001:
            issue("ackermann_geometry_disagreement")
        if row["target_y"] != target:
            if target is not None:
                direction = 1 if row["target_y"] > target else -1
            target = row["target_y"]
            segments.append(dict(start_s=row["t"], target_m=target, peak_overshoot_m=None))
        segment = segments[-1]
        if direction:
            segment["peak_overshoot_m"] = max(segment["peak_overshoot_m"] or 0, direction*(row["y"]-target))
        segment["end_error_m"] = round(row["y"]-target, 5)
    return dict(command_findings=issues, segments=segments)


def feedback(file, review_rad):
    fields = ("t_command", "t_measured", "left_requested_rad", "right_requested_rad",
              "left_actual_rad", "right_actual_rad")
    data = records(file, fields)
    ordered(data, "t_command")
    ordered(data, "t_measured")
    if any(row["t_measured"] <= row["t_command"] for row in data):
        raise ValueError("Joint feedback must follow its command")
    report = {}
    for side in ("left", "right"):
        diffs = [abs(r[side+"_requested_rad"] - r[side+"_actual_rad"]) for r in data]
        first = next((r for r, diff in zip(data, diffs) if diff > review_rad), None)
        report[side] = dict(median_error_rad=median(diffs), max_error_rad=max(diffs),
                            first_above_review_threshold=first,
                            samples_above_threshold=sum(e > review_rad for e in diffs))
    return dict(samples=len(data), review_threshold_rad=review_rad, wheels=report)


def main():
    root = Path(__file__).resolve().parents[4]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", nargs="?", type=Path, default=root / "task1b/trajectory.csv",
                   help="Task 1B trajectory CSV or a simulator run folder")
    p.add_argument("--feedback", type=Path, help="Paired command/measurement steering.csv")
    p.add_argument("--review-deg", type=float, default=5, help="Local one-step residual review threshold")
    a = p.parse_args()
    if not math.isfinite(a.review_deg) or a.review_deg <= 0:
        p.error("--review-deg must be positive")
    trace = a.trace / "trajectory.csv" if a.trace.is_dir() else a.trace
    pair = a.feedback or trace.with_name("steering.csv")
    rows = records(trace, ("t", "y", "target_y", "steering", "left_wheel", "right_wheel"))
    ordered(rows, "t")
    report = commands(rows)
    report.update(samples=len(rows), trajectory=str(trace))
    report["feedback"] = feedback(pair, math.radians(a.review_deg)) if a.feedback or pair.is_file() else None
    print(json.dumps(report, indent=2))
    return int(bool(report["command_findings"]))


if __name__ == "__main__":
    raise SystemExit(main())
