#!/usr/bin/env python3
"""Safely fix the lane-change NameError in a Task 3 controller file."""
from pathlib import Path
import argparse
import sys

OLD = """            tracker_switch = (
                abs(abs(jump) - LANE_WIDTH_M) <= 0.14
            )
"""
NEW = """            expected_jump = self.lane_change_direction * LANE_WIDTH_M
            tracker_switch = (
                self.lane_change_direction * jump > 0.0
                and abs(jump - expected_jump) <= 0.14
            )
"""

# Also repair the exact bad block introduced by the follow-up command.
BAD = """            expected_jump = direction * LANE_WIDTH_M
            tracker_switch = (
                direction * jump > 0.0
                and abs(jump - expected_jump) <= 0.14
            )
"""


def main():
    parser = argparse.ArgumentParser(
        description="Fix NameError: direction is not defined in Task 3 lane-change logic."
    )
    parser.add_argument("file", nargs="?", default="task3.py", help="Task 3 Python file (default: task3.py)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"ERROR: file not found: {path}")
        return 2

    source = path.read_text(encoding="utf-8")
    bad_count = source.count(BAD)
    old_count = source.count(OLD)

    if bad_count == 1:
        updated = source.replace(BAD, NEW, 1)
        reason = "replaced the undefined local 'direction' with self.lane_change_direction"
    elif bad_count == 0 and old_count == 1:
        updated = source.replace(OLD, NEW, 1)
        reason = "added direction-aware lane-tracker switch verification"
    elif source.count("self.lane_change_direction * jump > 0.0") == 1:
        print("No change needed: the lane-change direction check is already fixed.")
        return 0
    else:
        print(
            "ERROR: could not safely identify exactly one lane-change tracker-switch block.\n"
            f"Found bad-block matches: {bad_count}; original-block matches: {old_count}.\n"
            "No file was changed. Send the actual task3.py so I can patch its exact version."
        )
        return 3

    try:
        compile(updated, str(path), "exec")
    except SyntaxError as exc:
        print(f"ERROR: proposed edit failed Python syntax validation: {exc}")
        return 4

    backup = path.with_suffix(path.suffix + ".before_nameerror_fix")
    if backup.exists():
        print(f"ERROR: backup already exists: {backup}; refusing to overwrite it.")
        return 5

    backup.write_text(source, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")

    print(f"Fixed: {path}")
    print(f"Backup: {backup}")
    print(f"Change: {reason}")
    print("Python syntax check: PASS")
    if "LANE_CHANGE_RAMP_FRAMES" not in updated or "lane_change_target_m" not in updated:
        print(
            "WARNING: this file does not appear to contain the smooth lane-change-ramp patch. "
            "Do not assume the complete v2 lane-change fix is present."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
