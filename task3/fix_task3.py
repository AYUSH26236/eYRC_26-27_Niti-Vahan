#!/usr/bin/env python3
"""Apply the reviewed Task 3 safety and speed fixes to task3.py.

Usage:
    python fix_task3.py task3.py

The original file is left untouched. The patched file is written as
    task3_fixed.py
in the same directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(
            f"Could not safely apply '{label}': expected one matching block, "
            f"found {count}. Your task3.py may differ from the pasted version."
        )
    return source.replace(old, new, 1)


def patch(source: str) -> str:
    # 1) Moderate speed tuning. The hard 3.0 rad/s and Speed 40 caps remain.
    source = replace_once(
        source,
        """EASE_E_GAIN = 2.8
EASE_PSI_GAIN = 0.55
MIN_SPEED_FRACTION = 0.55

NORMAL_SPEED = 3.00
SPEED_40_LIMIT = 1.18
ABSOLUTE_SPEED_LIMIT = 3.00

LANE_CHANGE_SPEED = 1.35
OBSTACLE_NEAR_SPEED = 1.05
""",
        """EASE_E_GAIN = 2.2
EASE_PSI_GAIN = 0.45
MIN_SPEED_FRACTION = 0.65

NORMAL_SPEED = 3.00
SPEED_40_LIMIT = 1.18
ABSOLUTE_SPEED_LIMIT = 3.00

# Slightly less conservative outside the restricted zone. The Speed 40
# cap below still overrides every one of these speeds when the zone is active.
LANE_CHANGE_SPEED = 1.65
OBSTACLE_NEAR_SPEED = 1.25
""",
        "moderate outside-zone speed tuning",
    )

    source = replace_once(
        source,
        "LANE_CHANGE_CROSSING_M = 0.18\n",
        "",
        "remove obsolete crossing threshold",
    )

    # 2) Remove the elapsed-time exit-zone guess; rely on the second sign.
    source = replace_once(
        source,
        """# If the 2nd Speed40 sign is missed, allow parking after this many
# simulation seconds inside the zone (good run: zone lasted ~45 s).
ZONE_FALLBACK_S = float(os.environ.get("NV_ZONE_FALLBACK_S", "50"))
LANE_CHANGE_MAX_FRAMES = 120
""",
        """# An unverified lane change gets a bounded slow recovery period; if it
# still cannot be verified, the controller stops without snapping back.
LANE_CHANGE_RAMP_FRAMES = 18
LANE_CHANGE_MAX_FRAMES = 120
LANE_CHANGE_RECOVERY_FRAMES = 100
LANE_CHANGE_RECOVERY_SPEED = 0.95
""",
        "remove elapsed-time Speed 40 fallback and add lane-change ramp setting",
    )

    # 3) Accept a clipped red-ring Speed 40 sign at any image boundary.
    source = replace_once(
        source,
        """        clipped_speed = (
            fill_ratio < 0.72
            and edge_clipped
            and x >= int(width * 0.82)
            and y <= int(height * 0.40)
            and 0.50 <= aspect <= 1.55
            and circularity >= 0.20
            and 300.0 <= area <= frame_area * 0.05
        )
""",
        """        clipped_speed = (
            fill_ratio < 0.72
            and edge_clipped
            and 0.50 <= aspect <= 1.55
            and circularity >= 0.20
            and max(150.0, frame_area * 0.0005)
                <= area <= frame_area * 0.05
        )
""",
        "location-independent clipped Speed 40 detection",
    )

    # 4) Speed-zone state no longer stores a timer for a guessed exit.
    source = replace_once(
        source,
        """        self.speed40_events = 0
        self.speed_zone = False
        self.speed_zone_start = None
        self.now_s = 0.0
""",
        """        self.speed40_events = 0
        self.speed_zone = False
        self.now_s = 0.0
""",
        "remove Speed 40 fallback timer state",
    )

    # 5) Separate the final lane-change target from its smoothly ramped offset.
    source = replace_once(
        source,
        """        self.lane_offset_m = 0.0
        self.corner_bias_m = 0.0
        self.lane_change_active = False
        self.lane_change_direction = 0
        self.lane_change_stable = 0
        self.previous_lane_error = None
""",
        """        self.lane_offset_m = 0.0
        self.lane_change_target_m = 0.0
        self.lane_change_ramp_progress = 0.0
        self.lane_change_recovery = False
        self.lane_change_faulted = False
        self.corner_bias_m = 0.0
        self.lane_change_active = False
        self.lane_change_direction = 0
        self.lane_change_stable = 0
        self.lane_change_progress = 0.0
        self.previous_lane_error = None
""",
        "lane-change ramp and fail-safe state",
    )

    source = replace_once(
        source,
        """        if NO_LANE_CHANGE:
            return False

        if lane is None:
""",
        """        if NO_LANE_CHANGE:
            return False

        # After the bounded recovery window, a failed change keeps its target
        # offset and holds the vehicle stopped. Never snap back into the old lane.
        if self.lane_change_faulted:
            self.target_lane_waiting = True
            return False

        if lane is None:
""",
        "hold a failed lane change stopped",
    )

    source = replace_once(
        source,
        """            self.lane_change_direction = direction

            self.lane_offset_m = (
                direction
                * LANE_CHANGE_DISTANCE_M
            )

            self.lane_change_active = True
            self.lane_change_frames = 0

            self.lane_change_stable = 0
""",
        """            self.lane_change_direction = direction

            # Ramp the lateral target in over several control frames rather
            # than applying the full 0.40 m offset in one abrupt step.
            self.lane_change_target_m = (
                direction * LANE_CHANGE_DISTANCE_M
            )
            self.lane_offset_m = 0.0
            self.lane_change_ramp_progress = 0.0
            self.lane_change_recovery = False
            self.lane_change_faulted = False

            self.lane_change_active = True
            self.lane_change_frames = 0

            self.lane_change_stable = 0
""",
        "start lane change with a smooth offset ramp",
    )

    # 6) Completing a lane change clears only the temporary control offset.
    source = replace_once(
        source,
        """        self.lane_offset_m = 0.0

        self.lane_change_active = False
        self.lane_change_stable = 0
        self.lane_change_progress = 0.0
        self.previous_lane_error = None
""",
        """        self.lane_offset_m = 0.0
        self.lane_change_target_m = 0.0
        self.lane_change_ramp_progress = 0.0
        self.lane_change_recovery = False
        self.lane_change_faulted = False

        self.lane_change_active = False
        self.lane_change_stable = 0
        self.previous_lane_error = None
""",
        "reset lane-change state after verified completion",
    )

    # 7) Replace the old abort/crossing logic with ramp + verified completion.
    old_update = """    def _update_lane_change(
        self,
        lane
    ):
        if not self.lane_change_active:
            return

        self.lane_change_frames += 1

        if self.lane_change_frames > LANE_CHANGE_MAX_FRAMES:
            self.lane_offset_m = 0.0
            self.lane_change_active = False
            self.lane_change_stable = 0
            self.lane_change_progress = 0.0
            self.previous_lane_error = None
            self.last_lane_change_frame = self.frame_count
            print("[LANE CHANGE] ABORT: did not finish in %d frames; offset cleared"
                  % LANE_CHANGE_MAX_FRAMES)
            return

        if lane is None:
            return

        direction = self.lane_change_direction
        current_error = float(lane["e"])

        previous_error = self.previous_lane_error

        jump = (
            None
            if previous_error is None
            else current_error - previous_error
        )

        # Lane tracker may switch suddenly to the adjacent lane.
        if jump is not None:
            tracker_switch = (
                abs(
                    abs(jump)
                    - LANE_WIDTH_M
                )
                <= 0.14
            )

            if tracker_switch:
                self._finish_lane_change(
                    "lane tracker switched "
                    "(jump=%+.3f m)" % jump
                )
                return

        # Fallback: observe lateral crossing.
        progress = (
            -direction
            * current_error
        )

        self.lane_change_progress = max(
            self.lane_change_progress,
            progress
        )

        if (
            self.lane_change_progress
            >= LANE_CHANGE_CROSSING_M
            and (
                direction * current_error
                > 0.0
            )
        ):
            self._finish_lane_change(
                "crossing detected"
            )
            return

        # Fallback: target held stably.
        target_error = (
            current_error
            + self.lane_offset_m
        )

        if (
            abs(target_error)
            <= LANE_CHANGE_TARGET_TOLERANCE_M
        ):
            self.lane_change_stable += 1
        else:
            self.lane_change_stable = 0

        if (
            self.lane_change_stable
            >= LANE_CHANGE_STABLE_FRAMES
        ):
            self._finish_lane_change(
                "adjacent lane target stable"
            )
            return

        self.previous_lane_error = (
            current_error
        )
"""
    new_update = """    def _update_lane_change(
        self,
        lane
    ):
        if (
            not self.lane_change_active
            or self.lane_change_faulted
        ):
            return

        self.lane_change_frames += 1

        if self.lane_change_frames > LANE_CHANGE_MAX_FRAMES:
            # Continue the same target with a lower speed for a bounded
            # recovery window. Never snap the offset back toward the obstacle.
            if not self.lane_change_recovery:
                self.lane_change_recovery = True
                print(
                    "[LANE CHANGE] recovery mode: target offset retained; "
                    "slowing to %.2f rad/s"
                    % LANE_CHANGE_RECOVERY_SPEED
                )

            if (
                self.lane_change_frames
                > LANE_CHANGE_MAX_FRAMES + LANE_CHANGE_RECOVERY_FRAMES
            ):
                self.lane_change_faulted = True
                self.target_lane_waiting = True
                self.last_lane_change_frame = self.frame_count
                print(
                    "[LANE CHANGE] FAIL-SAFE STOP: adjacent lane was not "
                    "verified after recovery; target offset retained"
                )
                return

        # Smoothstep ramp: zero initial/final slope avoids a steering step.
        self.lane_change_ramp_progress = min(
            1.0,
            self.lane_change_frames / float(LANE_CHANGE_RAMP_FRAMES)
        )
        p = self.lane_change_ramp_progress
        smooth_p = p * p * (3.0 - 2.0 * p)
        self.lane_offset_m = (
            self.lane_change_target_m * smooth_p
        )

        if lane is None:
            return

        current_error = float(lane["e"])
        previous_error = self.previous_lane_error

        jump = (
            None
            if previous_error is None
            else current_error - previous_error
        )

        # A sudden one-lane-width change means the lane tracker has adopted
        # the adjacent lane; only then is it safe to remove the temporary bias.
        if jump is not None:
            tracker_switch = (
                abs(abs(jump) - LANE_WIDTH_M) <= 0.14
            )

            if tracker_switch:
                self._finish_lane_change(
                    "lane tracker switched (jump=%+.3f m)" % jump
                )
                return

        # Require the vehicle to reach the FULL adjacent-lane target. Compare
        # against the final target, not the currently ramped offset, otherwise
        # the controller could declare success halfway through the manoeuvre.
        target_error = (
            current_error + self.lane_change_target_m
        )

        if abs(target_error) <= LANE_CHANGE_TARGET_TOLERANCE_M:
            self.lane_change_stable += 1
        else:
            self.lane_change_stable = 0

        if self.lane_change_stable >= LANE_CHANGE_STABLE_FRAMES:
            self._finish_lane_change(
                "adjacent lane target stable"
            )
            return

        self.previous_lane_error = current_error
"""
    source = replace_once(
        source,
        old_update,
        new_update,
        "safe lane-change ramp, completion, and timeout behaviour",
    )

    source = replace_once(
        source,
        """        # Lane change
        if (
            self.lane_change_active
            and speed > LANE_CHANGE_SPEED
        ):
            speed = LANE_CHANGE_SPEED
            reason = "lane_change"

        # Obstacle proximity
""",
        """        # Lane change. Once the normal window expires, keep steering toward
        # the same target but reduce speed for a bounded recovery period.
        if self.lane_change_active:
            lane_change_limit = (
                LANE_CHANGE_RECOVERY_SPEED
                if self.lane_change_recovery
                else LANE_CHANGE_SPEED
            )
            if speed > lane_change_limit:
                speed = lane_change_limit
                reason = (
                    "lane_change_recovery"
                    if self.lane_change_recovery
                    else "lane_change"
                )

        # Obstacle proximity
""",
        "slow the extended lane-change recovery period",
    )

    # 8) Do not let a timer expire and falsely satisfy the parking prerequisite.
    source = replace_once(
        source,
        """    def _prerequisites_complete(self):
        if not self.hospital_done:
            return False

        if self.speed40_events >= 2 and not self.speed_zone:
            return True

        # Fallback: second Speed40 sign was missed, but we have been in
        # the restricted zone long enough that the exit must have passed.
        if (
            self.speed40_events >= 1
            and self.speed_zone
            and self.speed_zone_start is not None
            and (self.now_s - self.speed_zone_start) >= ZONE_FALLBACK_S
        ):
            return True

        return False
""",
        """    def _prerequisites_complete(self):
        # Parking is enabled only after the Hospital stop is complete and
        # the second distinct Speed40 sign confirms the zone exit.
        return (
            self.hospital_done
            and self.speed40_events >= 2
            and not self.speed_zone
        )
""",
        "require visual confirmation of the Speed 40 exit sign",
    )

    # 9) Remove the now-unused speed-zone timer update.
    source = replace_once(
        source,
        """        self.now_s = now
        self._update_hospital_detection(signs)
        self._update_speed_zone(signs)
        if self.speed_zone and self.speed_zone_start is None:
            self.speed_zone_start = now
        elif not self.speed_zone:
            self.speed_zone_start = None

        self.obstacle_tracker.update(
""",
        """        self.now_s = now
        self._update_hospital_detection(signs)
        self._update_speed_zone(signs)

        self.obstacle_tracker.update(
""",
        "remove elapsed-time zone bookkeeping",
    )

    # 10) Apply the restricted-zone speed cap even in the exclusive parking mode.
    source = replace_once(
        source,
        """        if self.parking_started:
            steering, speed, parked = (
                self.parking_controller.step(
                    frame,
                    self.parking_detector,
                    now
                )
            )

            self.last_steering = steering
            self.done = parked
            self.cap_reason = "parking"

            return steering, speed
""",
        """        if self.parking_started:
            steering, speed, parked = (
                self.parking_controller.step(
                    frame,
                    self.parking_detector,
                    now
                )
            )

            # Keep the Speed 40 limit invariant across every controller mode.
            if self.speed_zone:
                speed = min(float(speed), SPEED_40_LIMIT)

            self.last_steering = steering
            self.done = parked
            self.cap_reason = (
                "parking_speed_zone"
                if self.speed_zone
                else "parking"
            )

            return steering, speed
""",
        "apply Speed 40 cap during parking control",
    )

    source = replace_once(
        source,
        """            self.last_steering = steering
            self.done = parked
            self.cap_reason = "parking"

            return steering, speed

        # --------------------------------------------------------------
        # Obstacle avoidance
""",
        """            if self.speed_zone:
                speed = min(float(speed), SPEED_40_LIMIT)

            self.last_steering = steering
            self.done = parked
            self.cap_reason = (
                "parking_speed_zone"
                if self.speed_zone
                else "parking"
            )

            return steering, speed

        # --------------------------------------------------------------
        # Obstacle avoidance
""",
        "apply Speed 40 cap on the frame parking is engaged",
    )

    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default="task3.py",
        help="original Task 3 source file (default: task3.py)",
    )
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.is_file():
        print(f"ERROR: source file not found: {source_path}", file=sys.stderr)
        return 2

    original = source_path.read_text(encoding="utf-8")

    try:
        modified = patch(original)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "No output was written and the original file was not changed.",
            file=sys.stderr,
        )
        return 3

    output_path = source_path.with_name("task3_fixed.py")
    output_path.write_text(modified, encoding="utf-8")

    # Syntax-check the result before reporting success.
    try:
        compile(modified, str(output_path), "exec")
    except SyntaxError as exc:
        output_path.unlink(missing_ok=True)
        print(f"ERROR: patched file has a syntax error: {exc}", file=sys.stderr)
        return 4

    print(f"Patched file created: {output_path}")
    print("Original task3.py was left unchanged.")
    print("Run the evaluator with task3_fixed.py after reviewing the diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
