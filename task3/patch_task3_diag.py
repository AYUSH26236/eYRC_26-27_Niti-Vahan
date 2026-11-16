#!/usr/bin/env python3
"""
Diagnostic + safety patch for the "circles after first obstacle" problem.
Works on v2 or v3.

  python patch_task3_diag.py task3_patched_v2.py   ->  task3_diag.py

Adds:
  1. NV_NO_LANE_CHANGE=1   disables lane changes (bisect switch)
  2. Per-frame log while a lane change / wait is active (NV_DEBUG_LOG=run.txt)
  3. Lane-change abort after LANE_CHANGE_MAX_FRAMES (offset can no longer stick)
  4. Steering no longer saturates while stopped in the "target lane blocked" wait
"""
import re
import sys
import py_compile

src = sys.argv[1] if len(sys.argv) > 1 else "task3_patched_v2.py"
out = "task3_diag.py"

with open(src, encoding="utf-8") as f:
    code = f.read().replace("\r\n", "\n")

fails = []


def sub(tag, pattern, repl):
    global code
    code, n = re.subn(pattern, repl, code, count=0)
    if n != 1:
        fails.append("%s: expected 1 match, found %d" % (tag, n))


sub("consts",
    r'(DEBUG_LOG_PATH = os\.environ\.get\("NV_DEBUG_LOG", ""\)\.strip\(\)\n)',
    r'\1NO_LANE_CHANGE = os.environ.get("NV_NO_LANE_CHANGE", "0").strip() == "1"\n'
    r'LANE_CHANGE_MAX_FRAMES = 120\n')

sub("init",
    r'( +)self\.lane_change_stable = 0\n( +)self\.lane_change_progress = 0\.0\n( +)self\.previous_lane_error = None\n( +)self\.last_lane_change_frame = -LANE_CHANGE_REARM_FRAMES\n',
    r'\1self.lane_change_stable = 0\n\2self.lane_change_progress = 0.0\n\3self.previous_lane_error = None\n'
    r'\4self.last_lane_change_frame = -LANE_CHANGE_REARM_FRAMES\n\4self.lane_change_frames = 0\n')

sub("switch",
    r'(    def _maybe_start_lane_change\(\n        self,\n        frame,\n        lane,\n        obstacles\n    \):\n)',
    r'\1        if NO_LANE_CHANGE:\n            return False\n\n')

sub("start-reset",
    r'( +)self\.lane_change_active = True\n',
    r'\1self.lane_change_active = True\n\1self.lane_change_frames = 0\n')

sub("abort",
    r'(    def _update_lane_change\(\n        self,\n        lane\n    \):\n        if not self\.lane_change_active:\n            return\n)',
    r'\1\n'
    '        self.lane_change_frames += 1\n\n'
    '        if self.lane_change_frames > LANE_CHANGE_MAX_FRAMES:\n'
    '            self.lane_offset_m = 0.0\n'
    '            self.lane_change_active = False\n'
    '            self.lane_change_stable = 0\n'
    '            self.lane_change_progress = 0.0\n'
    '            self.previous_lane_error = None\n'
    '            self.last_lane_change_frame = self.frame_count\n'
    '            print("[LANE CHANGE] ABORT: did not finish in %d frames; offset cleared"\n'
    '                  % LANE_CHANGE_MAX_FRAMES)\n'
    '            return\n')

sub("stanley-speed",
    r'( +)longitudinal_speed = max\(\n +speed \* WHEEL_RADIUS,\n +0\.05\n +\)\n',
    r'\1ref_speed = (\n\1    OBSTACLE_NEAR_SPEED\n\1    if self.target_lane_waiting\n\1    else speed\n\1)\n'
    r'\1longitudinal_speed = max(ref_speed * WHEEL_RADIUS, 0.05)\n')

sub("framelog",
    r'( +)return \(\n +self\.last_steering,\n +speed\n +\)\n',
    r'\1if DEBUG_LOG_PATH and (self.lane_change_active or self.target_lane_waiting):\n'
    r'\1    _debug_log(\n'
    r'\1        "LC frame=%d active=%s wait=%s dir=%d offset=%.3f e_lane=%s e_total=%.3f psi=%.3f steer=%.1fdeg speed=%.2f reason=%s lc_frames=%d"\n'
    r'\1        % (\n'
    r'\1            self.frame_count, self.lane_change_active, self.target_lane_waiting,\n'
    r'\1            self.lane_change_direction, self.lane_offset_m,\n'
    r'\1            None if lane is None else round(float(lane["e"]), 3),\n'
    r'\1            e, psi, math.degrees(self.last_steering), speed, reason,\n'
    r'\1            self.lane_change_frames,\n'
    r'\1        )\n'
    r'\1    )\n\n'
    r'\g<0>')

if fails:
    print("NOTHING WRITTEN. Failed:")
    for x in fails:
        print("  -", x)
    sys.exit(1)

with open(out, "w", encoding="utf-8") as f:
    f.write(code)
py_compile.compile(out, doraise=True)
print("OK: wrote", out)
