---
name: steering-path-debugger
description: Diagnose wrong-way steering, lane overshoot, asymmetric turns, or wheel-angle tracking errors in Niti Vahan using the command trace and measured steering joint positions.
---

# Steering and path tracking

Start with `./test-vehicle steering` for the archived Task 1B commands. For the current controller, run `./test-vehicle sim`, then `./test-vehicle steering <run-directory>`. Run both in the project's Python environment. The simulator recorder pairs each wheel's requested radians before the physics step with its measured joint angle immediately after it.

1. Read the task's frame and joint definitions. In Task 1B the vehicle drives along world −x. A positive steering command turns left toward smaller y. Test straight, left, and right from identical controller state, with zero heading error. Inspect the first command after the target changes, subsequent yaw and subsequent y. A correction can legitimately reverse sign near the lane center.
2. Run the analyzer. Inspect the virtual steering limit, both wheel signs and the inner wheel magnitude. For a positive turn the left wheel is inner, and for a negative turn the right wheel is inner. Task 1B uses a 0.120 m wheelbase and an effective half-track of 0.110/2 − 0.0275 m. The command columns are radians. Check for sign reversals, swapped sides, the scale factor 180/π and unit conversion at each interface. Re-derive these conventions for another vehicle.
3. Read the first feedback discrepancy and inspect adjacent samples in `steering.csv`. The default 5° threshold selects samples for review. A short-lived residual at a command transition can be actuator lag. Test whether it remains at a steady target, reverses with direction, or occurs only on one wheel. If feedback is absent, report actuator behavior as unmeasured.
4. Match the event time to `trajectory.csv` for lateral position, target and yaw. Report target-change time, peak overshoot in meters and residual lane error. Only tune gains after resolving command sign, radians, geometry, joint mapping and tracking. Rerun the same scene and schedule after an edit.
