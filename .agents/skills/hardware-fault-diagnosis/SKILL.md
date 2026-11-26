---
name: hardware-fault-diagnosis
description: Trace physical Niti Vahan robot symptoms such as steering pull or asymmetric travel through the actual board pinout, wiring, firmware, serial commands, and independent measurements.
---

# Hardware fault diagnosis

Establish the exact robot revision, firmware build, symptom, and conditions that reproduce it. Look in the project and the user's supplied files for the board pinout, as-built wiring or clear connector photos, code defining the output pins and calibration, and a timestamped serial capture. Match connector orientation, pin labels and firmware symbols using those artifacts. If an essential observation is missing, identify the smallest missing item that can distinguish the remaining causes.

For a robot that pulls left despite tracking straight in simulation:

1. Trace one observed event through desired steering angle, post-calibration command, output channel or pin, actuator and independently measured wheel angle. Cite the exact firmware location, schematic net or connector and log timestamp when known. Preserve original units and distinguish a PWM command or echoed setpoint from a physical angle.
2. Align command and feedback by timestamp or by a shared neutral/left/right transition. Document any measured clock offset, angle zero and sign convention. Compare steady samples separately from transient response. If there is no angle sensor, a marked wheel photographed against a fixed reference can give an independent angle after calibration.
3. Find the first disagreement. A nonzero request during a straight command points upstream to sensing or control. A centered request followed by an offset output after conversion points to calibration, inversion or clipping. A correct electrical command followed by an offset wheel shifts the investigation toward the actuator or linkage. Straight wheels with a persistent pull warrant a left/right drive-speed or slip comparison. Treat each as a hypothesis until the next observation distinguishes it.
4. Report the evidence, remaining alternatives and one discriminating measurement, with the two predicted outcomes. Use actual pin names and ratings from the board documentation for any proposed measurement. Keep firmware and wiring changes scoped to a verified finding, then replay the same input sequence and compare the before/after observations.

Use the simulation steering trace as a reference command when the task, units and conventions match. Keep physical observations, simulator joint feedback, and requested angles clearly identified.
