---
name: camera-lane-debugger
description: Replay Niti Vahan Task 1C camera clips, inspect annotated lane detections, localize the first wrong frame, and compare results after detector changes.
---

# Camera and lane detection

Use the environment with OpenCV and NumPy. From the project root run `./test-vehicle vision` for all supplied clips, or append specific clip paths such as `task1c/public/clip_17.mp4`. The replay calls the current detector on original BGR frames in order, resetting temporal state between clips. It writes a fresh folder containing per-frame predictions, clip hashes, counts, and annotated images.

Read `summary.json` and inspect the overlay before, at and after the first flagged frame. Frame indices start at zero. The red mark is the predicted lane center. When supported by the current detector, cyan marks the fitted divider and magenta marks sampled road edges. Check the expected side at the detector's reference row in the original image. Inspect the mask, fitted geometry or temporal fallback only as needed to explain the first divergence.

A label switch or center jump selects a frame for review. Confirm an error from the footage or reviewed labels before changing the detector. If a failure is known elsewhere, replay from frame zero to preserve temporal memory and add the reviewed frame to a CSV with the columns `clip,frame,lane,center_x`. Use the clip filename, original pixel column and task's lane vocabulary. Run with `--labels <csv> --label-tolerance <pixels>` to score those frames and save the first mismatch. State the label coverage and tolerance in the result.

After an edit, pass `--baseline <previous-folder>/predictions.csv` with the same clips and labels. Inspect the first changed output and its neighbors, and check the clip hashes. Rerun the Task 1C unit tests. Report confirmed errors separately from unlabeled changes, with clip, frame, time and evidence.
