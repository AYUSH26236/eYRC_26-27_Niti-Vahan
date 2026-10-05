# Task 1C: Lane Detection - solution notes (eYRC 2026-27, Niti Vahan, Team NV_6236)

`lane_detection.py` reads one 640x480 frame at a time and returns two things:

| Output | Meaning |
|---|---|
| `lane` | `"left"` or `"right"`: which lane the vehicle is in (`"unknown"` if it cannot tell) |
| `center_x` | x-pixel of the centre of that lane (`-1` if unknown) |

It uses only OpenCV and NumPy. There is no training and no GPU.

## Results

| Set | Score | Lane | Centre |
|---|---|---|---|
| Public clips (autoeval) | 48.12 / 50 | 18.96 / 20 | 29.16 / 30 |
| Private clips (portal) | **49.24 / 50** | 19.87 / 20 | 29.37 / 30 |

- Mean centre error on the private set: 8.6 px.
- Speed: about 10-13 s for 4000 frames on a MacBook Air. The evaluator limit is 60 s.

## How it works

Each frame goes through five steps.

1. **Colour masks.** HSV thresholds pick out the white dashed divider, the yellow road edges and the dark asphalt. The vehicle's own white chassis (a fixed box at the bottom of the frame) is blanked out.
2. **Find the dashes.** A white blob counts as a divider dash only if it has a dash-like size and is surrounded by dark asphalt. Building walls, sky and kerbs fail this test.
3. **Fit the divider.** One polynomial `x = f(y)` is fitted through all dash pixels.
   - Dashes far from the line are dropped (at most two).
   - A single tiny dash far from the reading row is not trusted.
   - The fit is a curve when three or more dashes span enough of the road, otherwise a straight line.
4. **Read the road edges.** The nearest yellow edge on each side of the divider is sampled in several row bands and fitted with a line.
5. **Decide lane and centre.**
   - The divider's x at row 300 is compared with the frame centre. Divider to the right of the vehicle means the left lane, otherwise the right lane.
   - The lane centre is halfway between the divider and that lane's outer yellow edge.
   - If the outer edge is hidden, the learned lane width is used instead.

**When there is no divider** (a gap between dashes):
1. If both yellow edges are visible and a sensible distance apart, the divider is placed halfway between them.
2. Otherwise the last confident result is reused for at most 5 frames.
3. Otherwise the frame returns `unknown`.

**Between frames.** The code remembers the last confident result and the measured lane width. A sudden large change in the picture (a new scene) resets this memory, so one clip does not leak into the next.

## Settings you can tune

All settings are constants at the top of the implementation block.

| Constant | Value | What it controls |
|---|---|---|
| `_WHITE_HSV_LOW` / `_HIGH` | (0,0,200) / (180,40,255) | what counts as white |
| `_YELLOW_HSV_LOW` / `_HIGH` | (15,100,100) / (35,255,255) | what counts as yellow |
| `_DARK_V_MAX` | 50 | brightest pixel still treated as asphalt |
| `_CAR_BOX` | (225,425,415,480) | the vehicle chassis area that is ignored |
| `_REF_ROW` | 300 | row where every x-position is read |
| `_MIN_ASPHALT_FRACTION` | 0.6 | share of the blob's surroundings that must be dark |
| `_QUAD_MIN_SPAN` | 150 | rows the dashes must cover before a curve is fitted |
| `_MAX_HOLD_FRAMES` | 5 | frames the last result may be reused |
| `_SCENE_CUT_THRESHOLD` | 12.0 | picture change that resets the memory |
| `_RAISE_ERRORS` | `False` | set `True` to see real errors instead of `unknown` |

## Run it

```
# print left / right / unknown counts per public clip
python lane_detection.py public

# official public-set score
eyantra-autoeval evaluate --year 2026 --theme NV --task 1c
```

## Testing notes

Compared with the first version, on the 20 public clips:

| Check | First version | Current |
|---|---|---|
| Lane flips inside a clip | 12 | 4 (one genuine lane change in each of clips 17-20) |
| Frame-to-frame centre jumps over 25 px | 43 | 6 |
| Average frame-to-frame centre movement | 2.4 px | 1.05 px |

- The old code sometimes mistook white building walls for the divider (clip 20). The asphalt check fixes this.
- Centre jumps on clips 10, 13 and 15 came from dashes leaving the scan band. Fitting through all dash pixels fixes this.

## Known limitations

- **Public clip_20:** the evaluator accepts the lane label on only 59 of 137 scored frames. The video shows the same picture as clips the evaluator marks correct, so this could not be fixed in the detector. Private lane accuracy is 99.7%.
- **Curves:** centre error is still 10-20 px on the sharpest bends (clips 13, 18, 19).
- **Lane changes:** the lane label can be off by a few frames around the moment the vehicle crosses the divider.
- **Frame size:** the settings assume 640x480 frames.

## Credits

The idea of fitting a second-order polynomial through lane pixels comes from public "Advanced Lane Lines" projects. No code was copied from them.
