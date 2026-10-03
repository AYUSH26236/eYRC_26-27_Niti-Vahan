'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < NV_6236 >
# Author List:      < Ayush Tiwari, Anushka Telore , Atharva Jadhav , Utkarsh Singh >
# Filename:         lane_detection.py
# Functions:        detect_lane, _collect_marking_samples,
#                    _pick_lane_and_outer_edge, _lane_centre,
#                    _x_at_reference_row, _lane_colour_masks,
#                    _remove_border_touching, _roi_mask, _row_band_clusters,
#                    _FrameMemory (class: remember, reuse_last_good, update_lane_width)
# Global variables: _WHITE_HSV_LOW, _WHITE_HSV_HIGH, _YELLOW_HSV_LOW,
#                    _YELLOW_HSV_HIGH, _SCAN_ROW_START, _SCAN_ROW_STOP,
#                    _SCAN_ROW_STEP, _SCAN_ROW_CENTRES, _ROW_BAND_HALF_HEIGHT,
#                    _REF_ROW, _INITIAL_LANE_WIDTH_PX, _LANE_WIDTH_MIN,
#                    _LANE_WIDTH_MAX, _LANE_WIDTH_SMOOTHING, _ROAD_SPAN_MIN,
#                    _ROAD_SPAN_MAX, _BORDER_KEEP_MAX_WIDTH, _FIT_OUTLIER_PX,
#                    _FIT_MAX_SLOPE, _CENTRE_DEADZONE_PX, _MAX_HOLD_FRAMES,
#                    _USE_ROI_MASK, _ROI_BOTTOM_WIDTH, _ROI_TOP_WIDTH, _ROI_TOP_ROW,
#                    _roi_cache, _OFFSET_LEFT_LANE_PX, _OFFSET_RIGHT_LANE_PX, _memory


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        A single BGR frame read from the video, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "center_x" : int,   x-pixel of the lane centre in this frame,
                                or -1 if the lane could not be found
            "lane"     : str,   "left", "right" or "unknown"
        }

    Example call:
    ---
    result = detect_lane(frame)

    COORDINATE SYSTEM:
    ---
    `center_x` is an absolute pixel column in the frame AS RECEIVED - the
    dataset's own resolution, 640x480. It is compared against a ground truth
    measured in those pixels, so it only means anything in them.

    You may resize, crop or warp all you like inside this function, but scale
    the answer back before returning it. A centre found in a 320x240 copy is
    half the value it should be, and a centre read off a bird's-eye view is in
    warped coordinates, not frame ones - map the point back through the inverse
    of your transform. Do not re-encode or resize the clip files themselves.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.
    '''

    center_x = -1
    lane = LANE_UNKNOWN

    #################### ADD YOUR CODE HERE ####################
    try:
        width = frame.shape[1]
        vehicle_x = width / 2.0

        # 1. Find the coloured markings (white divider, yellow road edges).
        white_mask, yellow_mask = _lane_colour_masks(frame)

        # 2. Sample them in several row bands and read each marking's x at
        #    one fixed reference row.
        divider_pts, left_pts, right_pts = _collect_marking_samples(
            white_mask, yellow_mask, vehicle_x)
        divider_x = _x_at_reference_row(divider_pts)
        left_edge_x = _x_at_reference_row(left_pts)
        right_edge_x = _x_at_reference_row(right_pts)

        # 3. No divider (dash gap): reuse the last good answer briefly.
        if divider_x is None:
            return _memory.reuse_last_good()

        if abs(divider_x - vehicle_x) < _CENTRE_DEADZONE_PX:
            return {"center_x": -1, "lane": LANE_UNKNOWN}

        # 4. Decide the lane and compute its centre.
        _memory.update_lane_width(left_edge_x, right_edge_x)
        lane, outer_x = _pick_lane_and_outer_edge(
            divider_x, vehicle_x, left_edge_x, right_edge_x)
        center_x = _lane_centre(divider_x, outer_x, lane, width)

        _memory.remember({"center_x": center_x, "lane": lane})
    except Exception:
        center_x = -1
        lane = LANE_UNKNOWN
    ############################################################

    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------

# ----- Colour thresholds (OpenCV HSV: H 0-180, S 0-255, V 0-255) -----
_WHITE_HSV_LOW = (0, 0, 200)         # dashed centre divider: bright, unsaturated
_WHITE_HSV_HIGH = (180, 40, 255)
_YELLOW_HSV_LOW = (15, 100, 100)     # solid road edges
_YELLOW_HSV_HIGH = (35, 255, 255)

# ----- Where to look -----
_SCAN_ROW_START = 300                # first (lowest) band centre row
_SCAN_ROW_STOP = 120                 # bands stop above this row
_SCAN_ROW_STEP = 30                  # distance between band centres
_SCAN_ROW_CENTRES = list(range(_SCAN_ROW_START, _SCAN_ROW_STOP, -_SCAN_ROW_STEP))
_ROW_BAND_HALF_HEIGHT = 15           # half-height (px) of each horizontal band
_REF_ROW = 300                       # every x-position is read off at this row

# ----- Geometry sanity limits (px, 640x480 frame) -----
_INITIAL_LANE_WIDTH_PX = 315         # starting lane width until measured
_LANE_WIDTH_MIN = 220                # accepted range for a measured lane width
_LANE_WIDTH_MAX = 420
_LANE_WIDTH_SMOOTHING = 0.2          # weight of each new lane-width measurement
_ROAD_SPAN_MIN = 450                 # plausible distance between both yellow edges
_ROAD_SPAN_MAX = 800
_BORDER_KEEP_MAX_WIDTH = 50          # top/bottom-touching blob this narrow = divider dash

# ----- Line fitting -----
_FIT_OUTLIER_PX = 20                 # drop one sample further than this from the fit
_FIT_MAX_SLOPE = 1.5                 # steeper fits are rejected as unreliable

# ----- Optional trapezoid region of interest (white divider only) -----
_USE_ROI_MASK = True                # False = behaviour unchanged
_ROI_BOTTOM_WIDTH = 1.00             # trapezoid width at the bottom row, fraction of frame width
_ROI_TOP_WIDTH = 0.70                # trapezoid width at its top row, fraction of frame width
_ROI_TOP_ROW = 100                   # rows above this are ignored (scan bands start near 105)

# ----- Decision tuning -----
_CENTRE_DEADZONE_PX = 0              # divider this close to vehicle x -> unknown
_MAX_HOLD_FRAMES = 5                 # reuse last good result for at most this many frames
_OFFSET_LEFT_LANE_PX = 0             # bias correction added to center_x, left lane
_OFFSET_RIGHT_LANE_PX = 0            # bias correction added to center_x, right lane


class _FrameMemory:
    '''
    Small amount of state carried between frames (frames are processed in
    order): the last good result, how long it has been reused, and the lane
    width measured so far.
    '''

    def __init__(self):
        self.last_good = None
        self.hold_count = 0
        self.lane_width = float(_INITIAL_LANE_WIDTH_PX)

    def remember(self, result):
        self.last_good = dict(result)
        self.hold_count = 0

    def reuse_last_good(self):
        '''Return the last good result (up to _MAX_HOLD_FRAMES times), else "unknown".'''
        if self.last_good is not None and self.hold_count < _MAX_HOLD_FRAMES:
            self.hold_count += 1
            return dict(self.last_good)
        return {"center_x": -1, "lane": LANE_UNKNOWN}

    def update_lane_width(self, left_edge_x, right_edge_x):
        '''Learn the real lane width whenever both yellow edges are visible.'''
        if left_edge_x is None or right_edge_x is None:
            return
        measured = (right_edge_x - left_edge_x) / 2.0
        if _LANE_WIDTH_MIN <= measured <= _LANE_WIDTH_MAX:
            self.lane_width = ((1.0 - _LANE_WIDTH_SMOOTHING) * self.lane_width
                               + _LANE_WIDTH_SMOOTHING * measured)


_memory = _FrameMemory()


def _collect_marking_samples(white_mask, yellow_mask, vehicle_x):
    '''
    Purpose:
    ---
    Scan several horizontal bands and collect (row, x) samples of the
    divider and of the nearest yellow edge on each side of it. Using many
    bands means dash gaps in the divider do not break detection.

    Input Arguments:
    ---
    `white_mask`, `yellow_mask` :  [ numpy.ndarray ]  binary masks
    `vehicle_x` :  [ float ]  x-pixel of the vehicle (frame centre)

    Returns:
    ---
    `divider_pts`, `left_pts`, `right_pts` :  [ list of (row, x) ]
    '''
    divider_pts, left_pts, right_pts = [], [], []

    for y_center in _SCAN_ROW_CENTRES:
        if y_center - _ROW_BAND_HALF_HEIGHT < 0:
            break

        white_clusters = _row_band_clusters(white_mask, y_center, _ROW_BAND_HALF_HEIGHT)
        if not white_clusters:
            continue
        yellow_clusters = _row_band_clusters(yellow_mask, y_center, _ROW_BAND_HALF_HEIGHT)

        # If both yellow road edges are visible, the divider must sit about
        # halfway between them - use that to reject false white blobs.
        target_x = vehicle_x
        if len(yellow_clusters) >= 2:
            span = max(yellow_clusters) - min(yellow_clusters)
            if _ROAD_SPAN_MIN <= span <= _ROAD_SPAN_MAX:
                target_x = (max(yellow_clusters) + min(yellow_clusters)) / 2.0

        divider = min(white_clusters, key=lambda x: abs(x - target_x))
        divider_pts.append((y_center, divider))

        lefts = [x for x in yellow_clusters if x < divider]
        rights = [x for x in yellow_clusters if x > divider]
        if lefts:
            left_pts.append((y_center, max(lefts)))
        if rights:
            right_pts.append((y_center, min(rights)))

    return divider_pts, left_pts, right_pts


def _pick_lane_and_outer_edge(divider_x, vehicle_x, left_edge_x, right_edge_x):
    '''
    Purpose:
    ---
    The vehicle is in the left lane when the divider is to its right, and in
    the right lane otherwise. The outer edge of that lane is the matching
    yellow line, or - if it is not visible - the divider shifted by the
    learned lane width.

    Returns:
    ---
    `lane`, `outer_x` :  [ str, float ]
    '''
    if divider_x > vehicle_x:
        lane, outer_x = LANE_LEFT, left_edge_x
    else:
        lane, outer_x = LANE_RIGHT, right_edge_x

    if outer_x is None:
        sign = -1.0 if lane == LANE_LEFT else 1.0
        outer_x = divider_x + sign * _memory.lane_width
    return lane, outer_x


def _lane_centre(divider_x, outer_x, lane, frame_width):
    '''Midpoint of the lane, plus the per-lane bias correction, clamped to the frame.'''
    offset = _OFFSET_LEFT_LANE_PX if lane == LANE_LEFT else _OFFSET_RIGHT_LANE_PX
    center_x = int(round((divider_x + outer_x) / 2.0)) + offset
    return max(0, min(frame_width - 1, center_x))


def _x_at_reference_row(points):
    '''
    Purpose:
    ---
    Given (row, x) samples of one marking, return its x at _REF_ROW. One
    sample is used as is; several are fitted with a line (one outlier
    dropped) so dash gaps and perspective drift do not move the answer.

    Input Arguments:
    ---
    `points` :  [ list of (row, x) ]

    Returns:
    ---
    `x` :  [ float or None ]  None when there are no samples
    '''
    if not points:
        return None
    if len(points) == 1:
        return float(points[0][1])

    ys = np.array([p[0] for p in points], dtype=np.float64)
    xs = np.array([p[1] for p in points], dtype=np.float64)
    nearest = float(xs[np.argmin(np.abs(ys - _REF_ROW))])

    try:
        slope, intercept = np.polyfit(ys, xs, 1)
        if len(points) >= 3:
            residuals = np.abs(xs - (slope * ys + intercept))
            worst = int(np.argmax(residuals))
            if residuals[worst] > _FIT_OUTLIER_PX:
                keep = np.arange(len(xs)) != worst
                slope, intercept = np.polyfit(ys[keep], xs[keep], 1)
        if abs(slope) > _FIT_MAX_SLOPE:
            return nearest
        return float(slope * _REF_ROW + intercept)
    except Exception:
        return nearest


def _lane_colour_masks(frame):
    '''
    Purpose:
    ---
    Threshold `frame` for the track's two marking colours - solid yellow
    edges and dashed white centre divider - and remove render artefacts
    that would otherwise be mistaken for the divider.

    Returns:
    ---
    `white_mask`, `yellow_mask` :  [ numpy.ndarray ]  binary masks
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, _WHITE_HSV_LOW, _WHITE_HSV_HIGH)
    yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LOW, _YELLOW_HSV_HIGH)

    white_mask = _remove_border_touching(white_mask)
    if _USE_ROI_MASK:
        white_mask = cv2.bitwise_and(white_mask, _roi_mask(white_mask.shape))
    return white_mask, yellow_mask


_roi_cache = {}


def _roi_mask(shape):
    '''
    Purpose:
    ---
    Trapezoid region of interest: wide at the bottom of the frame, narrower
    at _ROI_TOP_ROW, because lane markings converge with distance. White
    blobs outside it (sky, side clutter) are ignored. The mask is built once
    per frame size and cached.

    Input Arguments:
    ---
    `shape` :  [ tuple ]  (height, width) of the mask

    Returns:
    ---
    `mask` :  [ numpy.ndarray ]  uint8, 255 inside the trapezoid
    '''
    if shape not in _roi_cache:
        height, width = shape[:2]
        bottom_half = _ROI_BOTTOM_WIDTH * width / 2.0
        top_half = _ROI_TOP_WIDTH * width / 2.0
        mid = width / 2.0
        polygon = np.array([[
            (mid - bottom_half, height - 1),
            (mid - top_half, _ROI_TOP_ROW),
            (mid + top_half, _ROI_TOP_ROW),
            (mid + bottom_half, height - 1),
        ]], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 255)
        _roi_cache[shape] = mask
    return _roi_cache[shape]


def _remove_border_touching(mask):
    '''
    Purpose:
    ---
    Remove white blobs that touch the frame edge (vehicle chassis/HUD, curb
    artefacts) - EXCEPT narrow blobs touching only the top or bottom edge,
    which are divider dashes running off the frame and must be kept.
    '''
    height, width = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n <= 1:
        return mask

    # Vectorised: decide per label, then clear every removed label in one
    # pass (a per-blob full-frame comparison is too slow over thousands of frames).
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    touches_side = (x <= 0) | (x + w >= width)
    touches_top_or_bottom = (y <= 0) | (y + h >= height)
    is_divider_dash = touches_top_or_bottom & ~touches_side & (w <= _BORDER_KEEP_MAX_WIDTH)
    remove = (touches_side | touches_top_or_bottom) & ~is_divider_dash
    remove[0] = False  # label 0 is the background
    if not remove.any():
        return mask

    cleaned = mask.copy()
    cleaned[remove[labels]] = 0
    return cleaned


def _row_band_clusters(mask, y_center, half_height):
    '''
    Purpose:
    ---
    Sum a binary mask over a horizontal band of rows and collapse the
    result into one weighted-centroid x-position per contiguous run of
    lit columns - each run is one candidate marking in that band.

    Returns:
    ---
    `clusters` :  [ list of float ]  x-positions, left to right
    '''
    height, _ = mask.shape
    y_low = max(0, y_center - half_height)
    y_high = min(height, y_center + half_height)

    column_counts = np.count_nonzero(mask[y_low:y_high, :], axis=0).astype(np.float32)

    # Vectorised run detection (a per-column Python loop is too slow when
    # several bands are scanned on every frame).
    lit = np.concatenate(([0], (column_counts > 0).astype(np.int8), [0]))
    edges = np.diff(lit)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    clusters = []
    for start, end in zip(starts, ends):
        weights = column_counts[start:end]
        total = float(weights.sum())
        if total > 0:
            clusters.append(float((np.arange(start, end) * weights).sum() / total))
    return clusters


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()