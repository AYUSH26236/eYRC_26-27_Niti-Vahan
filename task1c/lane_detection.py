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
# Functions:        detect_lane, _lane_colour_masks, _find_dashes, _fit_divider,
#                    _collect_edge_samples, _lane_result, _lane_from_road_edges,
#                    _x_at_reference_row, _row_band_clusters,
#                    _DividerFit (class: x_at, confident),
#                    _FrameMemory (class: reset_tracking, start_frame, remember,
#                                  reuse_last_good, update_lane_width)
# Global variables: _WHITE_HSV_LOW, _WHITE_HSV_HIGH, _YELLOW_HSV_LOW,
#                    _YELLOW_HSV_HIGH, _DARK_V_MAX, _CAR_BOX, _SCAN_ROW_START,
#                    _SCAN_ROW_STOP, _SCAN_ROW_STEP, _SCAN_ROW_CENTRES,
#                    _ROW_BAND_HALF_HEIGHT, _REF_ROW, _DASH_MIN_AREA,
#                    _DASH_MAX_AREA, _DASH_MAX_WIDTH, _RING_PX,
#                    _MIN_ASPHALT_FRACTION, _LONE_DASH_MAX_ROW_GAP,
#                    _FIT_MAX_REJECTIONS, _FIT_OUTLIER_PX, _FIT_MAX_PIXELS,
#                    _QUAD_MIN_SPAN, _MAX_EXTRAPOLATION, _CONFIDENT_MIN_DASHES,
#                    _CONFIDENT_MIN_PIXELS, _INITIAL_LANE_WIDTH_PX,
#                    _LANE_WIDTH_MIN, _LANE_WIDTH_MAX, _LANE_WIDTH_SMOOTHING,
#                    _ROAD_SPAN_MIN, _ROAD_SPAN_MAX, _EDGE_FIT_OUTLIER_PX,
#                    _EDGE_FIT_MAX_SLOPE, _MAX_HOLD_FRAMES,
#                    _SCENE_CUT_THRESHOLD, _RAISE_ERRORS, _memory


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
        _memory.start_frame(frame)

        # 1. Colour masks: white divider, yellow road edges, dark asphalt.
        white_mask, yellow_mask, dark_mask = _lane_colour_masks(frame)

        # 2. Divider = the white dashes that sit on asphalt, fitted with one
        #    polynomial x = f(y) through their pixels.
        divider = _fit_divider(_find_dashes(white_mask, dark_mask))

        # 3. Road edges (yellow), read at the same reference row.
        left_pts, right_pts = _collect_edge_samples(yellow_mask, divider)
        left_edge_x = _x_at_reference_row(left_pts)
        right_edge_x = _x_at_reference_row(right_pts)

        if divider is not None:
            result = _lane_result(divider.x_at(_REF_ROW), left_edge_x, right_edge_x, width)
            if divider.confident:
                _memory.remember(result)
        else:
            # No usable divider (dash gap): rebuild it from the road edges,
            # else reuse the last confident result for a few frames.
            result = _lane_from_road_edges(left_edge_x, right_edge_x, width)
            if result is None:
                result = _memory.reuse_last_good()

        center_x, lane = result["center_x"], result["lane"]
    except Exception:
        if _RAISE_ERRORS:
            raise
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
_DARK_V_MAX = 50                     # asphalt is very dark (V about 20)

# ----- The vehicle's own chassis is always white at the bottom of the frame -----
_CAR_BOX = (225, 425, 415, 480)      # x0, y0, x1, y1

# ----- Where to look -----
_SCAN_ROW_START = 300                # first (lowest) band centre row
_SCAN_ROW_STOP = 120                 # bands stop above this row
_SCAN_ROW_STEP = 30                  # distance between band centres
_SCAN_ROW_CENTRES = list(range(_SCAN_ROW_START, _SCAN_ROW_STOP, -_SCAN_ROW_STEP))
_ROW_BAND_HALF_HEIGHT = 15           # half-height (px) of each horizontal band
_REF_ROW = 300                       # every x-position is read off at this row

# ----- Is a white blob a divider dash? -----
_DASH_MIN_AREA = 8                   # px; smaller is noise
_DASH_MAX_AREA = 12000               # px; larger is not a dash
_DASH_MAX_WIDTH = 160                # px
_RING_PX = 10                        # look this far around a blob for asphalt
_MIN_ASPHALT_FRACTION = 0.6          # share of the surrounding ring that must be dark
_LONE_DASH_MAX_ROW_GAP = 60          # a single dash must be this close to the reference row

# ----- Divider polynomial fit -----
_FIT_MAX_REJECTIONS = 2              # dashes dropped as outliers
_FIT_OUTLIER_PX = 30                 # a dash further than this from the line is dropped
_FIT_MAX_PIXELS = 800                # subsample pixels above this many
_QUAD_MIN_SPAN = 150                 # rows the dashes must span before a curve is fitted
_MAX_EXTRAPOLATION = 80              # rows a fit may be extended beyond its data
_CONFIDENT_MIN_DASHES = 2            # a divider fit with this many dashes is trusted ...
_CONFIDENT_MIN_PIXELS = 150          # ... or with this many pixels

# ----- Geometry sanity limits (px, 640x480 frame) -----
_INITIAL_LANE_WIDTH_PX = 315         # starting lane width until measured
_LANE_WIDTH_MIN = 220                # accepted range for a measured lane width
_LANE_WIDTH_MAX = 420
_LANE_WIDTH_SMOOTHING = 0.2          # weight of each new lane-width measurement
_ROAD_SPAN_MIN = 450                 # plausible distance between both yellow edges
_ROAD_SPAN_MAX = 800
_EDGE_FIT_OUTLIER_PX = 20            # road-edge samples further than this are dropped
_EDGE_FIT_MAX_SLOPE = 1.5            # steeper edge fits are rejected as unreliable

# ----- Between-frame behaviour -----
_MAX_HOLD_FRAMES = 5                 # reuse last confident result for at most this many frames
_SCENE_CUT_THRESHOLD = 12.0          # mean pixel change (16x12 thumbnail) that means a new scene

# ----- Debugging -----
_RAISE_ERRORS = False                # True = let errors surface instead of returning "unknown"


class _DividerFit:
    '''The divider as a polynomial x = f(y), plus how much evidence supports it.'''

    def __init__(self, coeffs, n_dashes, n_pixels):
        self.coeffs = coeffs
        self.n_dashes = n_dashes
        self.n_pixels = n_pixels

    def x_at(self, row):
        return float(np.polyval(self.coeffs, row))

    @property
    def confident(self):
        return (self.n_dashes >= _CONFIDENT_MIN_DASHES
                or self.n_pixels >= _CONFIDENT_MIN_PIXELS)


class _FrameMemory:
    '''
    Small amount of state carried between frames (frames are processed in
    order): the last confident result, how long it has been reused, the lane
    width measured so far, and a thumbnail used to notice a change of scene.
    '''

    def __init__(self):
        self.lane_width = float(_INITIAL_LANE_WIDTH_PX)
        self.reset_tracking()
        self._thumbnail = None

    def reset_tracking(self):
        self.last_good = None
        self.hold_count = 0

    def start_frame(self, frame):
        '''Forget the last result if this frame is a different scene (e.g. a new clip).'''
        thumbnail = cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA).astype(np.int16)
        if (self._thumbnail is not None
                and np.abs(thumbnail - self._thumbnail).mean() > _SCENE_CUT_THRESHOLD):
            self.reset_tracking()
        self._thumbnail = thumbnail

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


def _lane_colour_masks(frame):
    '''
    Purpose:
    ---
    Threshold `frame` for the three things the pipeline needs: the white
    divider, the yellow road edges and the dark asphalt. The vehicle's own
    white chassis is blanked out of the white mask.

    Returns:
    ---
    `white_mask`, `yellow_mask`, `dark_mask` :  [ numpy.ndarray ]
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, _WHITE_HSV_LOW, _WHITE_HSV_HIGH)
    yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LOW, _YELLOW_HSV_HIGH)
    dark_mask = hsv[:, :, 2] < _DARK_V_MAX

    x0, y0, x1, y1 = _CAR_BOX
    white_mask[y0:y1, x0:x1] = 0
    return white_mask, yellow_mask, dark_mask


def _find_dashes(white_mask, dark_mask):
    '''
    Purpose:
    ---
    Find white blobs that really are divider dashes. A dash has a sensible
    size and lies ON the road, so the pixels around it are dark asphalt;
    building walls, sky and kerbs fail that test.

    Returns:
    ---
    `dashes` :  [ list of (centre_row, centre_col, area, rows, cols) ]
    '''
    height, width = white_mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (white_mask > 0).astype(np.uint8), connectivity=8)

    dashes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < _DASH_MIN_AREA or area > _DASH_MAX_AREA or w > _DASH_MAX_WIDTH:
            continue

        x0, x1 = max(0, x - _RING_PX), min(width, x + w + _RING_PX)
        y0, y1 = max(0, y - _RING_PX), min(height, y + h + _RING_PX)
        window = labels[y0:y1, x0:x1]
        surroundings = window == 0
        n_surroundings = np.count_nonzero(surroundings)
        if n_surroundings == 0:
            continue
        dark_share = np.count_nonzero(dark_mask[y0:y1, x0:x1] & surroundings) / n_surroundings
        if dark_share < _MIN_ASPHALT_FRACTION:
            continue

        rows, cols = np.nonzero(window == label)
        dashes.append((centroids[label][1], centroids[label][0], area, rows + y0, cols + x0))
    return dashes


def _fit_divider(dashes):
    '''
    Purpose:
    ---
    Fit x = f(y) through the pixels of the dashes: a straight line normally,
    a second-order curve when the dashes span enough of the road to show a
    bend. Dashes far from the line are dropped first. One tiny far-away
    dash is not trusted.

    Returns:
    ---
    `divider` :  [ _DividerFit or None ]
    '''
    kept = list(dashes)
    if not kept:
        return None

    for _ in range(_FIT_MAX_REJECTIONS):
        if len(kept) < 3:
            break
        rows = np.array([d[0] for d in kept])
        cols = np.array([d[1] for d in kept])
        weights = np.sqrt([d[2] for d in kept])
        slope, intercept = np.polyfit(rows, cols, 1, w=weights)
        residuals = np.abs(cols - (slope * rows + intercept))
        worst = int(np.argmax(residuals))
        if residuals[worst] <= _FIT_OUTLIER_PX:
            break
        del kept[worst]

    if len(kept) == 1 and abs(kept[0][0] - _REF_ROW) > _LONE_DASH_MAX_ROW_GAP:
        return None

    ys = np.concatenate([d[3] for d in kept]).astype(np.float64)
    xs = np.concatenate([d[4] for d in kept]).astype(np.float64)
    if len(ys) > _FIT_MAX_PIXELS:
        step = len(ys) // _FIT_MAX_PIXELS + 1
        ys, xs = ys[::step], xs[::step]

    span = ys.max() - ys.min()
    outside = max(ys.min() - _REF_ROW, _REF_ROW - ys.max(), 0.0)
    degree = 2 if (len(kept) >= 3 and span >= _QUAD_MIN_SPAN
                   and outside <= _MAX_EXTRAPOLATION) else 1
    if span < 3:
        coeffs = np.array([0.0, float(xs.mean())])   # a vertical sliver: no slope to fit
    else:
        try:
            coeffs = np.polyfit(ys, xs, degree)
        except (np.linalg.LinAlgError, ValueError):
            return None
    return _DividerFit(coeffs, len(kept), int(len(ys)))


def _collect_edge_samples(yellow_mask, divider):
    '''
    Purpose:
    ---
    Scan several horizontal bands and collect (row, x) samples of the yellow
    road edge on each side of the divider. With no divider, the outermost
    two yellow marks in a band are taken as the two road edges.

    Returns:
    ---
    `left_pts`, `right_pts` :  [ list of (row, x) ]
    '''
    left_pts, right_pts = [], []

    for y_center in _SCAN_ROW_CENTRES:
        if y_center - _ROW_BAND_HALF_HEIGHT < 0:
            break
        clusters = _row_band_clusters(yellow_mask, y_center, _ROW_BAND_HALF_HEIGHT)

        if divider is not None:
            divider_x = divider.x_at(y_center)
            lefts = [x for x in clusters if x < divider_x]
            rights = [x for x in clusters if x > divider_x]
            if lefts:
                left_pts.append((y_center, max(lefts)))
            if rights:
                right_pts.append((y_center, min(rights)))
        elif len(clusters) >= 2:
            left_pts.append((y_center, min(clusters)))
            right_pts.append((y_center, max(clusters)))

    return left_pts, right_pts


def _lane_result(divider_x, left_edge_x, right_edge_x, frame_width):
    '''
    Purpose:
    ---
    The vehicle is in the left lane when the divider is to its right, and in
    the right lane otherwise. The lane centre is halfway between the divider
    and that lane's outer edge - the matching yellow line, or, if it is not
    visible, the divider shifted by the learned lane width.

    Returns:
    ---
    `result` :  [ dict ]  {"center_x": int, "lane": str}
    '''
    vehicle_x = frame_width / 2.0
    _memory.update_lane_width(left_edge_x, right_edge_x)

    if divider_x > vehicle_x:
        lane, outer_x = LANE_LEFT, left_edge_x
    else:
        lane, outer_x = LANE_RIGHT, right_edge_x

    if outer_x is None:
        sign = -1.0 if lane == LANE_LEFT else 1.0
        outer_x = divider_x + sign * _memory.lane_width

    center_x = int(round((divider_x + outer_x) / 2.0))
    return {"center_x": max(0, min(frame_width - 1, center_x)), "lane": lane}


def _lane_from_road_edges(left_edge_x, right_edge_x, frame_width):
    '''
    Purpose:
    ---
    With no divider visible, the divider sits halfway between the two road
    edges. Use that - but only if both edges are seen and are a plausible
    distance apart.

    Returns:
    ---
    `result` :  [ dict or None ]
    '''
    if left_edge_x is None or right_edge_x is None:
        return None
    if not _ROAD_SPAN_MIN <= right_edge_x - left_edge_x <= _ROAD_SPAN_MAX:
        return None
    return _lane_result((left_edge_x + right_edge_x) / 2.0, left_edge_x, right_edge_x, frame_width)


def _x_at_reference_row(points):
    '''
    Purpose:
    ---
    Given (row, x) samples of one road edge, return its x at _REF_ROW. One
    sample is used as is; several are fitted with a line (one outlier
    dropped).

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
            if residuals[worst] > _EDGE_FIT_OUTLIER_PX:
                keep = np.arange(len(xs)) != worst
                slope, intercept = np.polyfit(ys[keep], xs[keep], 1)
        if abs(slope) > _EDGE_FIT_MAX_SLOPE:
            return nearest
        return float(slope * _REF_ROW + intercept)
    except (np.linalg.LinAlgError, ValueError):
        return nearest


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