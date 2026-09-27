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
# Functions:        detect_lane, _lane_colour_masks, _remove_border_touching,
#                    _row_band_clusters
# Global variables: _LANE_WIDTH_PX_FRAME, _ROW_BAND_HALF_HEIGHT,
#                    _CENTRE_DEADZONE_PX, _SCAN_ROW_CENTRES


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

    This implementation works directly in the frame's own pixel grid (no
    resizing/warping), so nothing needs to be scaled or mapped back before
    returning.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.

    METHOD (see helpers below for details):
    ---
    The track has a solid YELLOW line on each outer edge and a dashed WHITE
    line down the centre. Colour alone tells the two apart, so there is no
    need to guess which detected line is "the dashed one" - the white mask
    only ever contains the divider (once the vehicle's own on-screen model
    and a couple of border-hugging render artefacts are stripped out).

    1. Threshold the frame for white (divider) and yellow (edges).
    2. Strip any white blob touching the frame border - this removes the
       vehicle's own chassis/HUD, which is rendered at a fixed screen
       position and would otherwise look like a lane marking.
    3. Scan horizontal bands from just above the chassis upward; the first
       band with a divider pixel becomes the reference row (closest to the
       vehicle that isn't blocked by its own rendered body).
    4. Compare the divider's x position to the vehicle's (frame centre) x
       position to decide "left"/"right", then take the midpoint of the
       divider and the nearest yellow edge on that side as the lane centre.
    '''

    center_x = -1
    lane = LANE_UNKNOWN

    #################### ADD YOUR CODE HERE ####################
    try:
        height, width = frame.shape[:2]
        vehicle_x = width / 2.0

        white_mask, yellow_mask = _lane_colour_masks(frame)

        divider_x = None
        left_edge_x = None
        right_edge_x = None

        for y_center in _SCAN_ROW_CENTRES:
            if y_center - _ROW_BAND_HALF_HEIGHT < 0:
                break

            white_clusters = _row_band_clusters(white_mask, y_center, _ROW_BAND_HALF_HEIGHT)
            if not white_clusters:
                continue

            # Several small artefacts can occasionally survive the cleanup;
            # the real divider is the one closest to the vehicle's own
            # position, since it runs down the middle of the track.
            divider_x = min(white_clusters, key=lambda x: abs(x - vehicle_x))

            yellow_clusters = _row_band_clusters(yellow_mask, y_center, _ROW_BAND_HALF_HEIGHT)
            left_candidates = [x for x in yellow_clusters if x < divider_x]
            right_candidates = [x for x in yellow_clusters if x > divider_x]
            left_edge_x = max(left_candidates) if left_candidates else None
            right_edge_x = min(right_candidates) if right_candidates else None
            break

        if divider_x is None:
            return {"center_x": -1, "lane": LANE_UNKNOWN}

        # Divider is (numerically) right where the vehicle is - too
        # ambiguous to call a side confidently.
        if abs(divider_x - vehicle_x) < _CENTRE_DEADZONE_PX:
            return {"center_x": -1, "lane": LANE_UNKNOWN}

        if divider_x > vehicle_x:
            # Divider to the right of the vehicle -> left lane.
            lane = LANE_LEFT
            outer_x = left_edge_x
        else:
            # Divider to the left of the vehicle -> right lane.
            lane = LANE_RIGHT
            outer_x = right_edge_x

        if outer_x is None:
            # Outer edge wasn't detected on this frame (e.g. it left the
            # frame on a curve) - fall back on the approximate lane width
            # stated in the task spec.
            outer_x = (divider_x - _LANE_WIDTH_PX_FRAME if lane == LANE_LEFT
                       else divider_x + _LANE_WIDTH_PX_FRAME)

        center_x = int(round((divider_x + outer_x) / 2.0))
        center_x = max(0, min(width - 1, center_x))
    except Exception:
        # Any unexpected failure is reported honestly rather than guessed.
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

_LANE_WIDTH_PX_FRAME = 315   # approx. lane width in frame pixels (from task spec) - fallback only
_ROW_BAND_HALF_HEIGHT = 15   # half-height (px) of each horizontal scan band
_CENTRE_DEADZONE_PX = 3      # divider within this many px of vehicle x -> unknown
_MIN_CLUSTER_WEIGHT = 15     # a real dash/edge spans many columns and rows; a
                             # single anti-aliasing pixel at a colour boundary
                             # does not - this discards the latter
# Scan bands from just above the vehicle's own on-screen model upward.
_SCAN_ROW_CENTRES = list(range(465, 130, -30))


def _lane_colour_masks(frame):
    '''
    Threshold `frame` for the track's two marking colours - solid yellow
    edges and dashed white centre divider - and remove render artefacts
    that would otherwise be mistaken for the divider.
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))

    # The vehicle's own chassis/HUD (light grey/white, fixed screen
    # position) and a couple of thin render artefacts along the track's
    # curb all touch a frame border; the dashed divider never does. This
    # is a cheap, robust way to drop all of them at once.
    white_mask = _remove_border_touching(white_mask)

    return white_mask, yellow_mask


def _remove_border_touching(mask):
    '''
    Remove every connected white blob that touches any edge of the frame.
    '''
    height, width = mask.shape
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))[:2]
    if num_labels <= 1:
        return mask

    border_labels = set(np.unique(labels[0, :]))
    border_labels |= set(np.unique(labels[-1, :]))
    border_labels |= set(np.unique(labels[:, 0]))
    border_labels |= set(np.unique(labels[:, -1]))
    border_labels.discard(0)

    if not border_labels:
        return mask

    cleaned = mask.copy()
    cleaned[np.isin(labels, list(border_labels))] = 0
    return cleaned


def _row_band_clusters(mask, y_center, half_height):
    '''
    Sum a binary mask over a horizontal band of rows and collapse the
    result into one weighted-centroid x-position per contiguous run of
    lit columns - each run is one candidate marking in that band.

    Runs lighter than `_MIN_CLUSTER_WEIGHT` are discarded: a real dash or
    edge segment spans many columns and rows within the band, whereas a
    single anti-aliasing pixel at a colour boundary (e.g. where the yellow
    edge meets the black road) only ever lights up a column or two. This
    keeps such artefacts from ever being mistaken for the divider.
    '''
    height, width = mask.shape
    y_low = max(0, y_center - half_height)
    y_high = min(height, y_center + half_height)

    column_counts = np.sum(mask[y_low:y_high, :] > 0, axis=0).astype(np.float32)

    clusters = []
    in_run = False
    run_start = 0
    for x in range(width + 1):
        value = column_counts[x] if x < width else 0.0
        above = value > 0
        if above and not in_run:
            in_run = True
            run_start = x
        elif not above and in_run:
            in_run = False
            xs = np.arange(run_start, x)
            weights = column_counts[run_start:x]
            total = weights.sum()
            if total >= _MIN_CLUSTER_WEIGHT:
                clusters.append(float((xs * weights).sum() / total))

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