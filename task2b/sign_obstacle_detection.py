'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
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
# Author List:      < Ayush Tiwari, Anushka Telore, Atharva Jadhav, Utkarsh Singh >
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle, _red_mask, _red_fill, _rectify
# Global variables: None


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    '''
    Purpose:
    ---
    Detect every road sign visible in a single image, and report what each
    one is and how large it appears. An image can hold zero, one, or more
    than one sign at once.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `detections` :  [ list of dict ]
        Zero, one, or many entries - one per sign found in the image:
        [
            {
                "label" : str,   one of SIGN_LABELS
                "area"  : int,   that sign's area in pixels
            },
            ...
        ]

    Example call:
    ---
    detections = detect_signs(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    detections = []

    #################### ADD YOUR CODE HERE ####################
    if image is None or image.size == 0:
        return detections

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    frame_area = height * width
    shapes = []   # (x, y, w, h, label, area)

    # ---- Red-ring signs: circle = Speed40, triangle = School ----------------
    red = _red_mask(hsv)
    red_contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_floor = max(220.0, frame_area * 0.00030)
    for contour in red_contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < red_floor:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < max(14, int(min(height, width) * 0.022)):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        # A sign is a red RING (white inside); a solid red blob is the obstacle.
        if _red_fill(red, contour) > 0.70:
            continue
        polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
        aspect = w / float(h)
        if 0.60 <= aspect <= 1.40 and circularity >= 0.66 and len(polygon) >= 5:
            # circle: mean of bounding-box and filled-contour area
            shapes.append((x, y, w, h, "Speed40", 0.5 * (w * h + contour_area)))
        elif len(polygon) == 3 and 0.48 <= aspect <= 1.35:
            shapes.append((x, y, w, h, "School", contour_area))

    # ---- Blue square signs: Parking / Hospital -----------------------------
    blue = cv2.inRange(hsv, np.array([88, 75, 45], dtype=np.uint8),
                       np.array([140, 255, 255], dtype=np.uint8))
    blue_contours, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        blue_floor = max(120.0, frame_area * 0.00018)
        for index, contour in enumerate(blue_contours):
            if hierarchy[index][3] != -1:
                continue
            contour_area = cv2.contourArea(contour)
            if contour_area < blue_floor:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < max(20, int(min(height, width) * 0.030)):
                continue
            # tilted signs have a taller/wider bounding box, so be tolerant
            if not (0.60 <= w / float(h) <= 1.60):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area <= 0 or contour_area / hull_area < 0.80:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            polygon = cv2.approxPolyDP(contour, 0.055 * perimeter, True)
            if len(polygon) < 4 or len(polygon) > 8:
                continue
            clipped = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1

            # Straighten the sign face into 64x64 so tilt/perspective do not matter.
            if clipped:
                face = cv2.resize(image[y:y + h, x:x + w], (64, 64))
            else:
                face = _rectify(image, contour)
            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            # Letter = bright, unsaturated pixels. Building windows have none.
            glyph = (face_hsv[:, :, 2] >= 115) & (face_hsv[:, :, 1] <= 105)
            inner_fraction = float(glyph[12:52, 12:52].mean())
            if not (0.15 <= inner_fraction <= 0.70):
                continue

            # H has a leg in the lower-right, P has none. Compare the two lower halves.
            lower_left = float(glyph[36:56, 14:30].mean())
            lower_right = float(glyph[36:56, 34:50].mean())
            p_like = lower_right < 0.40 * max(lower_left, 0.05)

            # Second cue: the closed counter of the P is a blue island inside the white letter.
            depth = 0
            child = hierarchy[index][2]
            while child != -1:
                depth = max(depth, 1)
                grand = hierarchy[child][2]
                while grand != -1:
                    depth = 2
                    grand = hierarchy[grand][0]
                child = hierarchy[child][0]
            # A sign cut off by the frame hides its lower-right corner, which
            # looks like a P, so a clipped sign is only called Hospital when
            # its H crossbar/legs are clearly visible.
            label = "Parking" if (p_like or depth >= 2 or clipped) else "Hospital"
            shapes.append((x, y, w, h, label, 0.5 * (w * h + contour_area)))

    # ---- Merge fragments of the same sign, keeping the largest -------------
    accepted = []
    for candidate in sorted(shapes, key=lambda item: item[5], reverse=True):
        x, y, w, h, label, _ = candidate
        cx, cy = x + w / 2.0, y + h / 2.0
        duplicate = False
        for old in accepted:
            ox, oy, ow, oh, old_label, _ = old
            if (abs(cx - (ox + ow / 2.0)) < 0.45 * max(w, ow)
                    and abs(cy - (oy + oh / 2.0)) < 0.45 * max(h, oh)
                    and (label == old_label or label in ("Parking", "Hospital")
                         and old_label in ("Parking", "Hospital"))):
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)

    for x, y, w, h, label, area in accepted:
        detections.append({"label": label, "area": int(round(area))})
    detections.sort(key=lambda entry: (SIGN_LABELS.index(entry["label"]), entry["area"]))
    ############################################################

    return detections


def detect_obstacle(image):
    '''
    Purpose:
    ---
    Detect whether the obstacle is visible in a single image, and how large
    it appears.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "present" : bool,   True if the obstacle is visible
            "area"    : int,    its area in pixels, 0 if not present
        }

    Example call:
    ---
    result = detect_obstacle(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    present = False
    area = 0

    #################### ADD YOUR CODE HERE ####################
    if image is None or image.size == 0:
        return {"present": present, "area": area}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    red = _red_mask(hsv)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(300.0, height * width * 0.0004)
    candidates = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 14:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0:
            continue
        # The obstacle is a SOLID, convex red block. Red-ring signs are hollow
        # (white inside) and the hospital's red cross is not convex.
        if contour_area / hull_area < 0.82:
            continue
        if _red_fill(red, contour) < 0.72:
            continue
        if not (0.35 <= w / float(h) <= 2.5):
            continue
        candidates.append((contour_area, 0.5 * (w * h + contour_area)))
    if candidates:
        present = True
        area = int(round(max(candidates, key=lambda item: item[0])[1]))
    ############################################################

    return {"present": present, "area": area}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------

def _red_fill(mask, contour):
    """Fraction of the area inside `contour` that is red in `mask` (ring ~0.35, solid ~0.95)."""
    x, y, w, h = cv2.boundingRect(contour)
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(filled, [contour - np.array([x, y])], -1, 255, thickness=-1)
    inside = int(np.count_nonzero(filled))
    if inside == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(filled, mask[y:y + h, x:x + w]))) / inside


def _rectify(image, contour):
    """Warp the (possibly tilted) sign face to an upright 64x64 square."""
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    s = box.sum(axis=1)
    d = np.diff(box, axis=1).ravel()
    src = np.float32([box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]])
    dst = np.float32([[0, 0], [63, 0], [63, 63], [0, 63]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (64, 64))


def _red_mask(hsv):
    """Return a cleaned mask covering both ends of OpenCV's red hue range."""
    lower = cv2.inRange(hsv, np.array([0, 95, 55], dtype=np.uint8),
                        np.array([12, 255, 255], dtype=np.uint8))
    upper = cv2.inRange(hsv, np.array([168, 95, 55], dtype=np.uint8),
                        np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area