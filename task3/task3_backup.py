#!/usr/bin/env python3
"""
Niti Vahan - Stage 1, Task 3
Team: NV_6236

Camera-only Task 3 controller:
- lane following
- Hospital pickup halt
- Speed 40 zone handling
- obstacle detection and lane changing
- ArUco parking

The script does not start/stop the simulation and does not read vehicle/sign/
obstacle positions from the simulator.
"""

import math
import os
import sys
import time

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================================
# 1. SCENE INTERFACE
# ============================================================================

VISION_SENSOR = "/Niti_Vahan/Camera_joint/Cuboid/visionSensor"
STEER_LEFT = "/steeringLeft"
STEER_RIGHT = "/steeringRight"
MOTOR_LEFT = "/motorLeft"
MOTOR_RIGHT = "/motorRight"

TEAM_ID = "NV_6236"


# ============================================================================
# 2. TASK 3 CONFIGURATION
# ============================================================================

WHEELBASE = 0.120
WHEEL_RADIUS = 0.050
TRACK_WIDTH = 0.110
WHEEL_OFFSET = 0.0275
STEER_LIMIT = math.radians(45.0)

LANE_WIDTH_M = 0.40
CORNER_CUT_M = 0.02
CORNER_CURVATURE_GAIN = 800.0
ROI_TOP = 0.45
SCAN_STEP = 4
NEAR_ROW = 0.90

YELLOW_LO = (15, 80, 80)
YELLOW_HI = (40, 255, 255)

WHITE_LO = (0, 0, 170)
WHITE_HI = (180, 60, 255)

K_STANLEY = 1.5
K_SOFT = 0.05
PSI_GAIN = 0.5

STEER_SMOOTH = 0.50
MAX_LANE_MISSES = 8

EASE_E_GAIN = 2.8
EASE_PSI_GAIN = 0.55
MIN_SPEED_FRACTION = 0.55

NORMAL_SPEED = 3.00
SPEED_40_LIMIT = 1.18
ABSOLUTE_SPEED_LIMIT = 3.00

LANE_CHANGE_SPEED = 1.35
OBSTACLE_NEAR_SPEED = 1.05

# Hospital
HOSPITAL_TRIGGER_FRAMES = 2
HOSPITAL_STILL_FRAMES = 6
HOSPITAL_STILL_DIFF_THRESHOLD = 2.0
HOSPITAL_STILL_MIN_SECONDS = 0.25
HOSPITAL_HOLD_SECONDS = 5.5
HOSPITAL_STOP_WIDTH_PX = 22
HOSPITAL_SIGHT_FRAMES = 3
HOSPITAL_APPROACH_SPEED = 1.05

# Distinct sign detection
SIGN_CONFIRM_FRAMES = 2
SIGN_REARM_FRAMES = 8

# Speed 40
SPEED_SIGN_MIN_AREA = 110

# Obstacles
OBSTACLE_MIN_AREA = 300
OBSTACLE_TRIGGER_BOTTOM_FRACTION = 0.34
OBSTACLE_TRACK_PASS_BOTTOM_FRACTION = 0.42
OBSTACLE_TRACK_MISSING_FRAMES = 5
OBSTACLE_TRACK_EXPIRE_FRAMES = 30
OBSTACLE_TRACK_MATCH_PX = 65.0

# Lane change
LANE_CHANGE_DISTANCE_M = LANE_WIDTH_M
LANE_CHANGE_TARGET_TOLERANCE_M = 0.095
LANE_CHANGE_CROSSING_M = 0.18
LANE_CHANGE_STABLE_FRAMES = 7
LANE_CHANGE_FALLBACK_DIRECTION = +1
LANE_CHANGE_REARM_FRAMES = 60

LANE_CHANGE_DEBUG = (
    os.environ.get("NV_DEBUG_LANE_CHANGE", "0").strip() == "1"
)

PARKING_DEBUG = (
    os.environ.get("NV_DEBUG_PARKING", "0").strip() == "1"
)

SIGN_DEBUG = (
    os.environ.get("NV_DEBUG_SIGNS", "0").strip() == "1"
)

DEBUG_LOG_PATH = os.environ.get("NV_DEBUG_LOG", "").strip()

_SIM_NOW = [0.0]
_raw_print = print


def print(*args, **kwargs):
    _raw_print("[t=%6.1fs]" % _SIM_NOW[0], *args, **kwargs)


def _debug_log(message):
    if DEBUG_LOG_PATH:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
            debug_file.write(str(message) + "\n")


# ============================================================================
# 3. PARKING CONFIGURATION
# ============================================================================

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

IMG_W = 640
IMG_H = 480

FOCAL_PX = (
    (IMG_W / 2.0) /
    math.tan(math.radians(95.0) / 2.0)
)

CAMERA_MATRIX = np.array([
    [FOCAL_PX, 0.0, IMG_W / 2.0],
    [0.0, FOCAL_PX, IMG_H / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

CAM_PITCH = math.radians(30.0)
CAM_ABOVE_MK = 0.176 - 0.110
CAM_AHEAD = 0.047

MARKER_SIDE = 0.096
_MARKER_HALF = MARKER_SIDE / 2.0

MARKER_CORNERS_3D = np.array([
    [-_MARKER_HALF, _MARKER_HALF, 0.0],
    [_MARKER_HALF, _MARKER_HALF, 0.0],
    [_MARKER_HALF, -_MARKER_HALF, 0.0],
    [-_MARKER_HALF, -_MARKER_HALF, 0.0],
], dtype=np.float32)

BAY_MIDDLE_FORWARD = 0.19
STOP_LEAD = 0.004

PARK_CRUISE_SPEED = 3.0
PARK_CREEP_SPEED = 2.0
PARK_CREEP_FROM = 0.30
PARK_CRUISE_UNTIL = 0.70

PARK_PLAN_END_FORWARD = 0.55
PARK_MIN_PLAN_RANGE = 0.10

PARK_HEADING_CAP = math.radians(42.0)
PARK_BEARING_LIMIT = math.radians(43.0)

PARK_PATH_GAIN = 2.0
PARK_STEER_GAIN = 0.8

PARK_HOLD_FRAMES = 6
PARK_GIVE_UP_FRAMES = 80
PARK_LOST_ODOM_SECONDS = 6.0
PARK_SEARCH_STEERING = math.radians(20.0)

PARK_YAW_BLEND = 0.12
PARK_YAW_GATE = math.radians(15.0)
PARK_AMBIGUITY_RATIO = 1.5

PARK_FINAL_SETTLE_SECONDS = 1.5
PARK_MARKER_STABLE_FRAMES = 5


# ============================================================================
# 4. SHARED PERCEPTION HELPERS
# ============================================================================

def _red_mask(hsv):
    lower = cv2.inRange(
        hsv,
        np.array([0, 95, 55], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8)
    )
    upper = cv2.inRange(
        hsv,
        np.array([168, 95, 55], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )
    mask = cv2.bitwise_or(lower, upper)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=1
    )


def _red_fill(mask, contour):
    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return 0.0

    filled = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)

    cv2.drawContours(filled, [shifted], -1, 255, thickness=-1)

    inside = int(np.count_nonzero(filled))
    if inside == 0:
        return 0.0

    red_pixels = int(np.count_nonzero(
        cv2.bitwise_and(
            filled,
            mask[y:y + h, x:x + w]
        )
    ))

    return red_pixels / float(inside)


def _rectify_sign(image, contour):
    box = cv2.boxPoints(cv2.minAreaRect(contour))

    sums = box.sum(axis=1)
    diffs = np.diff(box, axis=1).ravel()

    src = np.float32([
        box[np.argmin(sums)],
        box[np.argmin(diffs)],
        box[np.argmax(sums)],
        box[np.argmax(diffs)],
    ])

    dst = np.float32([
        [0, 0],
        [63, 0],
        [63, 63],
        [0, 63]
    ])

    return cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(src, dst),
        (64, 64)
    )


def detect_signs_detailed(image):
    detections = []

    if image is None or image.size == 0:
        return detections

    height, width = image.shape[:2]
    frame_area = height * width

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    candidates = []

    # ------------------------------------------------------------------
    # Red signs: Speed40 and School
    # ------------------------------------------------------------------

    red = _red_mask(hsv)

    red_contours, _ = cv2.findContours(
        red,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    red_floor = max(110.0, frame_area * 0.00030)

    for contour in red_contours:
        area = cv2.contourArea(contour)

        if area < red_floor:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if min(w, h) < max(12, int(min(height, width) * 0.018)):
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        fill_ratio = _red_fill(red, contour)

        polygon = cv2.approxPolyDP(
            contour,
            0.035 * perimeter,
            True
        )

        circularity = (
            4.0 * math.pi * area /
            (perimeter * perimeter)
        )

        aspect = w / float(h)

        # Speed40: ring-like, therefore not too filled.
        normal_ring = (
            fill_ratio < 0.70
            and 0.60 <= aspect <= 1.40
            and circularity >= 0.62
            and len(polygon) >= 5
        )

        edge_clipped = (
            x <= 1 or y <= 1
            or x + w >= width - 1
            or y + h >= height - 1
        )

        clipped_speed = (
            fill_ratio < 0.72
            and edge_clipped
            and x >= int(width * 0.82)
            and y <= int(height * 0.40)
            and 0.50 <= aspect <= 1.55
            and circularity >= 0.20
            and 300.0 <= area <= frame_area * 0.05
        )

        if normal_ring or clipped_speed:
            max_area_fraction = 0.05 if edge_clipped else 0.02

            if area <= frame_area * max_area_fraction:
                candidates.append(
                    ("Speed40", area, (x, y, w, h))
                )
            continue

        # School sign: triangular red sign.
        if (
            len(polygon) == 3
            and 0.48 <= aspect <= 1.35
            and not edge_clipped
            and area <= frame_area * 0.02
        ):
            candidates.append(
                ("School", area, (x, y, w, h))
            )

    # ------------------------------------------------------------------
    # Blue signs: Hospital and Parking
    # ------------------------------------------------------------------

    blue = cv2.inRange(
        hsv,
        np.array([88, 75, 45], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8)
    )

    blue_contours, hierarchy = cv2.findContours(
        blue,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        blue_floor = max(100.0, frame_area * 0.00018)

        for index, contour in enumerate(blue_contours):
            if hierarchy[index][3] != -1:
                continue

            area = cv2.contourArea(contour)

            if area < blue_floor:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if min(w, h) < max(14, int(min(height, width) * 0.025)):
                continue

            aspect = w / float(h)

            if not (0.60 <= aspect <= 1.60):
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))

            if hull_area <= 0 or area / hull_area < 0.80:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0:
                continue

            polygon = cv2.approxPolyDP(
                contour,
                0.055 * perimeter,
                True
            )

            if len(polygon) < 4 or len(polygon) > 8:
                continue

            clipped = (
                x <= 1
                or y <= 1
                or x + w >= width - 1
                or y + h >= height - 1
            )

            face = (
                cv2.resize(image[y:y + h, x:x + w], (64, 64))
                if clipped
                else _rectify_sign(image, contour)
            )

            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

            glyph = (
                (face_hsv[:, :, 2] >= 115)
                & (face_hsv[:, :, 1] <= 105)
            )

            inner_fraction = float(
                glyph[12:52, 12:52].mean()
            )

            if not (0.15 <= inner_fraction <= 0.70):
                continue

            lower_left = float(
                glyph[36:56, 14:30].mean()
            )

            lower_right = float(
                glyph[36:56, 34:50].mean()
            )

            p_like = (
                lower_right
                < 0.40 * max(lower_left, 0.05)
            )

            depth = 0
            child = hierarchy[index][2]

            while child != -1:
                depth = max(depth, 1)

                grand = hierarchy[child][2]

                while grand != -1:
                    depth = 2
                    grand = hierarchy[grand][0]

                child = hierarchy[child][0]

            label = (
                "Parking"
                if (p_like or depth >= 2 or clipped)
                else "Hospital"
            )

            if SIGN_DEBUG:
                print(
                    "[SIGN DEBUG] blue=%s bbox=%s area=%d "
                    "inner=%.3f lower_left=%.3f lower_right=%.3f "
                    "p_like=%s depth=%d clipped=%s"
                    % (
                        label,
                        (x, y, w, h),
                        int(round(area)),
                        inner_fraction,
                        lower_left,
                        lower_right,
                        p_like,
                        depth,
                        clipped,
                    )
                )

            candidates.append(
                (label, area, (x, y, w, h))
            )

    # ------------------------------------------------------------------
    # Merge duplicate fragments
    # ------------------------------------------------------------------

    accepted = []

    for item in sorted(candidates, key=lambda q: q[1], reverse=True):
        label, area, box = item
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0

        duplicate = False

        for old_label, _, old_box in accepted:
            ox, oy, ow, oh = old_box

            ocx = ox + ow / 2.0
            ocy = oy + oh / 2.0

            same_blue_sign = (
                label in ("Parking", "Hospital")
                and old_label in ("Parking", "Hospital")
            )

            if (
                abs(cx - ocx) < 0.45 * max(w, ow)
                and abs(cy - ocy) < 0.45 * max(h, oh)
                and (label == old_label or same_blue_sign)
            ):
                duplicate = True
                break

        if not duplicate:
            accepted.append(item)

    for label, area, box in accepted:
        detections.append({
            "label": label,
            "area": int(round(area)),
            "bbox": tuple(int(v) for v in box),
        })

    return detections


def detect_obstacles_detailed(image):
    if image is None or image.size == 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    height, width = image.shape[:2]
    red = _red_mask(hsv)

    contours, _ = cv2.findContours(
        red,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    minimum_area = max(
        float(OBSTACLE_MIN_AREA),
        height * width * 0.0004
    )

    found = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w > 0.60 * width or h > 0.60 * height:
            continue

        if min(w, h) < 14:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        polygon = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        # Reject triangular School signs.
        if len(polygon) == 3:
            continue

        circularity = (
            4.0 * math.pi * area /
            (perimeter * perimeter)
        )

        # Reject strongly circular Speed40 contours.
        if circularity > 0.78:
            continue

        hull_area = cv2.contourArea(
            cv2.convexHull(contour)
        )

        if hull_area <= 0 or area / hull_area < 0.82:
            continue

        if _red_fill(red, contour) < 0.72:
            continue

        aspect = w / float(h)

        if not (0.35 <= aspect <= 2.5):
            continue

        found.append({
            "area": float(area),
            "bbox": (
                int(x),
                int(y),
                int(w),
                int(h)
            ),
            "center": (
                float(x + w / 2.0),
                float(y + h / 2.0)
            ),
            "bottom": int(y + h),
        })

    found.sort(
        key=lambda q: q["area"],
        reverse=True
    )

    return found


# ============================================================================
# 5. LANE PERCEPTION
# ============================================================================

_LANE_MEMORY = {
    "row_width": {},
    "row_center": {},
    "last": None,
    "miss": 0,
}


def _marking_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(
        hsv,
        np.array(YELLOW_LO, np.uint8),
        np.array(YELLOW_HI, np.uint8)
    )

    white = cv2.inRange(
        hsv,
        np.array(WHITE_LO, np.uint8),
        np.array(WHITE_HI, np.uint8)
    )

    all_marks = cv2.bitwise_or(yellow, white)

    kernel = np.ones((3, 3), dtype=np.uint8)

    all_marks = cv2.morphologyEx(
        all_marks,
        cv2.MORPH_OPEN,
        kernel
    )

    return all_marks, yellow, white


def _line_colour_at(yellow, white, x, y):
    h, w = yellow.shape[:2]

    x0 = max(0, x - 3)
    x1 = min(w, x + 4)

    y0 = max(0, y - 2)
    y1 = min(h, y + 3)

    yc = int(
        np.count_nonzero(
            yellow[y0:y1, x0:x1]
        )
    )

    wc = int(
        np.count_nonzero(
            white[y0:y1, x0:x1]
        )
    )

    if yc > wc * 1.25 and yc >= 2:
        return "yellow"

    if wc > yc * 1.25 and wc >= 2:
        return "white"

    return "unknown"


def detect_lane(frame):
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    cx = (w - 1) / 2.0

    mask, yellow, white = _marking_masks(frame)

    y0 = int(h * ROI_TOP)

    rows = []
    centres = []

    left_samples = []
    right_samples = []

    left_colours = []
    right_colours = []

    for y in range(h - 1, y0, -SCAN_STEP):
        cols = np.flatnonzero(mask[y])

        if cols.size == 0:
            continue

        left_cols = cols[cols < cx - 3]
        right_cols = cols[cols > cx + 3]

        left_x = int(left_cols[-1]) if left_cols.size else None
        right_x = int(right_cols[0]) if right_cols.size else None

        known_width = _LANE_MEMORY["row_width"].get(y)
        previous_center = _LANE_MEMORY["row_center"].get(y, cx)

        center = None

        if left_x is not None and right_x is not None:
            gap = right_x - left_x

            # Guard new row-width samples using nearby rows. Because apparent
            # lane width changes with perspective, compare only against rows
            # close to the current scanline instead of using a global median.
            nearby_known_widths = [
                value
                for row, value in _LANE_MEMORY["row_width"].items()
                if abs(row - y) <= 24
            ]

            if known_width is None:
                if gap < 12 or gap > 0.85 * w:
                    continue

                if len(nearby_known_widths) >= 3:
                    local_width = float(np.median(nearby_known_widths))

                    if not (
                        0.55 * local_width
                        <= gap
                        <= 1.80 * local_width
                    ):
                        continue

                center = 0.5 * (left_x + right_x)
                _LANE_MEMORY["row_width"][y] = float(gap)

            elif gap < 0.60 * known_width:
                continue

            elif gap <= 1.35 * known_width:
                center = 0.5 * (left_x + right_x)

                _LANE_MEMORY["row_width"][y] = (
                    0.90 * known_width
                    + 0.10 * gap
                )

            else:
                c1 = left_x + 0.5 * known_width
                c2 = right_x - 0.5 * known_width

                center = (
                    c1
                    if abs(c1 - previous_center)
                    <= abs(c2 - previous_center)
                    else c2
                )

        elif known_width is not None and left_x is not None:
            center = left_x + 0.5 * known_width

        elif known_width is not None and right_x is not None:
            center = right_x - 0.5 * known_width

        if center is None:
            continue

        _LANE_MEMORY["row_center"][y] = float(center)

        rows.append(float(y))
        centres.append(float(center))

        if left_x is not None:
            left_samples.append((y, left_x))
            left_colours.append(
                _line_colour_at(
                    yellow,
                    white,
                    left_x,
                    y
                )
            )

        if right_x is not None:
            right_samples.append((y, right_x))
            right_colours.append(
                _line_colour_at(
                    yellow,
                    white,
                    right_x,
                    y
                )
            )

    _LANE_MEMORY["row_width"] = {
        k: v
        for k, v in _LANE_MEMORY["row_width"].items()
        if y0 <= k < h
    }

    _LANE_MEMORY["row_center"] = {
        k: v
        for k, v in _LANE_MEMORY["row_center"].items()
        if y0 <= k < h
    }

    if len(rows) < 5:
        _LANE_MEMORY["miss"] += 1

        last = _LANE_MEMORY["last"]

        if (
            last is not None
            and _LANE_MEMORY["miss"] <= MAX_LANE_MISSES
        ):
            return dict(last)

        return None

    ys = np.asarray(rows, dtype=float)
    xs = np.asarray(centres, dtype=float)

    degree = 2 if len(ys) >= 8 else 1

    coeff = np.polyfit(ys, xs, degree)

    residuals = np.abs(
        np.polyval(coeff, ys) - xs
    )

    median_res = float(np.median(residuals))

    keep = (
        residuals
        <= max(3.0, 2.5 * median_res)
    )

    if 5 <= int(keep.sum()) < len(ys):
        coeff = np.polyfit(
            ys[keep],
            xs[keep],
            degree
        )

    reference_y = min(
        h * NEAR_ROW,
        float(ys.max())
    )

    center_x = float(
        np.polyval(coeff, reference_y)
    )

    slope = float(
        np.polyval(
            np.polyder(coeff),
            reference_y
        )
    )

    nearby_widths = [
        value
        for row, value in _LANE_MEMORY["row_width"].items()
        if abs(row - reference_y) <= 20
    ]

    if not nearby_widths:
        _LANE_MEMORY["miss"] += 1
        return None

    lane_width_px = float(
        np.median(nearby_widths)
    )

    metres_per_pixel = (
        LANE_WIDTH_M /
        max(lane_width_px, 1.0)
    )

    cross_track = (
        (cx - center_x) *
        metres_per_pixel
    )

    heading_error = (
        PSI_GAIN * math.atan(slope)
    )

    near_left_colours = [
        c
        for (yy, _), c in zip(
            left_samples,
            left_colours
        )
        if yy >= int(h * 0.65)
    ]

    near_right_colours = [
        c
        for (yy, _), c in zip(
            right_samples,
            right_colours
        )
        if yy >= int(h * 0.65)
    ]

    left_colour = (
        max(
            set(near_left_colours),
            key=near_left_colours.count
        )
        if near_left_colours
        else "unknown"
    )

    right_colour = (
        max(
            set(near_right_colours),
            key=near_right_colours.count
        )
        if near_right_colours
        else "unknown"
    )

    left_near = [
        x
        for y, x in left_samples
        if y >= h * 0.65
    ]

    right_near = [
        x
        for y, x in right_samples
        if y >= h * 0.65
    ]

    result = {
        "e": float(cross_track),
        "psi": float(heading_error),
        "center_x": center_x,
        "image_height": h,
        "image_width": w,
        "lane_width_px": lane_width_px,
        "m_per_px": metres_per_pixel,
        "reference_y": reference_y,
        "poly": coeff,
        "left_colour": left_colour,
        "right_colour": right_colour,
        "left_x": (
            float(np.median(left_near))
            if left_near else None
        ),
        "right_x": (
            float(np.median(right_near))
            if right_near else None
        ),
    }

    _LANE_MEMORY["miss"] = 0
    _LANE_MEMORY["last"] = dict(result)

    return result


def _lane_center_at_y(lane, y):
    if lane is None:
        return None

    return float(
        np.polyval(
            lane["poly"],
            float(y)
        )
    )


def _lane_change_direction(frame, lane):
    if lane is None:
        return LANE_CHANGE_FALLBACK_DIRECTION

    left_colour = lane.get("left_colour", "unknown")
    right_colour = lane.get("right_colour", "unknown")

    # Yellow is road edge; white is lane separator.
    if (
        left_colour == "white"
        and right_colour == "yellow"
    ):
        return +1

    if (
        right_colour == "white"
        and left_colour == "yellow"
    ):
        return -1

    mask, _, _ = _marking_masks(frame)

    y = int(frame.shape[0] * 0.76)

    left_x = lane.get("left_x")
    right_x = lane.get("right_x")

    left_extra = 0
    right_extra = 0

    if left_x is not None:
        left_extra = int(
            np.count_nonzero(
                mask[y, :max(0, int(left_x) - 8)]
            )
        )

    if right_x is not None:
        right_extra = int(
            np.count_nonzero(
                mask[
                    y,
                    min(mask.shape[1], int(right_x) + 8):
                ]
            )
        )

    if (
        left_extra > right_extra * 1.25
        and left_extra > 2
    ):
        return +1

    if (
        right_extra > left_extra * 1.25
        and right_extra > 2
    ):
        return -1

    return LANE_CHANGE_FALLBACK_DIRECTION


def _obstacle_in_target_lane(obstacle, lane, lane_offset_m):
    if lane is None:
        return False

    x, y, w, h = obstacle["bbox"]

    frame_h = float(
        lane.get("image_height", 480)
    )

    base_y = min(
        frame_h - 1.0,
        float(y + h)
    )

    base_y = min(
        max(base_y, 0.0),
        frame_h - 1.0
    )

    center_x = _lane_center_at_y(
        lane,
        base_y
    )

    if center_x is None:
        center_x = lane["center_x"]

    m_per_px = max(
        float(lane["m_per_px"]),
        1e-5
    )

    horizon_y = 71.0

    reference_y = max(
        float(lane.get(
            "reference_y",
            frame_h * NEAR_ROW
        )),
        horizon_y + 1.0
    )

    row_scale = max(
        0.20,
        (base_y - horizon_y) /
        (reference_y - horizon_y)
    )

    target_center_x = (
        center_x
        - lane_offset_m / m_per_px * row_scale
    )

    lane_width_px = max(
        float(lane["lane_width_px"]) * row_scale,
        12.0
    )

    lane_left = (
        target_center_x
        - 0.55 * lane_width_px
    )

    lane_right = (
        target_center_x
        + 0.55 * lane_width_px
    )

    box_left = float(x)
    box_right = float(x + w)

    horizontal_overlap = (
        box_right >= lane_left
        and box_left <= lane_right
    )

    return (
        bool(horizontal_overlap),
        base_y,
        target_center_x
    )


# ============================================================================
# 6. OBSTACLE PASS TRACKER (DIAGNOSTIC ONLY)
# ============================================================================

class ObstaclePassTracker:

    def __init__(self):
        self.frame_index = 0
        self.next_id = 1
        self.tracks = []
        self.passed = 0

    def update(
        self,
        detections,
        frame_height,
        lane=None,
        lane_offset_m=0.0
    ):
        self.frame_index += 1
        used_tracks = set()

        for det in detections:
            cx, cy = det["center"]
            area = float(det.get("area", 0.0))
            bottom = float(det["bottom"])

            road_candidate = False

            if lane is not None:
                for candidate_offset in (
                    lane_offset_m,
                    lane_offset_m + LANE_WIDTH_M,
                    lane_offset_m - LANE_WIDTH_M,
                ):
                    overlap = _obstacle_in_target_lane(
                        det,
                        lane,
                        candidate_offset
                    )

                    if overlap is not False and overlap[0]:
                        road_candidate = True
                        break

            best = None
            best_dist = float("inf")

            for track in self.tracks:
                if track["id"] in used_tracks:
                    continue

                dt = (
                    self.frame_index
                    - track["last_seen"]
                )

                if dt > OBSTACLE_TRACK_EXPIRE_FRAMES:
                    continue

                dist = math.hypot(
                    cx - track["cx"],
                    cy - track["cy"]
                )

                area_ratio = area / max(
                    float(track["peak_area"]),
                    1.0
                )

                if (
                    dist < OBSTACLE_TRACK_MATCH_PX
                    and 0.20 <= area_ratio <= 5.0
                    and dist < best_dist
                ):
                    best = track
                    best_dist = dist

            if best is None:
                best = {
                    "id": self.next_id,
                    "cx": cx,
                    "cy": cy,
                    "bottom": bottom,
                    "peak_area": area,
                    "road_hits": int(road_candidate),
                    "last_seen": self.frame_index,
                    "counted": False,
                }

                self.next_id += 1
                self.tracks.append(best)

            else:
                best["cx"] = (
                    0.35 * best["cx"]
                    + 0.65 * cx
                )

                best["cy"] = (
                    0.35 * best["cy"]
                    + 0.65 * cy
                )

                best["bottom"] = max(
                    best["bottom"],
                    bottom
                )

                best["peak_area"] = max(
                    best["peak_area"],
                    area
                )

                if road_candidate:
                    best["road_hits"] += 1

                best["last_seen"] = self.frame_index

            used_tracks.add(best["id"])

        surviving_tracks = []

        for track in self.tracks:
            missing = (
                self.frame_index
                - track["last_seen"]
            )

            reached_near_field = (
                track["bottom"]
                >= (
                    OBSTACLE_TRACK_PASS_BOTTOM_FRACTION
                    * frame_height
                )
            )

            if (
                not track["counted"]
                and missing >= OBSTACLE_TRACK_MISSING_FRAMES
                and reached_near_field
                and track.get("road_hits", 0) >= 2
            ):
                track["counted"] = True
                self.passed += 1

                print(
                    "[OBSTACLE] road obstacle cleared "
                    "(total=%d)"
                    % self.passed
                )

            if missing < OBSTACLE_TRACK_EXPIRE_FRAMES:
                surviving_tracks.append(track)

        self.tracks = surviving_tracks


# ============================================================================
# 7. TASK 2C ARUCO PARKING
# ============================================================================

class PoseEstimator:

    def __init__(self):
        self.heading = 0.0
        self.rho = None

    @staticmethod
    def bearing(corners):
        cx = corners[:, 0].mean()
        cy = corners[:, 1].mean()

        xc = (cx - IMG_W / 2.0) / FOCAL_PX
        yc = (IMG_H / 2.0 - cy) / FOCAL_PX

        return float(
            math.atan2(
                xc,
                math.cos(CAM_PITCH)
                + yc * math.sin(CAM_PITCH)
            )
        )

    @staticmethod
    def pnp(corners):
        result = cv2.solvePnPGeneric(
            MARKER_CORNERS_3D,
            corners.astype(np.float32),
            CAMERA_MATRIX,
            np.zeros(5),
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        _, rvecs, tvecs, errors = result[:4]

        solutions = []

        error_values = np.asarray(errors).flatten()

        for rv, tv, err in zip(
            rvecs,
            tvecs,
            error_values
        ):
            rotation, _ = cv2.Rodrigues(rv)

            forward = (
                rotation.T
                @ np.array([0.0, 0.0, 1.0])
            )

            solutions.append((
                float(err),
                float(
                    math.atan2(
                        forward[0],
                        -forward[2]
                    )
                ),
                tv.flatten(),
            ))

        if not solutions:
            raise RuntimeError(
                "ArUco PnP returned no pose solutions"
            )

        solutions.sort(key=lambda item: item[0])
        best = solutions[0]

        rho = float(
            math.sqrt(
                max(
                    float(best[2] @ best[2])
                    - CAM_ABOVE_MK ** 2,
                    1e-6
                )
            )
        )

        clear = (
            len(solutions) < 2
            or solutions[1][0]
            > (
                PARK_AMBIGUITY_RATIO
                * max(solutions[0][0], 1e-6)
            )
        )

        return (
            best[1] if clear else None,
            rho
        )

    def update(self, corners, last_steering):
        if corners is None:
            return None

        bearing = self.bearing(corners)

        yaw_seen, rho_now = self.pnp(corners)

        if self.rho is None:
            self.rho = rho_now

        rho_previous = self.rho

        self.rho = (
            0.60 * self.rho
            + 0.40 * rho_now
        )

        ds = min(
            max(
                (
                    rho_previous - self.rho
                ) / max(math.cos(bearing), 0.3),
                0.0
            ),
            0.01
        )

        self.heading -= (
            math.tan(last_steering)
            / WHEELBASE
            * ds
        )

        if (
            yaw_seen is not None
            and abs(
                yaw_seen - self.heading
            ) < PARK_YAW_GATE
        ):
            self.heading += (
                PARK_YAW_BLEND
                * (yaw_seen - self.heading)
            )

        line_of_sight = (
            bearing + self.heading
        )

        lateral_camera = (
            -self.rho
            * math.sin(line_of_sight)
        )

        forward_camera = (
            self.rho
            * math.cos(line_of_sight)
        )

        lateral_front = (
            lateral_camera
            + (
                WHEELBASE / 2.0 - CAM_AHEAD
            ) * math.sin(self.heading)
        )

        forward = (
            forward_camera
            + CAM_AHEAD * math.cos(self.heading)
        )

        return {
            "lateral": float(-lateral_front),
            "heading": float(-self.heading),
            "forward": float(forward),
            "bearing": float(-bearing),
        }


def _parking_path_steering(pose):
    lateral = pose["lateral"]
    heading = pose["heading"]

    plan_range = max(
        pose["forward"] - PARK_PLAN_END_FORWARD,
        PARK_MIN_PLAN_RANGE
    )

    wanted = -math.atan(
        PARK_PATH_GAIN
        * lateral
        / plan_range
    )

    wanted = float(
        np.clip(
            wanted,
            -PARK_HEADING_CAP,
            PARK_HEADING_CAP
        )
    )

    if (
        abs(pose["bearing"]) > PARK_BEARING_LIMIT
        and wanted * pose["bearing"] < 0.0
    ):
        wanted = (
            max(wanted, heading)
            if wanted < 0.0
            else min(wanted, heading)
        )

    return float(
        np.clip(
            PARK_STEER_GAIN * (wanted - heading),
            -STEER_LIMIT,
            STEER_LIMIT
        )
    )


class ParkingController:

    def __init__(self):
        self.estimator = PoseEstimator()
        self.last_steering = 0.0
        self.lost_frames = 0
        self.last_bearing = 0.0
        self.target_id = None
        self.parked = False
        self.debug_frame = 0
        self.last_pose = None
        self.last_speed = 0.0
        self.last_sim_time = None
        self.lost_sim_seconds = 0.0

    @staticmethod
    def _make_detector():
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is unavailable. "
                "Install an OpenCV build with the aruco module."
            )

        dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT_ID
        )

        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(
                dictionary,
                cv2.aruco.DetectorParameters()
            )

        return dictionary

    def _detect_marker(
        self,
        frame,
        detector,
        wanted_id=None
    ):
        if hasattr(detector, "detectMarkers"):
            corners, ids, _ = detector.detectMarkers(
                frame
            )
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame,
                detector
            )

        if ids is None or len(ids) == 0:
            return None, None

        best_index = None
        best_area = -1.0

        marker_ids = ids.flatten()

        for i, quad in enumerate(corners):
            if (
                wanted_id is not None
                and int(marker_ids[i]) != int(wanted_id)
            ):
                continue

            area = abs(
                cv2.contourArea(
                    quad.reshape(4, 2).astype(np.float32)
                )
            )

            if area > best_area:
                best_area = area
                best_index = i

        if best_index is None:
            return None, None

        return (
            corners[best_index].reshape(4, 2),
            int(marker_ids[best_index])
        )

    def step(
        self,
        frame,
        detector,
        sim_time
    ):
        self.debug_frame += 1

        now = float(sim_time)

        dt = (
            0.0
            if self.last_sim_time is None
            else max(
                0.0,
                min(now - self.last_sim_time, 0.10)
            )
        )

        self.last_sim_time = now

        corners, marker_id = self._detect_marker(
            frame,
            detector,
            self.target_id
        )

        if corners is not None:
            self.target_id = marker_id

            pose = self.estimator.update(
                corners,
                self.last_steering
            )

            if pose is not None:
                self.last_bearing = pose["bearing"]

            self.lost_frames = 0
            self.lost_sim_seconds = 0.0

            if pose is not None:
                self.last_pose = dict(pose)

        else:
            self.lost_frames += 1
            self.lost_sim_seconds += dt

            pose = None

            if (
                self.last_pose is not None
                and self.lost_sim_seconds
                <= PARK_LOST_ODOM_SECONDS
            ):
                pose = dict(self.last_pose)

                v = (
                    self.last_speed
                    * WHEEL_RADIUS
                )

                theta = pose["heading"]

                dtheta = (
                    math.tan(self.last_steering)
                    * v
                    * dt
                    / WHEELBASE
                )

                mid_theta = (
                    theta + 0.5 * dtheta
                )

                pose["lateral"] += (
                    v
                    * math.sin(mid_theta)
                    * dt
                )

                pose["forward"] -= (
                    v
                    * math.cos(mid_theta)
                    * dt
                )

                pose["heading"] += dtheta
                pose["bearing"] -= dtheta

                self.last_pose = dict(pose)

        if PARKING_DEBUG and self.debug_frame % 100 == 0:
            if pose is None:
                print(
                    "[PARK DEBUG] frame=%d marker=%s lost=%d"
                    % (
                        self.debug_frame,
                        self.target_id,
                        self.lost_frames
                    )
                )
            else:
                print(
                    "[PARK DEBUG] frame=%d id=%s "
                    "forward=%.3f lateral=%.3f "
                    "heading=%.3f bearing=%.3f"
                    % (
                        self.debug_frame,
                        marker_id,
                        pose["forward"],
                        pose["lateral"],
                        pose["heading"],
                        pose["bearing"],
                    )
                )

        if pose is not None:
            if (
                pose["forward"]
                <= BAY_MIDDLE_FORWARD + STOP_LEAD
            ):
                self.parked = True
                self.last_speed = 0.0

                return 0.0, 0.0, True

            steering = _parking_path_steering(
                pose
            )

            self.last_steering = steering

            speed = float(
                np.interp(
                    pose["forward"],
                    [
                        PARK_CREEP_FROM,
                        PARK_CRUISE_UNTIL
                    ],
                    [
                        PARK_CREEP_SPEED,
                        PARK_CRUISE_SPEED
                    ]
                )
            )

            self.last_speed = speed

            return steering, speed, False

        # IMPORTANT: lost_frames is not incremented again here.
        if self.lost_frames <= PARK_HOLD_FRAMES:
            self.last_speed = PARK_CREEP_SPEED

            return (
                self.last_steering,
                PARK_CREEP_SPEED,
                False
            )

        if self.lost_frames <= PARK_GIVE_UP_FRAMES:
            search = (
                math.copysign(
                    PARK_SEARCH_STEERING,
                    self.last_bearing
                )
                if self.last_bearing
                else 0.0
            )

            self.last_steering = search
            self.last_speed = PARK_CREEP_SPEED

            return (
                search,
                PARK_CREEP_SPEED,
                False
            )

        self.last_speed = 0.0
        return 0.0, 0.0, False


# ============================================================================
# 8. VEHICLE IO
# ============================================================================

def connect_and_locate():
    try:
        client = RemoteAPIClient()
        sim = client.getObject("sim")
        sim.getSimulationTime()

    except Exception as err:
        raise ConnectionError(
            "Could not connect to CoppeliaSim through "
            "the ZMQ Remote API. Check that Task 3 is open "
            "and only one CoppeliaSim instance is running. "
            "Simulator said: %s" % err
        )

    handles = {}

    for key, path in (
        ("vision_sensor", VISION_SENSOR),
        ("steer_left", STEER_LEFT),
        ("steer_right", STEER_RIGHT),
        ("motor_left", MOTOR_LEFT),
        ("motor_right", MOTOR_RIGHT),
    ):
        try:
            handles[key] = sim.getObject(path)

        except Exception as err:
            raise RuntimeError(
                'Could not find scene object "%s". '
                "Check the Task 3 scene hierarchy. "
                "Simulator said: %s"
                % (path, err)
            )

    return client, sim, handles


def read_frame(sim, handles):
    sim.handleVisionSensor(
        handles["vision_sensor"]
    )

    image_buffer, resolution = sim.getVisionSensorImg(
        handles["vision_sensor"],
        0
    )

    if not image_buffer:
        return None

    frame = np.frombuffer(
        image_buffer,
        dtype=np.uint8
    ).reshape(
        (
            resolution[1],
            resolution[0],
            3
        )
    )

    frame = cv2.flip(frame, 0)

    return cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )


def ackermann_wheel_angles(delta):
    if abs(delta) < 1e-9:
        return 0.0, 0.0

    effective_track = (
        TRACK_WIDTH
        - 2.0 * WHEEL_OFFSET
    )

    sine = math.sin(delta)
    cosine = math.cos(delta)

    left = math.atan2(
        2.0 * WHEELBASE * sine,
        2.0 * WHEELBASE * cosine
        - effective_track * sine
    )

    right = math.atan2(
        2.0 * WHEELBASE * sine,
        2.0 * WHEELBASE * cosine
        + effective_track * sine
    )

    return float(left), float(right)


def drive(
    sim,
    handles,
    steering,
    speed
):
    steering = float(
        np.clip(
            steering,
            -STEER_LIMIT,
            STEER_LIMIT
        )
    )

    speed = float(
        np.clip(
            speed,
            0.0,
            ABSOLUTE_SPEED_LIMIT
        )
    )

    left, right = ackermann_wheel_angles(
        steering
    )

    sim.setJointTargetPosition(
        handles["steer_left"],
        left
    )

    sim.setJointTargetPosition(
        handles["steer_right"],
        right
    )

    sim.setJointTargetVelocity(
        handles["motor_left"],
        speed
    )

    sim.setJointTargetVelocity(
        handles["motor_right"],
        speed
    )


# ============================================================================
# 9. TASK 3 STATE MACHINE
# ============================================================================

class Task3Controller:

    def __init__(self):
        # Speed 40
        self.speed40_events = 0
        self.speed_zone = False
        self.speed_latched = False
        self.speed_seen_frames = 0
        self.speed_absent_frames = 0

        # Hospital
        self.hospital_state = "WAIT_FIRST"
        self.hospital_events = 0
        self.hospital_latched = False
        self.hospital_seen_frames = 0
        self.hospital_absent_frames = 0
        self.hospital_trigger_frames = 0
        self.hospital_hold_start = None

        self.last_gray = None
        self.still_count = 0
        self.last_settle_sim_time = None
        self.still_start_sim_time = None

        # Parking
        self.parking_started = False
        self.parking_marker_candidate = None
        self.parking_marker_streak = 0
        self.parking_controller = ParkingController()
        self.parking_detector = None

        # Obstacles
        self.obstacle_tracker = ObstaclePassTracker()

        # lane_offset_m is ONLY a temporary lane-change target.
        self.lane_offset_m = 0.0
        self.corner_bias_m = 0.0
        self.lane_change_active = False
        self.lane_change_direction = 0
        self.lane_change_stable = 0
        self.lane_change_progress = 0.0
        self.previous_lane_error = None
        self.last_lane_change_frame = -LANE_CHANGE_REARM_FRAMES
        self.completed_obstacle_avoids = 0
        # True only while an obstacle blocks our lane and the adjacent
        # target lane is also occupied.  In that case we stop and wait
        # rather than committing to an unsafe lane change.
        self.target_lane_waiting = False

        # General
        self.last_steering = 0.0
        self.cap_reason = "lane_follow"
        self.frame_count = 0
        self.done = False
        self.travel_rad = 0.0

    @property
    def hospital_done(self):
        return self.hospital_state == "DONE"

    # ------------------------------------------------------------------
    # Speed 40
    # ------------------------------------------------------------------

    def _update_speed_zone(self, signs):
        visible = [
            sign
            for sign in signs
            if (
                sign["label"] == "Speed40"
                and sign["area"] >= SPEED_SIGN_MIN_AREA
            )
        ]

        if visible:
            self.speed_absent_frames = 0

            if not self.speed_latched:
                self.speed_seen_frames += 1

                if (
                    self.speed_seen_frames
                    >= SIGN_CONFIRM_FRAMES
                ):
                    self.speed_latched = True
                    self.speed_seen_frames = 0

                    # Task order: ignore Speed40 sightings before Hospital.
                    if not self.hospital_done:
                        return

                    self.speed40_events += 1

                    best = max(
                        visible,
                        key=lambda s: s["area"]
                    )

                    if self.speed40_events == 1:
                        self.speed_zone = True

                        print(
                            "[SPEED] first distinct Speed40 "
                            "detected -> entering restricted zone "
                            "(bbox=%s area=%d)"
                            % (
                                best["bbox"],
                                best["area"]
                            )
                        )

                    elif self.speed40_events == 2:
                        self.speed_zone = False

                        print(
                            "[SPEED] second distinct Speed40 "
                            "detected -> leaving restricted zone "
                            "(bbox=%s area=%d)"
                            % (
                                best["bbox"],
                                best["area"]
                            )
                        )

                    else:
                        self.speed40_events = 2

        else:
            self.speed_seen_frames = 0

            if self.speed_latched:
                self.speed_absent_frames += 1

                if (
                    self.speed_absent_frames
                    >= SIGN_REARM_FRAMES
                ):
                    self.speed_latched = False
                    self.speed_absent_frames = 0

    # ------------------------------------------------------------------
    # Hospital
    # ------------------------------------------------------------------

    def _update_hospital_detection(self, signs):
        if self.hospital_state in (
            "SETTLING",
            "HOLDING",
            "DONE"
        ):
            return

        visible = [
            sign
            for sign in signs
            if sign["label"] == "Hospital"
        ]

        if not visible:
            self.hospital_seen_frames = 0
            # A stop trigger must come from consecutive visual evidence.
            # Do not allow threshold hits to accumulate across sign dropouts.
            self.hospital_trigger_frames = 0

            if self.hospital_latched:
                self.hospital_absent_frames += 1

                if (
                    self.hospital_absent_frames
                    >= SIGN_REARM_FRAMES
                ):
                    self.hospital_latched = False
                    self.hospital_absent_frames = 0

            return

        self.hospital_absent_frames = 0

        best = max(
            visible,
            key=lambda s: s["area"]
        )

        _, _, bw, bh = best["bbox"]
        sign_size = max(bw, bh)

        if not self.hospital_latched:
            self.hospital_seen_frames += 1

            if (
                self.hospital_seen_frames
                >= HOSPITAL_SIGHT_FRAMES
            ):
                self.hospital_latched = True
                self.hospital_seen_frames = 0
                self.hospital_events += 1

                print(
                    "[HOSPITAL] distinct Hospital sign #%d "
                    "(size=%d area=%d bbox=%s)"
                    % (
                        self.hospital_events,
                        sign_size,
                        best["area"],
                        best["bbox"]
                    )
                )

                if self.hospital_events == 1:
                    self.hospital_state = "APPROACH"

                    print(
                        "[HOSPITAL] approaching pickup at %.2f rad/s"
                        % HOSPITAL_APPROACH_SPEED
                    )

        # Second hospital sign acts as the visual stop landmark.
        if (
            self.hospital_state == "APPROACH"
            and self.hospital_events >= 2
        ):
            if sign_size >= HOSPITAL_STOP_WIDTH_PX:
                self.hospital_trigger_frames += 1
            else:
                self.hospital_trigger_frames = 0

            if (
                self.hospital_trigger_frames
                >= HOSPITAL_TRIGGER_FRAMES
            ):
                self.hospital_state = "SETTLING"

                self.last_gray = None
                self.still_count = 0
                self.still_start_sim_time = None
                self.last_settle_sim_time = None

                print(
                    "[HOSPITAL] second Hospital landmark "
                    "reached threshold; braking "
                    "(size=%d bbox=%s)"
                    % (
                        sign_size,
                        best["bbox"]
                    )
                )

    def _update_hospital_hold(
        self,
        frame,
        now
    ):
        now = float(now)

        if self.hospital_state == "SETTLING":
            if (
                self.last_settle_sim_time is not None
                and now
                <= self.last_settle_sim_time + 1e-9
            ):
                return

            self.last_settle_sim_time = now

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            roi = gray[
                int(gray.shape[0] * 0.30):
                ,
                :
            ]

            roi = cv2.resize(
                roi,
                (160, 90)
            )

            if self.last_gray is None:
                difference = float("inf")
            else:
                difference = float(
                    np.mean(
                        cv2.absdiff(
                            roi,
                            self.last_gray
                        )
                    )
                )

            self.last_gray = roi

            if (
                difference
                <= HOSPITAL_STILL_DIFF_THRESHOLD
            ):
                if self.still_count == 0:
                    self.still_start_sim_time = now

                self.still_count += 1
            else:
                self.still_count = 0
                self.still_start_sim_time = None

            stable_time = (
                now - self.still_start_sim_time
                if self.still_start_sim_time is not None
                else 0.0
            )

            if (
                self.still_count
                >= HOSPITAL_STILL_FRAMES
                and stable_time
                >= HOSPITAL_STILL_MIN_SECONDS
            ):
                self.hospital_state = "HOLDING"
                self.hospital_hold_start = now

                print(
                    "[HOSPITAL] vehicle stationary; "
                    "starting %.1f s simulation-time hold"
                    % HOSPITAL_HOLD_SECONDS
                )

        elif self.hospital_state == "HOLDING":
            if (
                self.hospital_hold_start is not None
                and (
                    now - self.hospital_hold_start
                    >= HOSPITAL_HOLD_SECONDS
                )
            ):
                self.hospital_state = "DONE"

                print(
                    "[HOSPITAL] stop complete "
                    "(%.1f s)"
                    % HOSPITAL_HOLD_SECONDS
                )

    # ------------------------------------------------------------------
    # Obstacle / lane change
    # ------------------------------------------------------------------

    def _obstacle_in_path(
        self,
        obstacles,
        lane,
        frame_h
    ):
        if lane is None:
            for obstacle in obstacles:
                if (
                    obstacle["bottom"]
                    >= (
                        OBSTACLE_TRIGGER_BOTTOM_FRACTION
                        * frame_h
                    )
                ):
                    return True

            return False

        for obstacle in obstacles:
            if (
                obstacle["bottom"]
                < (
                    OBSTACLE_TRIGGER_BOTTOM_FRACTION
                    * frame_h
                )
            ):
                continue

            result = _obstacle_in_target_lane(
                obstacle,
                lane,
                self.lane_offset_m
            )

            if result is not False and result[0]:
                return True

        return False

    def _target_lane_blocked(
        self,
        obstacles,
        lane,
        direction,
        ignore_obstacle=None
    ):
        """Return True when the requested adjacent lane contains an obstacle.

        This uses the same camera-derived lane geometry as the normal obstacle
        test.  No simulator object positions are read.
        """
        if lane is None:
            return True

        target_offset = (
            direction * LANE_CHANGE_DISTANCE_M
        )

        for obstacle in obstacles:
            if obstacle is ignore_obstacle:
                continue

            result = _obstacle_in_target_lane(
                obstacle,
                lane,
                target_offset
            )

            if result is False:
                continue

            in_lane, base_y, _ = result

            # Ignore extremely distant detections near the horizon.
            if (
                in_lane
                and base_y
                >= 0.25 * lane["image_height"]
            ):
                return True

        return False

    def _maybe_start_lane_change(
        self,
        frame,
        lane,
        obstacles
    ):
        if lane is None:
            self.target_lane_waiting = False
            return False

        if self.lane_change_active:
            self.target_lane_waiting = False
            return False

        # Re-evaluate the target lane every frame. Keep the previous WAIT
        # state only so logging happens on transitions rather than every frame.
        was_waiting = self.target_lane_waiting
        self.target_lane_waiting = False

        if (
            self.frame_count
            - self.last_lane_change_frame
            < LANE_CHANGE_REARM_FRAMES
        ):
            return False

        if self.hospital_state in (
            "SETTLING",
            "HOLDING"
        ):
            return False

        h, w = frame.shape[:2]

        ordered = sorted(
            obstacles,
            key=lambda ob: ob["bottom"],
            reverse=True
        )

        for obstacle in ordered:
            x, y, bw, bh = obstacle["bbox"]

            if (
                (
                    bw > 0.22 * w
                    or bh > 0.22 * h
                )
                and (
                    y + bh
                    < 0.55 * h
                )
            ):
                continue

            result = _obstacle_in_target_lane(
                obstacle,
                lane,
                0.0
            )

            if result is False:
                continue

            in_lane, base_y, _ = result

            if not in_lane:
                continue

            if (
                base_y
                < (
                    OBSTACLE_TRIGGER_BOTTOM_FRACTION
                    * h
                )
            ):
                continue

            direction = _lane_change_direction(
                frame,
                lane
            )

            if self._target_lane_blocked(
                obstacles,
                lane,
                direction,
                obstacle
            ):
                self.target_lane_waiting = True

                if not was_waiting:
                    print(
                        "[LANE CHANGE] adjacent target lane blocked; "
                        "stopping until it becomes free"
                    )

                return False

            if was_waiting:
                print(
                    "[LANE CHANGE] adjacent target lane is clear; "
                    "starting lane change"
                )

            self.lane_change_direction = direction

            self.lane_offset_m = (
                direction
                * LANE_CHANGE_DISTANCE_M
            )

            self.lane_change_active = True

            self.lane_change_stable = 0
            self.lane_change_progress = 0.0

            self.previous_lane_error = float(
                lane["e"]
            )

            self.last_lane_change_frame = (
                self.frame_count
            )

            print(
                "[LANE CHANGE] obstacle in current lane; "
                "changing %s "
                "(bbox=%s area=%.0f bottom=%.0f)"
                % (
                    "left"
                    if direction > 0
                    else "right",
                    obstacle["bbox"],
                    obstacle["area"],
                    base_y
                )
            )

            return True

        return False

    def _finish_lane_change(
        self,
        reason
    ):
        # The vehicle physically remains in the new lane.
        # We only remove the temporary offset because the lane
        # detector should now be tracking the new lane.
        self.lane_offset_m = 0.0

        self.lane_change_active = False
        self.lane_change_stable = 0
        self.lane_change_progress = 0.0
        self.previous_lane_error = None

        self.completed_obstacle_avoids += 1

        print(
            "[LANE CHANGE] complete: %s "
            "(total=%d)"
            % (
                reason,
                self.completed_obstacle_avoids
            )
        )

    def _update_lane_change(
        self,
        lane
    ):
        if not self.lane_change_active:
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

    # ------------------------------------------------------------------
    # Parking
    # ------------------------------------------------------------------

    def _prerequisites_complete(self):
        return (
            self.hospital_done
            and self.speed40_events >= 2
            and not self.speed_zone
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        frame,
        now
    ):
        self.frame_count += 1

        signs = detect_signs_detailed(frame)
        obstacles = detect_obstacles_detailed(frame)
        lane = detect_lane(frame)

        self._update_hospital_detection(signs)
        self._update_speed_zone(signs)

        self.obstacle_tracker.update(
            obstacles,
            frame.shape[0],
            lane,
            self.lane_offset_m
        )

        if DEBUG_LOG_PATH and self.frame_count % 25 == 0:
            lane_error = (
                None
                if lane is None
                else round(float(lane["e"]), 3)
            )

            _debug_log(
                "frame=%d e=%s offset=%.3f "
                "lane_change=%s obstacles=%s hospital=%s "
                "speed40=%d speed_zone=%s parking=%s done=%s signs=%s"
                % (
                    self.frame_count,
                    lane_error,
                    self.lane_offset_m,
                    self.lane_change_active,
                    [ob["bbox"] for ob in obstacles],
                    self.hospital_state,
                    self.speed40_events,
                    self.speed_zone,
                    self.parking_started,
                    self.done,
                    [
                        (s["label"], s["bbox"])
                        for s in signs
                    ],
                )
            )

        # Hospital stop has exclusive control.
        if self.hospital_state in (
            "SETTLING",
            "HOLDING"
        ):
            self._update_hospital_hold(
                frame,
                now
            )

            self.last_steering = 0.0
            self.cap_reason = "hospital"

            return 0.0, 0.0

        # Parking has exclusive control.
        if self.parking_started:
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

        # --------------------------------------------------------------
        # Parking marker detection
        # --------------------------------------------------------------

        marker_visible = False
        marker_id = None

        if self._prerequisites_complete():
            if self.parking_detector is None:
                self.parking_detector = (
                    ParkingController._make_detector()
                )

            _, marker_id = (
                self.parking_controller._detect_marker(
                    frame,
                    self.parking_detector
                )
            )

            marker_visible = (
                marker_id is not None
            )

        if marker_visible:
            if (
                marker_id
                == self.parking_marker_candidate
            ):
                self.parking_marker_streak += 1
            else:
                self.parking_marker_candidate = (
                    marker_id
                )

                self.parking_marker_streak = 1
        else:
            self.parking_marker_candidate = None
            self.parking_marker_streak = 0

        path_clear = (
            not self.lane_change_active
            and not self._obstacle_in_path(
                obstacles,
                lane,
                frame.shape[0]
            )
        )

        if (
            self._prerequisites_complete()
            and marker_visible
            and (
                self.parking_marker_streak
                >= PARK_MARKER_STABLE_FRAMES
            )
            and path_clear
        ):
            self.parking_started = True

            self.parking_controller.target_id = (
                marker_id
            )

            print(
                "[PARKING] stable marker %s detected; "
                "parking controller engaged"
                % marker_id
            )

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

        # --------------------------------------------------------------
        # Obstacle avoidance
        # --------------------------------------------------------------

        self._maybe_start_lane_change(
            frame,
            lane,
            obstacles
        )

        self._update_lane_change(
            lane
        )

        # --------------------------------------------------------------
        # Base lane following
        # --------------------------------------------------------------

        if lane is None:
            e = 0.0
            psi = 0.0

            speed = (
                NORMAL_SPEED
                * MIN_SPEED_FRACTION
            )

            reason = "no_lane"

        else:
            poly = np.asarray(
                lane["poly"],
                dtype=float
            )

            if poly.size >= 3:
                curvature = float(
                    np.polyval(
                        np.polyder(poly, 2),
                        lane["reference_y"]
                    )
                )
            else:
                curvature = 0.0

            target_bias = (
                -CORNER_CUT_M
                * math.tanh(
                    CORNER_CURVATURE_GAIN
                    * curvature
                )
            )

            self.corner_bias_m = (
                0.80 * self.corner_bias_m
                + 0.20 * target_bias
            )

            e = (
                float(lane["e"])
                + self.lane_offset_m
                + self.corner_bias_m
            )

            psi = float(lane["psi"])

            speed_fraction = (
                1.0
                - EASE_E_GAIN * abs(e)
                - EASE_PSI_GAIN * abs(psi)
            )

            speed_fraction = float(
                np.clip(
                    speed_fraction,
                    MIN_SPEED_FRACTION,
                    1.0
                )
            )

            speed = (
                NORMAL_SPEED
                * speed_fraction
            )

            reason = "lane_follow"

        # Hospital approach
        if (
            self.hospital_state == "APPROACH"
            and speed > HOSPITAL_APPROACH_SPEED
        ):
            speed = HOSPITAL_APPROACH_SPEED
            reason = "hospital_approach"

        # Lane change
        if (
            self.lane_change_active
            and speed > LANE_CHANGE_SPEED
        ):
            speed = LANE_CHANGE_SPEED
            reason = "lane_change"

        # Obstacle proximity
        if (
            self._obstacle_in_path(
                obstacles,
                lane,
                frame.shape[0]
            )
            and speed > OBSTACLE_NEAR_SPEED
        ):
            speed = OBSTACLE_NEAR_SPEED
            reason = "obstacle_near"

        # If both the current path and the intended adjacent lane are
        # blocked, stop rather than creeping into the obstacle.
        if self.target_lane_waiting:
            speed = 0.0
            reason = "target_lane_blocked"

        # Speed 40 legal cap
        if self.speed_zone:
            if speed > SPEED_40_LIMIT:
                speed = SPEED_40_LIMIT
                reason = "speed_zone"

        speed = float(
            np.clip(
                speed,
                0.0,
                ABSOLUTE_SPEED_LIMIT
            )
        )

        self.cap_reason = reason

        # --------------------------------------------------------------
        # Stanley steering
        # --------------------------------------------------------------

        longitudinal_speed = max(
            speed * WHEEL_RADIUS,
            0.05
        )

        desired_delta = (
            psi
            + math.atan2(
                K_STANLEY * e,
                longitudinal_speed + K_SOFT
            )
        )

        desired_delta = float(
            np.clip(
                desired_delta,
                -STEER_LIMIT,
                STEER_LIMIT
            )
        )

        delta = (
            STEER_SMOOTH
            * self.last_steering
            + (
                1.0
                - STEER_SMOOTH
            )
            * desired_delta
        )

        self.last_steering = float(
            np.clip(
                delta,
                -STEER_LIMIT,
                STEER_LIMIT
            )
        )

        return (
            self.last_steering,
            speed
        )


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    print(
        "Niti Vahan Task 3 - Team %s"
        % TEAM_ID
    )

    print(
        "Waiting for the evaluation tool "
        "to start the simulation..."
    )

    client, sim, handles = connect_and_locate()

    controller = Task3Controller()

    deadline = (
        time.monotonic() + 120.0
    )

    while (
        sim.getSimulationState()
        != sim.simulation_advancing_running
    ):
        if time.monotonic() > deadline:
            print(
                "Simulation did not start within "
                "120 real seconds; exiting."
            )

            return 1

        time.sleep(0.05)

    print(
        "Simulation started. "
        "Task 3 controller active."
    )

    frames = 0
    parked_settle_start = None

    time_by_reason = {}
    prev_now = float(sim.getSimulationTime())

    try:
        while (
            sim.getSimulationState()
            == sim.simulation_advancing_running
        ):
            # One simulation-time read per control-loop iteration. Pass this
            # timestamp into every sub-controller that needs simulation time.
            now = float(sim.getSimulationTime())
            _SIM_NOW[0] = now

            if parked_settle_start is not None:
                drive(
                    sim,
                    handles,
                    0.0,
                    0.0
                )

                if (
                    now - parked_settle_start
                    >= PARK_FINAL_SETTLE_SECONDS
                ):
                    print(
                        "[DONE] Final stop hold complete; "
                        "exiting controller."
                    )
                    break

                time.sleep(0.001)
                continue

            frame = read_frame(
                sim,
                handles
            )

            if frame is None:
                continue

            if frames == 0:
                frame_h, frame_w = frame.shape[:2]

                if (frame_w, frame_h) != (IMG_W, IMG_H):
                    raise RuntimeError(
                        "Unexpected camera resolution: %dx%d; "
                        "expected %dx%d because the lane and parking "
                        "calibration is resolution-specific"
                        % (
                            frame_w,
                            frame_h,
                            IMG_W,
                            IMG_H
                        )
                    )

                print(
                    "[CAMERA] resolution verified: %dx%d"
                    % (frame_w, frame_h)
                )

            frames += 1

            steering, speed = controller.step(
                frame,
                now
            )

            if controller.hospital_state in (
                "SETTLING",
                "HOLDING"
            ):
                why = "hospital"

            elif controller.parking_started:
                why = "parking"

            else:
                why = controller.cap_reason

            time_by_reason[why] = (
                time_by_reason.get(why, 0.0)
                + max(0.0, now - prev_now)
            )

            controller.travel_rad += (
                speed
                * max(0.0, now - prev_now)
            )

            prev_now = now

            drive(
                sim,
                handles,
                steering,
                speed
            )

            if controller.done:
                parked_settle_start = now

                drive(
                    sim,
                    handles,
                    0.0,
                    0.0
                )

                print(
                    "[DONE] Parking position reached; "
                    "holding zero motor speed for "
                    "%.1f simulation seconds."
                    % PARK_FINAL_SETTLE_SECONDS
                )

    except KeyboardInterrupt:
        print(
            "Interrupted by user."
        )

    except Exception as err:
        print(
            "Task 3 loop ended: %s: %s"
            % (
                type(err).__name__,
                err
            )
        )

    finally:
        try:
            drive(
                sim,
                handles,
                0.0,
                0.0
            )

        except Exception:
            pass

        client = None

    print(
        "Sim seconds by speed reason: %s"
        % (
            ", ".join(
                "%s=%.1f" % kv
                for kv in sorted(
                    time_by_reason.items(),
                    key=lambda kv: -kv[1]
                )
            ),
        )
    )

    print(
        "Frames: %d | Hospital: %s | "
        "Speed40 signs: %d | "
        "Road obstacles cleared: %d | "
        "Completed obstacle avoidances: %d | "
        "Parking started: %s | Parked: %s"
        % (
            frames,
            controller.hospital_state,
            controller.speed40_events,
            controller.obstacle_tracker.passed,
            controller.completed_obstacle_avoids,
            controller.parking_started,
            controller.done,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
