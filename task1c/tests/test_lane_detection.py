"""
Unit tests for task1c/lane_detection.py.

Run from the task1c folder:
    python -m unittest discover -s tests -v

Frames are drawn synthetically (dark asphalt, yellow road edges, white dashes),
so no video files are needed.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lane_detection as ld  # noqa: E402

ASPHALT = (18, 20, 18)       # BGR, very dark
YELLOW = (0, 220, 255)       # BGR
WHITE = (255, 255, 255)


def make_frame(divider_x=None, edges=(160, 480), dash_rows=(150, 230, 310, 390)):
    """A 640x480 road: asphalt, optional yellow edges, optional dashed divider."""
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:] = ASPHALT
    for edge_x in edges or ():
        cv2.rectangle(frame, (edge_x - 5, 0), (edge_x + 5, 480), YELLOW, -1)
    if divider_x is not None:
        for row in dash_rows:
            cv2.rectangle(frame, (divider_x - 4, row), (divider_x + 4, row + 40), WHITE, -1)
    return frame


class TestHelpers(unittest.TestCase):
    def test_x_at_reference_row_no_points(self):
        self.assertIsNone(ld._x_at_reference_row([]))

    def test_x_at_reference_row_single_point(self):
        self.assertEqual(ld._x_at_reference_row([(250, 123.0)]), 123.0)

    def test_x_at_reference_row_reads_line_at_reference_row(self):
        points = [(y, 0.5 * y + 10) for y in (150, 200, 250, 300)]
        self.assertAlmostEqual(ld._x_at_reference_row(points), 0.5 * 300 + 10, places=3)

    def test_row_band_clusters_finds_each_bar(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[280:320, 100:110] = 255
        mask[280:320, 400:420] = 255
        clusters = ld._row_band_clusters(mask, 300, 15)
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(clusters[0], 104.5, delta=1.0)
        self.assertAlmostEqual(clusters[1], 409.5, delta=1.0)

    def test_row_band_clusters_empty_mask(self):
        self.assertEqual(ld._row_band_clusters(np.zeros((480, 640), np.uint8), 300, 15), [])


class TestDashes(unittest.TestCase):
    def _dashes(self, frame):
        white, _yellow, dark = ld._lane_colour_masks(frame)
        return ld._find_dashes(white, dark)

    def test_dash_on_asphalt_is_accepted(self):
        self.assertGreaterEqual(len(self._dashes(make_frame(divider_x=300))), 3)

    def test_white_wall_is_rejected(self):
        frame = make_frame(divider_x=None, edges=())
        frame[0:200, 450:640] = (190, 190, 190)             # bright wall, not asphalt
        cv2.rectangle(frame, (500, 60), (540, 120), WHITE, -1)   # white window on it
        self.assertEqual(self._dashes(frame), [])

    def test_vehicle_chassis_is_ignored(self):
        frame = make_frame(divider_x=None, edges=())
        cv2.rectangle(frame, (240, 430), (400, 478), WHITE, -1)
        white, _yellow, _dark = ld._lane_colour_masks(frame)
        self.assertEqual(int(np.count_nonzero(white)), 0)


class TestFitDivider(unittest.TestCase):
    def test_no_dashes_gives_none(self):
        self.assertIsNone(ld._fit_divider([]))

    def test_straight_divider_is_read_at_reference_row(self):
        frame = make_frame(divider_x=380)
        white, _yellow, dark = ld._lane_colour_masks(frame)
        fit = ld._fit_divider(ld._find_dashes(white, dark))
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.x_at(300), 380, delta=3)
        self.assertTrue(fit.confident)


class TestLaneDecision(unittest.TestCase):
    def setUp(self):
        ld._memory.__init__()

    def test_divider_right_of_vehicle_is_left_lane(self):
        result = ld.detect_lane(make_frame(divider_x=380))
        self.assertEqual(result["lane"], "left")
        self.assertAlmostEqual(result["center_x"], (380 + 160) / 2, delta=4)

    def test_divider_left_of_vehicle_is_right_lane(self):
        result = ld.detect_lane(make_frame(divider_x=260))
        self.assertEqual(result["lane"], "right")
        self.assertAlmostEqual(result["center_x"], (260 + 480) / 2, delta=4)

    def test_missing_outer_edge_uses_learned_lane_width(self):
        result = ld.detect_lane(make_frame(divider_x=260, edges=()))
        self.assertEqual(result["lane"], "right")
        self.assertAlmostEqual(result["center_x"], 260 + ld._INITIAL_LANE_WIDTH_PX / 2, delta=4)

    def test_no_divider_rebuilt_from_road_edges(self):
        result = ld.detect_lane(make_frame(divider_x=None, edges=(120, 600)))
        self.assertEqual(result["lane"], "left")             # midpoint 360 is right of the car
        self.assertAlmostEqual(result["center_x"], (360 + 120) / 2, delta=4)

    def test_road_edges_too_close_are_rejected(self):
        self.assertEqual(ld.detect_lane(make_frame(divider_x=None, edges=(200, 520))),
                         {"center_x": -1, "lane": "unknown"})

    def test_blank_frame_is_unknown(self):
        self.assertEqual(ld.detect_lane(np.zeros((480, 640, 3), np.uint8)),
                         {"center_x": -1, "lane": "unknown"})


class TestFrameMemory(unittest.TestCase):
    def setUp(self):
        ld._memory.__init__()

    def test_last_result_is_reused_then_expires(self):
        good = ld.detect_lane(make_frame(divider_x=380))
        gap = make_frame(divider_x=None, edges=())            # same asphalt, no markings
        for _ in range(ld._MAX_HOLD_FRAMES):
            self.assertEqual(ld.detect_lane(gap), good)
        self.assertEqual(ld.detect_lane(gap), {"center_x": -1, "lane": "unknown"})

    def test_scene_cut_forgets_last_result(self):
        ld.detect_lane(make_frame(divider_x=380))
        self.assertIsNotNone(ld._memory.last_good)
        ld.detect_lane(np.full((480, 640, 3), 200, np.uint8))  # completely different picture
        self.assertIsNone(ld._memory.last_good)

    def test_lane_width_moves_towards_measurement(self):
        ld._memory.update_lane_width(100, 500)               # 200 px: outside the accepted range
        self.assertEqual(ld._memory.lane_width, ld._INITIAL_LANE_WIDTH_PX)
        ld._memory.update_lane_width(80, 520)                # 220 px: accepted, smoothed
        expected = 0.8 * ld._INITIAL_LANE_WIDTH_PX + 0.2 * 220
        self.assertAlmostEqual(ld._memory.lane_width, expected, places=3)


class TestRobustness(unittest.TestCase):
    def setUp(self):
        ld._memory.__init__()

    def test_odd_frames_do_not_raise(self):
        rng = np.random.default_rng(0)
        frames = [np.full((480, 640, 3), 255, np.uint8),
                  rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)]
        ld._RAISE_ERRORS = True
        try:
            for frame in frames:
                result = ld.detect_lane(frame)
                self.assertIn(result["lane"], ("left", "right", "unknown"))
                self.assertIsInstance(result["center_x"], int)
        finally:
            ld._RAISE_ERRORS = False


if __name__ == "__main__":
    unittest.main()
