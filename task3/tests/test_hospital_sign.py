import sys
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "task3"))
import task3


class HospitalSignTests(unittest.TestCase):
    def test_hospital_signs_in_project_images(self):
        for name in ("H_1.png", "Obs_H_1.png", "Obs_H_2.png"):
            with self.subTest(name=name):
                frame = cv2.imread(str(ROOT / "task2b/public" / name))
                self.assertIsNotNone(frame)
                labels = [s["label"] for s in task3.detect_signs_detailed(frame)]
                self.assertIn("Hospital", labels)

    def test_cafe_lamp_does_not_trigger_pickup(self):
        video = cv2.VideoCapture(str(ROOT / "task1c/public/clip_12.mp4"))
        controller = task3.Task3Controller()
        frames = 0
        try:
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                frames += 1
                controller._update_hospital_detection(task3.detect_signs_detailed(frame))
                self.assertEqual(controller.hospital_state, "WAIT_FIRST", f"frame {frames-1}")
        finally:
            video.release()
        self.assertEqual(frames, 200)
