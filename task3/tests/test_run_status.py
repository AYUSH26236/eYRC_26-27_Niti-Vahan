import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task3


class RunStatusTests(unittest.TestCase):
    def run_session(self, times, *, parked=False, error=None):
        sim = MagicMock()
        sim.simulation_advancing_running = 1
        sim.getSimulationState.side_effect = [1] + [1] * len(times) + [0]
        sim.getSimulationTime.side_effect = [0.0] + times
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        controller = task3.Task3Controller()
        controller.done = parked
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(task3, "connect_and_locate", return_value=(None, sim, {})))
            stack.enter_context(patch.object(task3, "Task3Controller", return_value=controller))
            camera = stack.enter_context(patch.object(task3, "read_frame", return_value=frame, side_effect=error))
            step = stack.enter_context(patch.object(controller, "step", return_value=(0.0, 0.0)))
            drive = stack.enter_context(patch.object(task3, "drive"))
            status = task3.main()
        self.assertEqual(drive.call_args.args[-2:], (0.0, 0.0))
        return status, camera.call_count, [args.args[1] for args in step.call_args_list]

    def test_multiple_polls_of_one_frame_count_once(self):
        self.assertEqual(self.run_session([0, 0, 0, 0.05]), (1, 2, [0, 0.05]))

    def test_camera_error_is_a_failed_run(self):
        self.assertEqual(self.run_session([0], error=RuntimeError("camera unavailable"))[0], 1)

    def test_interruption_is_reported(self):
        self.assertEqual(self.run_session([0], error=KeyboardInterrupt())[0], 130)

    def test_parked_without_full_hold_is_incomplete(self):
        self.assertEqual(self.run_session([0, 1], parked=True)[0], 1)

    def test_full_parking_hold_succeeds(self):
        self.assertEqual(self.run_session([0, 1, 1.5], parked=True), (0, 1, [0]))


if __name__ == "__main__":
    unittest.main()
