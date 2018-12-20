import os
import unittest

from rollout import Controller


class TestRollout(unittest.TestCase):
    def setUp(self):
        self.controller = Controller()

    def test_rollouts(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.controller.rollout_and_render(path=path, render_full_vid=True)
        # cleanup
        if os.path.exists("trajectory.mp4"):
            os.remove("trajectory.mp4")


if __name__ == '__main__':
    unittest.main()
