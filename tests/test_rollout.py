import os
import unittest

from visualization.rollout import Controller


class TestRollout(unittest.TestCase):
    def setUp(self):
        class Args(object):
            pass

        args = Args()
        args.env = "cleanup"

        self.controller = Controller(args)

    def test_rollouts(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.controller.render_rollout(horizon=5, path=path)
        # cleanup
        if os.path.exists("cleanup_trajectory.mp4"):
            os.remove("cleanup_trajectory.mp4")


if __name__ == "__main__":
    unittest.main()
