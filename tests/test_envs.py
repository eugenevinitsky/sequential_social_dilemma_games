'''Unit tests for all of the envs'''

import numpy as np

from social_dilemmas.envs.harvest import HarvestEnv

MINI_HARVEST_MAP = [
'@@@@@@',
'@ P  @',
'@  A @',
'@ AAA@',
'@  AP@',
'@@@@@@',
]


import unittest

class TestHarvestEnv(unittest.TestCase):
    def setUp(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)

    def test_step(self):
        self.env.reset()
        # FIXME(ev) magic number
        for i in range(8):
            self.env.step({'agent-0': i})

    def test_reset(self):
        self.env.reset()

    def test_walls(self):
        """Check that the spawned map and base map have walls in the right place"""
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map[0,:], np.array(['@']*6))
        np.testing.assert_array_equal(self.env.base_map[-1, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.base_map[:, 0], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.base_map[:, -1], np.array(['@'] * 6))


    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        pass

    def test_apple_spawn(self):
        pass

    def test_firing(self):
        pass

    def test_move_agent(self):
        # if an agent tries to move through a wall they should stay in the same place

        # if an agent moves over an apple the apple disappears
        pass

    def test_agent_rewards(self):
        pass

if __name__ == '__main__':
    unittest.main()
