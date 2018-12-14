'''Unit tests for all of the envs'''

import numpy as np

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.agent import HarvestAgent

MINI_HARVEST_MAP = [
'@@@@@@',
'@ P  @',
'@  A @',
'@ AAA@',
'@  AP@',
'@@@@@@',
]

# maps used to test different spawn positions and apple positions
TEST_MAP_1 = []


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

        np.testing.assert_array_equal(self.env.map[0,:], np.array(['@']*6))
        np.testing.assert_array_equal(self.env.map[-1, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.map[:, 0], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.map[:, -1], np.array(['@'] * 6))


    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        self.env.reset()

    def test_apple_spawn(self):
        pass

    def test_firing(self):
        pass

    def test_agent_actions(self):
        # set up the map so that we know where the agents and apples are

        # if an agent tries to move through a wall they should stay in the same place

        # if an agent moves over an apple the apple disappears

        # rotations correctly update the agent state

        # actions interact correctly with rotations
        pass

    def test_agent_rewards(self):
        pass

    def clear_agents(self):
        self.env.agents = {}

    def add_agent(self, agent_id, start_pos, start_orientation, grid):
        self.env.agents[agent_id] = HarvestAgent(agent_id, start_pos, start_orientation,
                                                 grid)

if __name__ == '__main__':
    unittest.main()
