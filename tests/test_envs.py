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

# basic empty map with walls
TEST_MAP_1 = np.array(
    [['@'] * 7,
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] * 7]
)

import unittest


class TestHarvestEnv(unittest.TestCase):
    def setUp(self):
        """Construct the env"""
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)

    def test_step(self):
        """Just check that the step method works at all for all possible actions"""
        self.env.reset()
        # FIXME(ev) magic number
        for i in range(8):
            self.env.step({'agent-0': i})

    def test_reset(self):
        self.env.reset()

    def test_walls(self):
        """Check that the spawned map and base map have walls in the right place"""
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map[0, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.base_map[-1, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.base_map[:, 0], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.base_map[:, -1], np.array(['@'] * 6))

        np.testing.assert_array_equal(self.env.map[0, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.map[-1, :], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.map[:, 0], np.array(['@'] * 6))
        np.testing.assert_array_equal(self.env.map[:, -1], np.array(['@'] * 6))

    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        self.env.reset()

        # overwrite the map
        agent_id = 'agent-0'
        self.env.map = TEST_MAP_1
        self.clear_agents()

        # TODO(ev) It seems like this map might be transposed...

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, [3, 3], 'LEFT', self.env, 2)
        self.move_agent(agent_id, [3, 3])

        # check if the view is correct if there are no walls
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 5,
             [' '] * 5,
             [' '] * 2 + ['P'] + [' '] * 2,
             [' '] * 5,
             [' '] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if the view is correct if the top wall is just in view
        self.move_agent(agent_id, [2, 3])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [['@'] * 5,
             [' '] * 5,
             [' '] * 2 + ['P'] + [' '] * 2,
             [' '] * 5,
             [' '] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the top view
        self.move_agent(agent_id, [1, 3])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[''] * 5,
             ['@'] * 5,
             [' '] * 2 + ['P'] + [' '] * 2,
             [' '] * 5,
             [' '] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if the view is correct if the left wall is just in view
        self.move_agent(agent_id, [3, 2])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [['@'] + [' '] * 4,
             ['@'] + [' '] * 4,
             ['@'] + [' '] + ['P'] + [' '] * 2,
             ['@'] + [' '] * 4,
             ['@'] + [' '] * 4]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the left view
        self.move_agent(agent_id, [3, 1])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[''] + ['@'] + [' '] * 3,
             [''] + ['@'] + [' '] * 3,
             [''] + ['@'] + ['P'] + [' '] * 2,
             [''] + ['@'] + [' '] * 3,
             [''] + ['@'] + [' '] * 3]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if the view is correct if the bot wall is just in view
        self.move_agent(agent_id, [4, 3])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 5,
             [' '] * 5,
             [' '] * 2 + ['P'] + [' '] * 2,
             [' '] * 5,
             ['@'] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the bot view
        self.move_agent(agent_id, [5, 3])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 5,
             [' '] * 5,
             [' '] * 2 + ['P'] + [' '] * 2,
             ['@'] * 5,
             [''] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if the view is correct if the right wall is just in view
        self.move_agent(agent_id, [3, 4])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 4 + ['@'],
             [' '] * 4 + ['@'],
             [' '] * 2 + ['P'] + [' '] + ['@'],
             [' '] * 4 + ['@'],
             [' '] * 4 + ['@']]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the right view
        self.move_agent(agent_id, [3, 4])
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 3 + ['@'] + [''],
             [' '] * 3 + ['@'] + [''],
             [' '] * 2 + ['P'] + ['@'] + [''],
             [' '] * 3 + ['@'] + [''],
             [' '] * 3 + ['@'] + ['']]
        )
        np.testing.assert_array_equal(expected_view, agent_view)


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

    def add_agent(self, agent_id, start_pos, start_orientation, grid, view_len):
        self.env.agents[agent_id] = HarvestAgent(agent_id, start_pos, start_orientation,
                                                 grid, view_len)

    def move_agent(self, agent_id, new_pos):
        new_pos = self.env.update_map_agent_pos(self.env.agents[agent_id].get_pos(), new_pos)
        self.env.agents[agent_id].set_pos(new_pos)


if __name__ == '__main__':
    unittest.main()
