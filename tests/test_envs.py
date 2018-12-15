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

        agent_id = 'agent-0'
        self.construct_map_1(agent_id, [3,3], 'UP')

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
        self.move_agent(agent_id, [3, 5])
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
        # render apples a bunch of times and check that the probabilities are within
        # a bound of what you expect?
        pass

    def test_firing(self):
        pass

    def test_agent_actions(self):
        # set up the map
        agent_id = 'agent-0'
        self.construct_map_1(agent_id, [2,2], 'LEFT')

        # Test that all the moves and rotations work correctly
        # test when facing left
        self.env.update_map({agent_id: 'MOVE_LEFT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 3])
        self.env.update_map({agent_id: 'MOVE_RIGHT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.update_map({agent_id: 'MOVE_UP'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [1, 2])
        self.env.update_map({agent_id: 'MOVE_DOWN'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing up
        self.rotate_agent(agent_id, 'UP')
        self.env.update_map({agent_id: 'MOVE_LEFT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [1, 2])
        self.env.update_map({agent_id: 'MOVE_RIGHT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.update_map({agent_id: 'MOVE_UP'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 1])
        self.env.update_map({agent_id: 'MOVE_DOWN'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing down
        self.rotate_agent(agent_id, 'DOWN')
        self.env.update_map({agent_id: 'MOVE_LEFT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [3, 2])
        self.env.update_map({agent_id: 'MOVE_RIGHT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.update_map({agent_id: 'MOVE_UP'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 3])
        self.env.update_map({agent_id: 'MOVE_DOWN'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing right
        self.rotate_agent(agent_id, 'RIGHT')
        self.env.update_map({agent_id: 'MOVE_LEFT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 1])
        self.env.update_map({agent_id: 'MOVE_RIGHT'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.update_map({agent_id: 'MOVE_UP'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [3, 2])
        self.env.update_map({agent_id: 'MOVE_DOWN'})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])


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
        self.env.agents[agent_id].update_map_agent_pos(new_pos)

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_map_agent_rot(new_rot)

    # TODO(ev) test if an agent walking into another agent that is going to move is allowed
    # TODO(ev) it should be but it isn't right now
    def test_agent_conflict(self):
        pass

    def construct_map_1(self, agent_id, start_pos, start_orientation):
        # overwrite the map
        self.env.map = TEST_MAP_1.copy()
        self.clear_agents()

        # TODO(ev) It seems like this map might be transposed...
        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)
        # TODO(ev) hack for now, can't call render logic or else it will spawn apples
        self.move_agent(agent_id, start_pos)

if __name__ == '__main__':
    unittest.main()
