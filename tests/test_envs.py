'''Unit tests for all of the envs'''

import numpy as np
import unittest

from gym.spaces import Discrete
from social_dilemmas.envs.agent import Agent
from social_dilemmas.envs.agent import CleanupAgent
from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.envs.agent import BASE_ACTIONS
from social_dilemmas.envs.agent import HARVEST_ACTIONS
from social_dilemmas.envs.agent import CLEANUP_ACTIONS
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.map_env import MapEnv

import utility_funcs as util

# map actions to appropriate numbers
ACTION_MAP = {y: x for x, y in BASE_ACTIONS.items()}
HARVEST_ACTION_MAP = {y: x for x, y in HARVEST_ACTIONS.items()}
CLEANUP_ACTION_MAP = {y: x for x, y in CLEANUP_ACTIONS.items()}

# Maps for any env
BASE_MAP_1 = [
    '@@@@@@@',
    '@     @',
    '@     @',
    '@     @',
    '@     @',
    '@     @',
    '@@@@@@@'
]
TEST_MAP_1 = np.array(
    [['@'] * 7,
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] + [' '] * 5 + ['@'],
     ['@'] * 7]
)

# basic empty map with no apples
BASE_MAP_2 = [
    '@@@@@@',
    '@ P  @',
    '@    @',
    '@    @',
    '@   P@',
    '@@@@@@'
]
TEST_MAP_2 = np.array(
    [['@'] * 6,
     ['@'] + [' '] * 4 + ['@'],
     ['@'] + [' '] * 4 + ['@'],
     ['@'] + [' '] * 4 + ['@'],
     ['@'] + [' '] * 2 + ['A'] + [' '] + ['@'],
     ['@'] * 6]
)

# Maps for Harvest
MINI_HARVEST_MAP = [
    '@@@@@@',
    '@ P  @',
    '@  AA@',
    '@  AA@',
    '@  AP@',
    '@@@@@@',
]

# Maps for Cleanup
MINI_CLEANUP_MAP = [
    '@@@@@@',
    '@ P  @',
    '@H BB@',
    '@R BB@',
    '@S BP@',
    '@@@@@@',
]

# Map to check that cleanup beam removes waste correctly
FIRING_CLEANUP_MAP = [
    '@@@@@@',
    '@    @',
    '@HHP @',
    '@RH  @',
    '@H P @',
    '@@@@@@',
]

# Check that apples spawn correctly in cleanup
APPLE_SPAWN_MAP_CLEANUP = [
    '@@@@@@',
    '@ P  @',
    '@  BB@',
    '@  BB@',
    '@  BP@',
    '@@@@@@',
]

# Check that the spawn probabilities are correct in cleanup
# Map to check that cleanup beam removes waste correctly
CLEANUP_PROB_MAP = [
    '@@@@@@',
    '@    @',
    '@HHPB@',
    '@RH B@',
    '@H PB@',
    '@@@@@@',
]


# maps used to test different spawn positions and apple positions


class DummyMapEnv(MapEnv):
    """This class implements a few missing methods in map env that are needed for testing."""
    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = util.return_view(map_with_agents, spawn_point,
                                    2, 2)
            agent = DummyAgent(agent_id, spawn_point, rotation, grid, 2, 2)
            self.agents[agent_id] = agent

    def append_hiddens(self, new_pos, old_char, new_char):
        self.hidden_cells.append(new_pos + [old_char])

    def execute_custom_reservations(self):
        return


class DummyAgent(Agent):
    def reward_from_pos(self, new_pos):
        return 0

    def get_done(self):
        return False

    def action_map(self, action_number):
        return BASE_ACTIONS[action_number]

    @property
    def action_space(self):
        return Discrete(len(ACTION_MAP))


class TestMapEnv(unittest.TestCase):
    def tearDown(self):
        """Remove the env"""
        self.env = None

    def test_step(self):
        """Just check that the step method works at all for all possible actions"""
        self.env = DummyMapEnv(ascii_map=BASE_MAP_2, num_agents=1)
        self.env.reset()
        agents = list(self.env.agents.values())
        action_dim = agents[0].action_space.n
        for i in range(action_dim):
            self.env.step({'agent-0': i})

    def test_walls(self):
        """Check that the spawned map and base map have walls in the right place"""
        self.env = DummyMapEnv(BASE_MAP_1, num_agents=0)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map[0, :], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.base_map[-1, :], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.base_map[:, 0], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.base_map[:, -1], np.array(['@'] * 7))

        np.testing.assert_array_equal(self.env.map[0, :], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.map[-1, :], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.map[:, 0], np.array(['@'] * 7))
        np.testing.assert_array_equal(self.env.map[:, -1], np.array(['@'] * 7))

    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        agent_id = 'agent-0'
        self.construct_map(TEST_MAP_1, agent_id, [3, 3], 'UP')

        def convert_empty_cells(view):
            """Change all empty cells marked with '0' to '' for consistency."""
            view[view == '0'] = ''
            return view

        # check if the view is correct if there are no walls
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [[' '] * 5,
             [' '] * 5,
             [' '] * 2 + ['1'] + [' '] * 2,
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
             [' '] * 2 + ['1'] + [' '] * 2,
             [' '] * 5,
             [' '] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the top view
        self.move_agent(agent_id, [1, 3])
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = convert_empty_cells(agent_view)
        expected_view = np.array(
            [[''] * 5,
             ['@'] * 5,
             [' '] * 2 + ['1'] + [' '] * 2,
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
             ['@'] + [' '] + ['1'] + [' '] * 2,
             ['@'] + [' '] * 4,
             ['@'] + [' '] * 4]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the left view
        self.move_agent(agent_id, [3, 1])
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = convert_empty_cells(agent_view)
        expected_view = np.array(
            [[''] + ['@'] + [' '] * 3,
             [''] + ['@'] + [' '] * 3,
             [''] + ['@'] + ['1'] + [' '] * 2,
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
             [' '] * 2 + ['1'] + [' '] * 2,
             [' '] * 5,
             ['@'] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the bot view
        self.move_agent(agent_id, [5, 3])
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = convert_empty_cells(agent_view)
        expected_view = np.array(
            [[' '] * 5,
             [' '] * 5,
             [' '] * 2 + ['1'] + [' '] * 2,
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
             [' '] * 2 + ['1'] + [' '] + ['@'],
             [' '] * 4 + ['@'],
             [' '] * 4 + ['@']]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the view exceeds the right view
        self.move_agent(agent_id, [3, 5])
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = convert_empty_cells(agent_view)
        expected_view = np.array(
            [[' '] * 3 + ['@'] + [''],
             [' '] * 3 + ['@'] + [''],
             [' '] * 2 + ['1'] + ['@'] + [''],
             [' '] * 3 + ['@'] + [''],
             [' '] * 3 + ['@'] + ['']]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # check if if the view is correct if the agent is in the bottom right corner
        self.move_agent(agent_id, [5, 5])
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = convert_empty_cells(agent_view)
        expected_view = np.array(
            [[' '] * 3 + ['@'] + [''],
             [' '] * 3 + ['@'] + [''],
             [' '] * 2 + ['1'] + ['@'] + [''],
             ['@'] * 4 + [''],
             [''] * 5]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

    def test_agent_actions(self):
        # set up the map
        agent_id = 'agent-0'
        self.construct_map(TEST_MAP_1.copy(), agent_id, [2, 2], 'LEFT')

        # Test that all the moves and rotations work correctly
        # test when facing left
        self.env.step({agent_id: ACTION_MAP['MOVE_LEFT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 3])
        self.env.step({agent_id: ACTION_MAP['MOVE_RIGHT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [1, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_DOWN']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing up
        self.rotate_agent(agent_id, 'UP')
        self.env.step({agent_id: ACTION_MAP['MOVE_LEFT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [1, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_RIGHT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 1])
        self.env.step({agent_id: ACTION_MAP['MOVE_DOWN']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing down
        self.rotate_agent(agent_id, 'DOWN')
        self.env.step({agent_id: ACTION_MAP['MOVE_LEFT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [3, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_RIGHT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 3])
        self.env.step({agent_id: ACTION_MAP['MOVE_DOWN']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        # test when facing right
        self.rotate_agent(agent_id, 'RIGHT')
        self.env.step({agent_id: ACTION_MAP['MOVE_LEFT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 1])
        self.env.step({agent_id: ACTION_MAP['MOVE_RIGHT']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [3, 2])
        self.env.step({agent_id: ACTION_MAP['MOVE_DOWN']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])

        # check that stay works properly
        self.env.step({agent_id: ACTION_MAP['STAY']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])
        self.assertEqual(self.env.map[2, 2], 'P')

        # quick test of stay
        self.env.step({agent_id: ACTION_MAP['STAY']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 2])

        # if an agent tries to move through a wall they should stay in the same place
        self.rotate_agent(agent_id, 'UP')
        self.move_agent(agent_id, [2, 1])
        self.env.step({agent_id: ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents[agent_id].get_pos(), [2, 1])

        # rotations correctly update the agent state
        self.rotate_agent(agent_id, 'UP')
        # clockwise
        self.env.step({agent_id: ACTION_MAP['TURN_CLOCKWISE']})
        self.assertEqual('RIGHT', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_CLOCKWISE']})
        self.assertEqual('DOWN', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_CLOCKWISE']})
        self.assertEqual('LEFT', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_CLOCKWISE']})
        self.assertEqual('UP', self.env.agents[agent_id].get_orientation())

        # counterclockwise
        self.env.step({agent_id: ACTION_MAP['TURN_COUNTERCLOCKWISE']})
        self.assertEqual('LEFT', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_COUNTERCLOCKWISE']})
        self.assertEqual('DOWN', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_COUNTERCLOCKWISE']})
        self.assertEqual('RIGHT', self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP['TURN_COUNTERCLOCKWISE']})
        self.assertEqual('UP', self.env.agents[agent_id].get_orientation())

    def test_agent_conflict(self):
        '''Test that agent conflicts are correctly resolved'''

        # test that if there are two agents and two spawning points, they hit both of them
        self.env = DummyMapEnv(ascii_map=BASE_MAP_2, num_agents=2)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map, self.env.map)

        # test that agents can't walk into other agents
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [3, 4])
        self.rotate_agent('agent-0', 'UP')
        self.rotate_agent('agent-1', 'UP')
        self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN']})
        self.env.step({'agent-1': ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents['agent-0'].get_pos(), [3, 3])
        np.testing.assert_array_equal(self.env.agents['agent-1'].get_pos(), [3, 4])

        # test that agents can't walk through each other
        self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN'],
                       'agent-1': ACTION_MAP['MOVE_UP']})
        np.testing.assert_array_equal(self.env.agents['agent-0'].get_pos(), [3, 3])
        np.testing.assert_array_equal(self.env.agents['agent-1'].get_pos(), [3, 4])
        # also check that the map looks correct, no agent has disappeared
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'P', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

        # test that agents can walk into other agents if moves are de-conflicting
        # conflict only occurs stochastically so try it 50 times
        # TODO(ev) the percentages are consistent among agents
        # TODO(ev) but which agent gets which percent is not deterministic..
        np.random.seed(1)
        self.env.agents['agent-0'].update_agent_rot('UP')
        self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN']})
        for i in range(100):
            self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN'],
                           'agent-1': ACTION_MAP['MOVE_LEFT']})
            expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', ' ', ' ', ' ', 'P', '@'],
                                     ['@', ' ', ' ', ' ', 'P', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', '@', '@', '@', '@', '@']])
            np.testing.assert_array_equal(expected_map, self.env.map)
            self.env.step({'agent-0': ACTION_MAP['MOVE_UP'],
                           'agent-1': ACTION_MAP['MOVE_RIGHT']})

        # test that if two agents have a conflicting move then the tie is broken randomly
        num_agent_1 = 0.0
        num_agent_2 = 0.0
        for i in range(100):
            self.move_agent('agent-0', [3, 2])
            self.move_agent('agent-1', [3, 4])
            self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN'],
                           'agent-1': ACTION_MAP['MOVE_UP']})
            if self.env.agents['agent-0'].get_pos().tolist() == [3, 3]:
                num_agent_1 += 1
            else:
                num_agent_2 += 1
            # Also check that the map looks correct
            expect_1 = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', 'P', 'P', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
            expect_2 = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'P', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
            equal_1 = np.array_equal(self.env.map, expect_1)
            equal_2 = np.array_equal(self.env.map, expect_2)
            self.assertTrue(equal_1 or equal_2)
        agent_1_percent = num_agent_1 / (num_agent_1 + num_agent_2)
        with_expected_val = (.53 == agent_1_percent) or (.47 == agent_1_percent)
        self.assertTrue(with_expected_val)

        # check that this works correctly with three agents
        self.add_agent('agent-2', [2, 3], 'UP', self.env, 3)
        num_agent_1 = 0.0
        other_agents = 0.0
        for i in range(100):
            self.move_agent('agent-0', [3, 2])
            self.move_agent('agent-1', [3, 4])
            self.move_agent('agent-2', [2, 3])
            self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN'],
                           'agent-1': ACTION_MAP['MOVE_UP'],
                           'agent-2': ACTION_MAP['MOVE_RIGHT']})
            if self.env.agents['agent-2'].get_pos().tolist() == [3, 3]:
                num_agent_1 += 1
            else:
                other_agents += 1
            # Also check that the map looks correct
            expect_1 = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'P', ' ', '@'],
                                 ['@', ' ', 'P', 'P', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
            expect_2 = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', 'P', 'P', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
            expect_3 = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'P', ' ', '@'],
                                 ['@', ' ', ' ', 'P', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
            equal_1 = np.array_equal(self.env.map, expect_1)
            equal_2 = np.array_equal(self.env.map, expect_2)
            equal_3 = np.array_equal(self.env.map, expect_3)
            self.assertTrue(equal_1 or equal_2 or equal_3)
        agent_1_percent = num_agent_1 / (num_agent_1 + other_agents)
        within_bounds = (agent_1_percent > .27) and (agent_1_percent < .39)
        self.assertTrue(within_bounds)

        # you try to move into an agent that is in conflict with another agent
        # fifty percent of the time you should succeed
        percent_accomplished = 0
        percent_failed = 0
        for i in range(100):
            self.move_agent('agent-1', [3, 4])
            self.move_agent('agent-2', [2, 2])
            self.move_agent('agent-0', [3, 2])
            self.env.step({'agent-0': ACTION_MAP['MOVE_DOWN'],
                           'agent-1': ACTION_MAP['MOVE_UP'],
                           'agent-2': ACTION_MAP['MOVE_RIGHT']})
            if self.env.agents['agent-2'].get_pos().tolist() == [2, 2]:
                percent_failed += 1
                expect_1 = np.array([['@', '@', '@', '@', '@', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', ' ', 'P', ' ', ' ', '@'],
                                     ['@', ' ', 'P', 'P', ' ', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', '@', '@', '@', '@', '@']])
                np.testing.assert_array_equal(expect_1, self.env.map)
            else:
                percent_accomplished += 1
                expect_1 = np.array([['@', '@', '@', '@', '@', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', ' ', 'P', 'P', 'P', '@'],
                                     ['@', ' ', ' ', ' ', ' ', '@'],
                                     ['@', '@', '@', '@', '@', '@']])
                np.testing.assert_array_equal(expect_1, self.env.map)
        percent_success = percent_accomplished / (percent_accomplished + percent_failed)
        within_bounds = (.40 < percent_success) and (percent_success < .60)
        self.assertTrue(within_bounds)

        # Check that if there is more than one conflict simultaneously
        # that it is handled correctly
        agent_0_percent = 0
        agent_1_percent = 0
        num_trials = 100
        self.add_agent('agent-3', [1, 4], 'UP', self.env, 3)
        for i in range(num_trials):
            self.move_agent('agent-1', [3, 4])
            self.move_agent('agent-2', [1, 2])
            self.move_agent('agent-0', [3, 2])
            self.move_agent('agent-3', [1, 4])
            self.env.step({'agent-0': ACTION_MAP['MOVE_LEFT'],
                           'agent-2': ACTION_MAP['MOVE_RIGHT'],
                           'agent-1': ACTION_MAP['MOVE_LEFT'],
                           'agent-3': ACTION_MAP['MOVE_RIGHT']})
            if self.env.agents['agent-0'].get_pos().tolist() == [2, 2]:
                agent_0_percent += 1
            if self.env.agents['agent-1'].get_pos().tolist() == [2, 4]:
                agent_1_percent += 1
        agent_0_success = agent_0_percent/num_trials
        agent_1_success = agent_1_percent/num_trials
        within_bounds_0 = (.4 < agent_0_success) and (agent_0_success < .6)
        within_bounds_1 = (.4 < agent_1_success) and (agent_1_success < .6)
        self.assertTrue(within_bounds_0)
        self.assertTrue(within_bounds_1)

        # agent 3 wants to move into space [3,2] as does agent-2
        # however, agent-1 wants to move into [3,3] so technically
        # no move is possible and no agent should move
        self.move_agent('agent-0', [3, 2])
        self.move_agent('agent-2', [2, 2])
        self.move_agent('agent-1', [2, 3])
        self.move_agent('agent-3', [3, 3])
        self.env.step({'agent-0': ACTION_MAP['MOVE_LEFT'],
                       'agent-1': ACTION_MAP['MOVE_RIGHT'],
                       'agent-2': ACTION_MAP['MOVE_RIGHT'],
                       'agent-3': ACTION_MAP['MOVE_UP']})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', 'P', 'P', ' ', '@'],
                                 ['@', ' ', 'P', 'P', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def move_agent(self, agent_id, new_pos):
        self.env.reserved_slots.append([new_pos[0], new_pos[1], 'P', agent_id])
        self.env.execute_reservations()
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.grid = util.return_view(map_with_agents, agent.pos,
                                      agent.row_size, agent.col_size)

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = DummyMapEnv(map, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        # FIXME(ev) hack for now to make agents appear
        char = self.env.map[start_pos[0], start_pos[1]]
        self.env.hidden_cells.append([start_pos[0], start_pos[1], char])
        self.env.map[start_pos[0], start_pos[1]] = 'P'
        map_with_agents = env.get_map_with_agents()
        grid = util.return_view(map_with_agents, start_pos, view_len, view_len)
        self.env.agents[agent_id] = DummyAgent(agent_id, start_pos, start_orientation,
                                               grid, view_len, view_len)
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.grid = util.return_view(map_with_agents, agent.pos,
                                          agent.row_size, agent.col_size)


class TestHarvestEnv(unittest.TestCase):

    def tearDown(self):
        """Remove the env"""
        self.env = None

    def test_step(self):
        """Just check that the step method works at all for all possible actions"""
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)
        self.env.reset()
        agents = list(self.env.agents.values())
        action_dim = agents[0].action_space.n
        for i in range(action_dim):
            self.env.step({'agent-0': i})

    def test_reset(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=0)
        self.env.reset()
        # check that the map is full of apples
        test_map = np.array([['@', '@', '@', '@', '@', '@'],
                             ['@', ' ', ' ', ' ', ' ', '@'],
                             ['@', ' ', ' ', 'A', 'A', '@'],
                             ['@', ' ', ' ', 'A', 'A', '@'],
                             ['@', ' ', ' ', 'A', ' ', '@'],
                             ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(self.env.map, test_map)

    def test_apple_spawn(self):
        # render apples a bunch of times and check that the probabilities are within
        # a bound of what you expect. This test fill fail w/ <INSERT> probability
        self.env = HarvestEnv(MINI_HARVEST_MAP, num_agents=0)
        self.env.reset()
        self.env.map = TEST_MAP_2.copy()

        # First test, if we step 300 times, are there five apples there?
        # This should fail maybe one in 1000000 times
        for i in range(300):
            self.env.step({})
        num_apples = self.env.count_apples(self.env.map)
        self.assertEqual(num_apples, 5)

        # Now, if a point is temporarily obscured by a beam but an apple should spawn there
        # check that the apple still spawns there
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()
        self.move_agent('agent-0', [3, 1])
        self.move_agent('agent-1', [3, 3])
        self.rotate_agent('agent-0', 'UP')
        self.rotate_agent('agent-1', 'UP')
        self.env.step({'agent-1': HARVEST_ACTION_MAP['FIRE']})
        self.env.update_map_apples([[2, 1]])
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'A', ' ', 'A', 'A', '@'],
                                 ['@', 'P', ' ', 'P', 'A', '@'],
                                 ['@', ' ', ' ', 'A', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

        # If an agent is temporarily obscured by a beam, and an apple attempts to spawn there
        # no apple should spawn
        self.env.step({'agent-1': HARVEST_ACTION_MAP['FIRE']})
        self.env.update_map_apples([[3, 1]])
        self.env.step({})

        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'A', ' ', 'A', 'A', '@'],
                                 ['@', 'P', ' ', 'P', 'A', '@'],
                                 ['@', ' ', ' ', 'A', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def test_agent_actions(self):
        # set up the map
        agent_id = 'agent-0'
        self.construct_map(TEST_MAP_1.copy(), agent_id, [2, 2], 'LEFT')
        # test firing
        self.rotate_agent(agent_id, 'UP')
        self.move_agent(agent_id, [3, 2])
        self.env.step({agent_id: HARVEST_ACTION_MAP['FIRE']})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [['@'] + [' '] * 4,
             ['@'] + ['F'] * 2 + [' '] * 2,
             ['@'] + ['F'] + ['1'] + [' '] * 2,
             ['@'] + ['F'] * 2 + [' '] * 2,
             ['@'] + [' '] * 4]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # clean up the map and then check if firing looks right
        self.env.step({})
        self.rotate_agent(agent_id, 'DOWN')
        self.move_agent(agent_id, [3, 2])
        self.env.step({agent_id: HARVEST_ACTION_MAP['FIRE']})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [['@'] + [' '] * 4,
             ['@'] + [' '] + ['F'] * 3,
             ['@'] + [' '] + ['1'] + ['F'] * 2,
             ['@'] + [' '] + ['F'] * 3,
             ['@'] + [' '] * 4]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # Check that agents walking over apples makes them go away
        np.random.seed(10)
        self.construct_map(MINI_HARVEST_MAP.copy(), agent_id, [3, 2], 'RIGHT')
        self.env.step({agent_id: HARVEST_ACTION_MAP['MOVE_RIGHT']})
        self.env.step({agent_id: HARVEST_ACTION_MAP['MOVE_LEFT']})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [['@', ' ', ' ', ' ', ' '],
             ['@', ' ', ' ', 'A', 'A'],
             ['@', ' ', '1', ' ', 'A'],
             ['@', ' ', ' ', 'A', ' '],
             ['@', '@', '@', '@', '@']]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

    def test_agent_rewards(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()
        self.move_agent('agent-0', [2, 2])
        self.move_agent('agent-1', [3, 2])
        self.rotate_agent('agent-0', 'UP')
        self.rotate_agent('agent-1', 'UP')
        # walk over an apple
        _, rew, _, _ = self.env.step({'agent-0': HARVEST_ACTION_MAP['MOVE_DOWN'],
                                      'agent-1': HARVEST_ACTION_MAP['MOVE_DOWN']})
        self.assertTrue(rew['agent-0'] == 1)
        self.assertTrue(rew['agent-1'] == 1)
        # fire a beam from agent 1 to 2
        self.rotate_agent('agent-1', 'LEFT')
        _, rew, _, _ = self.env.step({'agent-1': HARVEST_ACTION_MAP['FIRE']})
        self.assertTrue(rew['agent-0'] == -50)
        self.assertTrue(rew['agent-1'] == -1)

    def test_agent_conflict(self):
        '''Test that agent conflicts are correctly resolved'''

        # test that if there are two agents and two spawning points, they hit both of them
        self.env = HarvestEnv(ascii_map=BASE_MAP_2, num_agents=2)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map, self.env.map)

        # test that agents can't walk into other agents
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [3, 4])
        self.rotate_agent('agent-0', 'UP')
        self.rotate_agent('agent-1', 'UP')

        # test that if an agents firing beam hits another agent it gets covered
        self.env.step({'agent-0': HARVEST_ACTION_MAP['MOVE_UP']})
        self.env.step({'agent-1': HARVEST_ACTION_MAP['FIRE']})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'F', 'F', 'F', 'F', '@'],
                                 ['@', 'F', 'F', 'F', 'P', '@'],
                                 ['@', 'F', 'F', 'F', 'F', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)
        # but by the next step, the agent is visible again
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', 'P', ' ', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

        # test that if two agents fire on each other than they're still there after
        self.env.agents['agent-0'].update_agent_rot('DOWN')
        self.env.step({'agent-0': HARVEST_ACTION_MAP['FIRE'],
                       'agent-1': HARVEST_ACTION_MAP['FIRE']})
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', 'P', ' ', 'P', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def test_beam_conflict(self):
        """Test that after the beam is fired, obscured apples and agents are returned"""
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()

        # test that agents can't walk into other agents
        self.move_agent('agent-0', [4, 2])
        self.move_agent('agent-1', [4, 4])
        self.env.agents['agent-0'].update_agent_rot('UP')
        self.env.agents['agent-1'].update_agent_rot('UP')
        # test that if an agents firing beam hits another agent it gets covered
        self.env.step({'agent-1': HARVEST_ACTION_MAP['FIRE']})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'A', 'A', '@'],
                                 ['@', 'F', 'F', 'F', 'F', '@'],
                                 ['@', 'F', 'F', 'F', 'P', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)
        # test that by the next step it will be returned
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', ' ', ' ', 'A', 'A', '@'],
                                 ['@', ' ', ' ', 'A', 'A', '@'],
                                 ['@', ' ', 'P', 'A', 'P', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def clear_agents(self):
        # FIXME(ev) this doesn't clear agent positions off the board
        self.env.agents = {}

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        # FIXME(ev) hack for now to make agents appear
        char = self.env.map[start_pos[0], start_pos[1]]
        self.env.hidden_cells.append([start_pos[0], start_pos[1], char])
        self.env.map[start_pos[0], start_pos[1]] = 'P'
        map_with_agents = env.get_map_with_agents()
        grid = util.return_view(map_with_agents, start_pos, view_len, view_len)
        self.env.agents[agent_id] = HarvestAgent(agent_id, start_pos, start_orientation,
                                                 grid, view_len)
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.grid = util.return_view(map_with_agents, agent.pos,
                                          agent.row_size, agent.col_size)

    def move_agent(self, agent_id, new_pos):
        self.env.reserved_slots.append([new_pos[0], new_pos[1], 'P', agent_id])
        self.env.execute_reservations()
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.grid = util.return_view(map_with_agents, agent.pos,
                                      agent.row_size, agent.col_size)

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = HarvestEnv(map, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)


class TestCleanupEnv(unittest.TestCase):
    def test_parameters(self):
        self.env = CleanupEnv(num_agents=0)
        self.assertEqual(self.env.potential_waste_area, 119)

    def test_reset(self):
        self.env = CleanupEnv(ascii_map=MINI_CLEANUP_MAP, num_agents=0)
        self.env.reset()
        # check that the map has no apples
        test_map = np.array([['@', '@', '@', '@', '@', '@'],
                             ['@', ' ', ' ', ' ', ' ', '@'],
                             ['@', 'H', ' ', ' ', ' ', '@'],
                             ['@', 'R', ' ', ' ', ' ', '@'],
                             ['@', 'S', ' ', ' ', ' ', '@'],
                             ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(self.env.map, test_map)

    def test_cleanup_beam(self):
        # TODO(ev) you have to disable spawning here
        self.env = CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=2)
        self.env.reset()
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [4, 2])
        self.rotate_agent('agent-0', 'UP')
        # check that cleanup beam does three things
        # 1. Cleans waste cells correctly
        # 2. Is blocked by the first waste cell it encounters
        # 3. Obscures agents when fired, doesn't remove them when cleaned
        self.env.step({'agent-0': CLEANUP_ACTION_MAP['CLEAN']})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'H', 'C', 'C', ' ', '@'],
                                 ['@', 'R', 'C', 'P', ' ', '@'],
                                 ['@', 'C', 'C', 'C', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)
        self.env.clean_map()
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'H', 'R', ' ', ' ', '@'],
                                 ['@', 'R', 'R', 'P', ' ', '@'],
                                 ['@', 'R', 'P', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

        # check that the cleanup beam doesn't remove apples
        self.env.reset()
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [4, 2])
        self.env.update_map_apples([[3, 4]])
        self.rotate_agent('agent-0', 'DOWN')
        self.env.step({'agent-0': CLEANUP_ACTION_MAP['CLEAN']})
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'H', 'H', ' ', ' ', '@'],
                                 ['@', 'R', 'H', 'P', 'A', '@'],
                                 ['@', 'H', 'P', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def test_firing_beam(self):
        self.env = CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=2)
        self.env.reset()
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [4, 2])
        self.rotate_agent('agent-0', 'UP')
        # check that cleanup beam does three things
        # 1. Cleans waste cells correctly
        # 2. Is blocked by the first waste cell it encounters
        # 3. Obscures agents when fired, doesn't remove them when cleaned

        self.env.step({'agent-0': CLEANUP_ACTION_MAP['FIRE']})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'F', 'F', 'F', ' ', '@'],
                                 ['@', 'F', 'F', 'P', ' ', '@'],
                                 ['@', 'F', 'F', 'F', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'H', 'H', ' ', ' ', '@'],
                                 ['@', 'R', 'H', 'P', ' ', '@'],
                                 ['@', 'H', 'P', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

        # check that the cleanup beam doesn't remove apples
        self.env.reset()
        self.move_agent('agent-0', [3, 3])
        self.move_agent('agent-1', [4, 2])
        self.env.update_map_apples([[3, 4]])
        self.rotate_agent('agent-0', 'DOWN')
        self.env.step({'agent-0': CLEANUP_ACTION_MAP['FIRE']})
        self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                 ['@', ' ', ' ', ' ', ' ', '@'],
                                 ['@', 'H', 'H', ' ', ' ', '@'],
                                 ['@', 'R', 'H', 'P', 'A', '@'],
                                 ['@', 'H', 'P', ' ', ' ', '@'],
                                 ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def test_apple_spawn(self):
        """Confirm that apples spawn correctly in cleanup"""
        self.env = CleanupEnv(ascii_map=APPLE_SPAWN_MAP_CLEANUP, num_agents=2)
        self.env.reset()
        for i in range(500):
            self.env.step({})
        expected_map = np.array([['@', '@', '@', '@', '@', '@'],
                                ['@', ' ', 'P', ' ', ' ', '@'],
                                ['@', ' ', ' ', 'A', 'A', '@'],
                                ['@', ' ', ' ', 'A', 'A', '@'],
                                ['@', ' ', ' ', 'A', 'P', '@'],
                                ['@', '@', '@', '@', '@', '@']])
        np.testing.assert_array_equal(expected_map, self.env.map)

    def test_spawn_probabilities(self):
        """Test that apple and waste spawn probabilities are set correctly"""
        self.env = CleanupEnv(ascii_map=CLEANUP_PROB_MAP, num_agents=2)
        self.env.reset()

        # Check that the permitted waste area is correct
        self.assertEqual(self.env.compute_permitted_area(), 1)

        # Check that the potential waste area is correct
        self.assertEqual(self.env.potential_waste_area, 5)

        # Check that the apple spawn probability is zero if there's too much
        # waste
        self.assertTrue(np.isclose(self.env.current_apple_spawn_prob, 0))
        # Check that the waste spawn probability is zero if there's too much
        # waste
        self.assertTrue(np.isclose(self.env.current_waste_spawn_prob, 0))

        # Check that the waste spawn probability is computed correctly
        # TODO(ev) this test depends on the answer to issue # 110
        self.rotate_agent('agent-0', 'UP')
        self.rotate_agent('agent-1', 'UP')
        self.env.step({'agent-0': CLEANUP_ACTION_MAP['CLEAN'],
                       'agent-1': CLEANUP_ACTION_MAP['CLEAN']})
        self.env.step({})
        self.assertTrue(np.isclose(self.env.current_waste_spawn_prob, 0.5))

        # TODO(ev) this test depends on the answer to issue # 110
        # check that the apple spawn probability is computed correctly
        while True:
            self.env.step({'agent-0': CLEANUP_ACTION_MAP['CLEAN'],
                           'agent-1': CLEANUP_ACTION_MAP['CLEAN']})
            self.env.step({})
            if self.env.compute_permitted_area() == 4:
                break
        # self.assertTrue(np.isclose(self.env.current_apple_spawn_prob, 0.125))

    def clear_agents(self):
        # FIXME(ev) this doesn't clear agent positions off the board
        self.env.agents = {}

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        # FIXME(ev) hack for now to make agents appear
        char = self.env.map[start_pos[0], start_pos[1]]
        self.env.hidden_cells.append([start_pos[0], start_pos[1], char])
        self.env.map[start_pos[0], start_pos[1]] = 'P'
        map_with_agents = env.get_map_with_agents()
        grid = util.return_view(map_with_agents, start_pos, view_len, view_len)
        self.env.agents[agent_id] = CleanupAgent(agent_id, start_pos, start_orientation,
                                                 grid, view_len)
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.grid = util.return_view(map_with_agents, agent.pos,
                                          agent.row_size, agent.col_size)

    def move_agent(self, agent_id, new_pos):
        self.env.reserved_slots.append([new_pos[0], new_pos[1], 'P', agent_id])
        self.env.execute_reservations()
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.grid = util.return_view(map_with_agents, agent.pos,
                                      agent.row_size, agent.col_size)

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = CleanupEnv(map, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)


if __name__ == '__main__':
    unittest.main()
