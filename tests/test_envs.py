"""Unit tests for all of the envs"""

import random
import unittest

import numpy as np

from social_dilemmas.envs.agent import (
    BASE_ACTIONS,
    CLEANUP_ACTIONS,
    HARVEST_ACTIONS,
    Agent,
    CleanupAgent,
    HarvestAgent,
)
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.map_env import MapEnv

# map actions to appropriate numbers
ACTION_MAP = {y: x for x, y in BASE_ACTIONS.items()}
HARVEST_ACTION_MAP = {y: x for x, y in HARVEST_ACTIONS.items()}
CLEANUP_ACTION_MAP = {y: x for x, y in CLEANUP_ACTIONS.items()}

# Maps for any env
BASE_MAP_1 = [
    "@@@@@@@",
    "@     @",
    "@     @",
    "@     @",
    "@     @",
    "@     @",
    "@@@@@@@",
]
TEST_MAP_1 = np.array(
    [
        [b"@"] * 7,
        [b"@"] + [b" "] * 5 + [b"@"],
        [b"@"] + [b" "] * 5 + [b"@"],
        [b"@"] + [b" "] * 5 + [b"@"],
        [b"@"] + [b" "] * 5 + [b"@"],
        [b"@"] + [b" "] * 5 + [b"@"],
        [b"@"] * 7,
    ]
)

FIRE_RANGE_MAP = np.array(
    [
        [b"@"] * 13,
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] + [b" "] * 11 + [b"@"],
        [b"@"] * 13,
    ]
)

# basic empty map with no apples
BASE_MAP_2 = ["@@@@@@", "@ P  @", "@    @", "@    @", "@   P@", "@@@@@@"]
TEST_MAP_2 = np.array(
    [
        [b"@"] * 6,
        [b"@"] + [b" "] * 4 + [b"@"],
        [b"@"] + [b" "] * 4 + [b"@"],
        [b"@"] + [b" "] * 4 + [b"@"],
        [b"@"] + [b" "] * 2 + [b"A"] + [b" "] + [b"@"],
        [b"@"] * 6,
    ]
)

# Maps for Harvest
MINI_HARVEST_MAP = [
    "@@@@@@",
    "@ P  @",
    "@  AA@",
    "@  AA@",
    "@  AP@",
    "@@@@@@",
]

# Maps for Cleanup
MINI_CLEANUP_MAP = [
    "@@@@@@",
    "@ P  @",
    "@H BB@",
    "@R BB@",
    "@S BP@",
    "@@@@@@",
]

# Map to check that cleanup beam removes waste correctly
FIRING_CLEANUP_MAP = [
    "@@@@@@",
    "@    @",
    "@HHP @",
    "@RH  @",
    "@H P @",
    "@@@@@@",
]

# Check that apples spawn correctly in cleanup
APPLE_SPAWN_MAP_CLEANUP = [
    "@@@@@@",
    "@ P  @",
    "@  BB@",
    "@  BB@",
    "@  BP@",
    "@@@@@@",
]

# Check that the spawn probabilities are correct in cleanup
# Map to check that cleanup beam removes waste correctly
CLEANUP_PROB_MAP = [
    "@@@@@@",
    "@    @",
    "@HHPB@",
    "@RH B@",
    "@H PB@",
    "@@@@@@",
]


def get_env_test_map(env):
    """Gets a version of the environment map where generic
    'P' characters have been replaced with specific agent IDs.

    Returns:
        2D array of strings representing the map.
    """
    grid = np.copy(env.world_map)

    for agent_id, agent in env.agents.items():
        # If agent is not within map, skip.
        if not (0 <= agent.pos[0] < grid.shape[0] and 0 <= agent.pos[1] < grid.shape[1]):
            continue

        grid[agent.pos[0], agent.pos[1]] = b"P"

    for beam_pos in env.beam_pos:
        grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

    return grid


class DummyMapEnv(MapEnv):
    """This class implements a few missing methods in map env that are needed for testing."""

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         2, 2)
            agent = DummyAgent(agent_id, spawn_point, rotation, grid, 2, 2)
            self.agents[agent_id] = agent

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
        return DiscreteWithDType(len(ACTION_MAP), dtype=np.uint8)

    def consume(self, char):
        return char


class TestMapEnv(unittest.TestCase):
    def tearDown(self):
        """Remove the env"""
        self.env = None

    def test_step(self):
        """Just check that the step method works at all for all possible actions"""
        self.env = DummyMapEnv(ascii_map=BASE_MAP_2, extra_actions={}, view_len=2, num_agents=1)
        self.env.reset()
        agents = list(self.env.agents.values())
        action_dim = agents[0].action_space.n
        for i in range(action_dim):
            self.env.step({"agent-0": i})

    def test_walls(self):
        """Check that the spawned map and base map have walls in the right place"""
        self.env = DummyMapEnv(BASE_MAP_1, extra_actions={}, view_len=2, num_agents=0)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map[0, :], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.base_map[-1, :], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.base_map[:, 0], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.base_map[:, -1], np.array([b"@"] * 7))

        np.testing.assert_array_equal(self.env.world_map[0, :], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.world_map[-1, :], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.world_map[:, 0], np.array([b"@"] * 7))
        np.testing.assert_array_equal(self.env.world_map[:, -1], np.array([b"@"] * 7))

    def assert_logical_and_color_view(self, agent_id, expected_view):
        agent_view = self.env.agents[agent_id].get_state()
        agent_view = self.convert_empty_cells(agent_view)
        agent_view_color = self.env.color_view(self.env.agents[agent_id])
        expected_view_color = self.env.map_to_colors(
            expected_view,
            self.env.color_map,
            np.zeros(shape=(*expected_view.shape, 3), dtype=np.uint8),
            self.env.agents[agent_id].orientation,
        )
        np.testing.assert_array_equal(expected_view, agent_view)
        np.testing.assert_array_equal(expected_view_color, agent_view_color)

    @staticmethod
    def convert_empty_cells(view):
        """Change all empty cells marked with '0' to ' ' for consistency."""
        # No mask because it doesn't work correctly on byte arrays
        for x in range(len(view)):
            for y in range(len(view[0])):
                view[x, y] = b" " if view[x, y] == b"0" else view[x, y]
        return view

    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        agent_id = "agent-0"
        self.construct_map(TEST_MAP_1, agent_id, [3, 3], "UP")

        # check if the view is correct if there are no walls
        expected_view = np.array(
            [[b" "] * 5, [b" "] * 5, [b" "] * 2 + [b"1"] + [b" "] * 2, [b" "] * 5, [b" "] * 5]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if the view is correct if the top wall is just in view
        self.move_agent(agent_id, [2, 3])
        expected_view = np.array(
            [[b"@"] * 5, [b" "] * 5, [b" "] * 2 + [b"1"] + [b" "] * 2, [b" "] * 5, [b" "] * 5]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if if the view is correct if the view exceeds the top view
        self.move_agent(agent_id, [1, 3])
        expected_view = np.array(
            [[b" "] * 5, [b"@"] * 5, [b" "] * 2 + [b"1"] + [b" "] * 2, [b" "] * 5, [b" "] * 5]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if the view is correct if the left wall is just in view
        self.move_agent(agent_id, [3, 2])
        expected_view = np.array(
            [
                [b"@"] + [b" "] * 4,
                [b"@"] + [b" "] * 4,
                [b"@"] + [b" "] + [b"1"] + [b" "] * 2,
                [b"@"] + [b" "] * 4,
                [b"@"] + [b" "] * 4,
            ]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if if the view is correct if the view exceeds the left view
        self.move_agent(agent_id, [3, 1])
        expected_view = np.array(
            [
                [b" "] + [b"@"] + [b" "] * 3,
                [b" "] + [b"@"] + [b" "] * 3,
                [b" "] + [b"@"] + [b"1"] + [b" "] * 2,
                [b" "] + [b"@"] + [b" "] * 3,
                [b" "] + [b"@"] + [b" "] * 3,
            ]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if the view is correct if the bot wall is just in view
        self.move_agent(agent_id, [4, 3])
        expected_view = np.array(
            [[b" "] * 5, [b" "] * 5, [b" "] * 2 + [b"1"] + [b" "] * 2, [b" "] * 5, [b"@"] * 5]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if if the view is correct if the view exceeds the bot view
        self.move_agent(agent_id, [5, 3])
        expected_view = np.array(
            [[b" "] * 5, [b" "] * 5, [b" "] * 2 + [b"1"] + [b" "] * 2, [b"@"] * 5, [b" "] * 5]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if the view is correct if the right wall is just in view
        self.move_agent(agent_id, [3, 4])
        expected_view = np.array(
            [
                [b" "] * 4 + [b"@"],
                [b" "] * 4 + [b"@"],
                [b" "] * 2 + [b"1"] + [b" "] + [b"@"],
                [b" "] * 4 + [b"@"],
                [b" "] * 4 + [b"@"],
            ]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if if the view is correct if the view exceeds the right view
        self.move_agent(agent_id, [3, 5])
        expected_view = np.array(
            [
                [b" "] * 3 + [b"@"] + [b" "],
                [b" "] * 3 + [b"@"] + [b" "],
                [b" "] * 2 + [b"1"] + [b"@"] + [b" "],
                [b" "] * 3 + [b"@"] + [b" "],
                [b" "] * 3 + [b"@"] + [b" "],
            ]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

        # check if if the view is correct if the agent is in the bottom right corner
        self.move_agent(agent_id, [5, 5])
        expected_view = np.array(
            [
                [b" "] * 3 + [b"@"] + [b" "],
                [b" "] * 3 + [b"@"] + [b" "],
                [b" "] * 2 + [b"1"] + [b"@"] + [b" "],
                [b"@"] * 4 + [b" "],
                [b" "] * 5,
            ]
        )
        self.assert_logical_and_color_view(agent_id, expected_view)

    def test_agent_visibility(self):
        self.env = DummyMapEnv(
            TEST_MAP_1.copy(), extra_actions={}, view_len=2, num_agents=0, return_agent_actions=True
        )
        self.env.reset()
        self.add_agent("agent-0", [1, 1], "UP", self.env, 2)
        self.add_agent("agent-1", [1, 3], "UP", self.env, 2)
        self.add_agent("agent-2", [1, 5], "UP", self.env, 2)
        obs, *_ = self.env.step({})
        visibility = [agent_obs["visible_agents"] for agent_obs in obs.values()]
        np.testing.assert_array_equal(visibility[0], [1, 0])
        np.testing.assert_array_equal(visibility[1], [1, 1])
        np.testing.assert_array_equal(visibility[2], [0, 1])

    def test_agent_actions(self):
        # set up the map
        agent_id = "agent-0"
        self.construct_map(TEST_MAP_1.copy(), agent_id, [2, 2], "LEFT")

        # Test that all the moves and rotations work correctly
        # test when facing left
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [3, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 1])
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        # test when facing up
        self.rotate_agent(agent_id, "UP")
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 1])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [1, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        # test when facing down
        self.rotate_agent(agent_id, "DOWN")
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 3])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [3, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        # test when facing right
        self.rotate_agent(agent_id, "RIGHT")
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [1, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 3])
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])

        # check that stay works properly
        self.env.step({agent_id: ACTION_MAP["STAY"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])
        self.assertEqual(get_env_test_map(self.env)[2, 2], b"P")

        # quick test of stay
        self.env.step({agent_id: ACTION_MAP["STAY"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 2])

        # if an agent tries to move through a wall they should stay in the same place
        # we check that this works correctly for both corner and non-corner edges
        self.rotate_agent(agent_id, "UP")
        self.move_agent(agent_id, [1, 1])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [1, 1])
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [1, 1])
        self.move_agent(agent_id, [4, 4])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [5, 5])
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [5, 5])
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        self.env.step({agent_id: ACTION_MAP["MOVE_DOWN"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [5, 4])
        self.move_agent(agent_id, [4, 5])
        self.env.step({agent_id: ACTION_MAP["MOVE_RIGHT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [4, 5])
        self.move_agent(agent_id, [2, 1])
        self.env.step({agent_id: ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [2, 1])
        self.move_agent(agent_id, [1, 2])
        self.env.step({agent_id: ACTION_MAP["MOVE_UP"]})
        np.testing.assert_array_equal(self.env.agents[agent_id].pos, [1, 2])

        # rotations correctly update the agent state
        self.rotate_agent(agent_id, "UP")
        # clockwise
        self.env.step({agent_id: ACTION_MAP["TURN_CLOCKWISE"]})
        self.assertEqual("RIGHT", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_CLOCKWISE"]})
        self.assertEqual("DOWN", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_CLOCKWISE"]})
        self.assertEqual("LEFT", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_CLOCKWISE"]})
        self.assertEqual("UP", self.env.agents[agent_id].get_orientation())

        # counterclockwise
        self.env.step({agent_id: ACTION_MAP["TURN_COUNTERCLOCKWISE"]})
        self.assertEqual("LEFT", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_COUNTERCLOCKWISE"]})
        self.assertEqual("DOWN", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_COUNTERCLOCKWISE"]})
        self.assertEqual("RIGHT", self.env.agents[agent_id].get_orientation())
        self.env.step({agent_id: ACTION_MAP["TURN_COUNTERCLOCKWISE"]})
        self.assertEqual("UP", self.env.agents[agent_id].get_orientation())

    def test_agent_conflict(self):
        """Test that agent conflicts are correctly resolved"""

        # test that if there are two agents and two spawning points, they hit both of them
        self.env = DummyMapEnv(ascii_map=BASE_MAP_2, extra_actions={}, view_len=2, num_agents=2)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map, get_env_test_map(self.env))

        # test that agents can't walk into other agents
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [3, 4])
        self.rotate_agent("agent-0", "UP")
        self.rotate_agent("agent-1", "UP")
        self.env.step({"agent-0": ACTION_MAP["MOVE_RIGHT"]})
        self.env.step({"agent-1": ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents["agent-0"].pos, [3, 3])
        np.testing.assert_array_equal(self.env.agents["agent-1"].pos, [3, 4])

        # test that agents can't walk through each other if they move simultaneously
        self.env.step({"agent-0": ACTION_MAP["MOVE_RIGHT"], "agent-1": ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(self.env.agents["agent-0"].pos, [3, 3])
        np.testing.assert_array_equal(self.env.agents["agent-1"].pos, [3, 4])
        # also check that the map looks correct, no agent has disappeared
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b"P", b"P", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # test that agents can walk into other agents if moves are de-conflicting
        # conflict only occurs stochastically so try it 50 times
        np.random.seed(1)
        for i in range(100):
            self.env.step({"agent-0": ACTION_MAP["MOVE_RIGHT"], "agent-1": ACTION_MAP["MOVE_UP"]})
            expected_map = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b"P", b"@"],
                    [b"@", b" ", b" ", b" ", b"P", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
            self.env.step({"agent-0": ACTION_MAP["MOVE_LEFT"], "agent-1": ACTION_MAP["MOVE_DOWN"]})

        # test that if two agents have a conflicting move then the tie is broken randomly
        num_agent_1 = 0.0
        num_agent_2 = 0.0
        for i in range(100):
            self.move_agent("agent-0", [3, 2])
            self.move_agent("agent-1", [3, 4])
            self.env.step({"agent-0": ACTION_MAP["MOVE_RIGHT"], "agent-1": ACTION_MAP["MOVE_LEFT"]})
            if self.env.agents["agent-0"].pos.tolist() == [3, 3]:
                num_agent_1 += 1
            else:
                num_agent_2 += 1
            # Also check that the map looks correct
            expect_1 = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b"P", b"P", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            expect_2 = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b"P", b"P", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            equal_1 = np.array_equal(get_env_test_map(self.env), expect_1)
            equal_2 = np.array_equal(get_env_test_map(self.env), expect_2)
            self.assertTrue(equal_1 or equal_2)
        agent_1_percent = num_agent_1 / (num_agent_1 + num_agent_2)
        with_expected_val = (0.53 == agent_1_percent) or (0.47 == agent_1_percent)
        self.assertTrue(with_expected_val)

        # check that this works correctly with three agents
        self.add_agent("agent-2", [2, 3], "UP", self.env, 3)
        num_agent_1 = 0.0
        other_agents = 0.0
        for i in range(100):
            self.move_agent("agent-0", [3, 2])
            self.move_agent("agent-1", [3, 4])
            self.move_agent("agent-2", [2, 3])
            self.env.step(
                {
                    "agent-0": ACTION_MAP["MOVE_RIGHT"],
                    "agent-1": ACTION_MAP["MOVE_LEFT"],
                    "agent-2": ACTION_MAP["MOVE_DOWN"],
                }
            )
            if self.env.agents["agent-2"].pos.tolist() == [3, 3]:
                num_agent_1 += 1
            else:
                other_agents += 1
            # Also check that the map looks correct
            expect_1 = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b"P", b" ", b"@"],
                    [b"@", b" ", b"P", b"P", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            expect_2 = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b"P", b"P", b"P", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            expect_3 = np.array(
                [
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b" ", b" ", b"P", b" ", b"@"],
                    [b"@", b" ", b" ", b"P", b"P", b"@"],
                    [b"@", b" ", b" ", b" ", b" ", b"@"],
                    [b"@", b"@", b"@", b"@", b"@", b"@"],
                ]
            )
            equal_1 = np.array_equal(get_env_test_map(self.env), expect_1)
            equal_2 = np.array_equal(get_env_test_map(self.env), expect_2)
            equal_3 = np.array_equal(get_env_test_map(self.env), expect_3)
            self.assertTrue(equal_1 or equal_2 or equal_3)
        agent_1_percent = num_agent_1 / (num_agent_1 + other_agents)
        within_bounds = (agent_1_percent > 0.27) and (agent_1_percent < 0.39)
        self.assertTrue(within_bounds)

        # you try to move into an agent that is in conflict with another agent
        # fifty percent of the time you should succeed
        percent_accomplished = 0
        percent_failed = 0
        for i in range(100):
            self.move_agent("agent-1", [3, 4])
            self.move_agent("agent-2", [2, 2])
            self.move_agent("agent-0", [3, 2])
            self.env.step(
                {
                    "agent-0": ACTION_MAP["MOVE_RIGHT"],
                    "agent-1": ACTION_MAP["MOVE_LEFT"],
                    "agent-2": ACTION_MAP["MOVE_DOWN"],
                }
            )
            if self.env.agents["agent-2"].pos.tolist() == [2, 2]:
                percent_failed += 1
                expect_1 = np.array(
                    [
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b"P", b" ", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                    ]
                )
                np.testing.assert_array_equal(expect_1, get_env_test_map(self.env))
            else:
                percent_accomplished += 1
                expect_1 = np.array(
                    [
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b"P", b"P", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                    ]
                )
                np.testing.assert_array_equal(expect_1, get_env_test_map(self.env))
        percent_success = percent_accomplished / (percent_accomplished + percent_failed)
        within_bounds = (0.40 < percent_success) and (percent_success < 0.60)
        self.assertTrue(within_bounds)

        # Check that if there is more than one conflict simultaneously
        # that it is handled correctly
        agent_0_percent = 0
        agent_1_percent = 0
        num_trials = 100
        self.add_agent("agent-3", [1, 4], "UP", self.env, 3)
        for i in range(num_trials):
            self.move_agent("agent-1", [3, 4])
            self.move_agent("agent-2", [1, 2])
            self.move_agent("agent-0", [3, 2])
            self.move_agent("agent-3", [1, 4])
            self.env.step(
                {
                    "agent-0": ACTION_MAP["MOVE_UP"],
                    "agent-2": ACTION_MAP["MOVE_DOWN"],
                    "agent-1": ACTION_MAP["MOVE_UP"],
                    "agent-3": ACTION_MAP["MOVE_DOWN"],
                }
            )
            if self.env.agents["agent-0"].pos.tolist() == [2, 2]:
                agent_0_percent += 1
            if self.env.agents["agent-1"].pos.tolist() == [2, 4]:
                agent_1_percent += 1
        agent_0_success = agent_0_percent / num_trials
        agent_1_success = agent_1_percent / num_trials
        within_bounds_0 = (0.4 < agent_0_success) and (agent_0_success < 0.6)
        within_bounds_1 = (0.4 < agent_1_success) and (agent_1_success < 0.6)
        self.assertTrue(within_bounds_0)
        self.assertTrue(within_bounds_1)

        # agent 3 wants to move into space [3,2] as does agent-2
        # however, agent-1 wants to move into [3,3] so technically
        # no move is possible and no agent should move
        self.move_agent("agent-0", [3, 2])
        self.move_agent("agent-2", [2, 2])
        self.move_agent("agent-1", [2, 3])
        self.move_agent("agent-3", [3, 3])
        self.env.step(
            {
                "agent-0": ACTION_MAP["MOVE_UP"],
                "agent-1": ACTION_MAP["MOVE_DOWN"],
                "agent-2": ACTION_MAP["MOVE_DOWN"],
                "agent-3": ACTION_MAP["MOVE_LEFT"],
            }
        )
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b"P", b"P", b" ", b"@"],
                [b"@", b" ", b"P", b"P", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # agent 3 wants to move into space [3,2] as does agent-2
        # agent-0 will move out of the way so one of them should successfully
        # get the cell
        self.move_agent("agent-0", [3, 2])
        self.move_agent("agent-2", [2, 2])
        # move this agent out of the way
        self.move_agent("agent-1", [4, 4])
        self.move_agent("agent-3", [3, 3])
        agent_2_success = 0
        for i in range(100):
            self.env.step(
                {
                    "agent-0": ACTION_MAP["MOVE_DOWN"],
                    "agent-2": ACTION_MAP["MOVE_DOWN"],
                    "agent-3": ACTION_MAP["MOVE_LEFT"],
                }
            )
            if self.env.agents["agent-2"].pos.tolist() == [3, 2]:
                agent_2_success += 1
                expected_map = np.array(
                    [
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b"P", b" ", b"@"],
                        [b"@", b" ", b"P", b" ", b"P", b"@"],
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                    ]
                )
                np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
            else:
                expected_map = np.array(
                    [
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                        [b"@", b" ", b" ", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b" ", b" ", b"@"],
                        [b"@", b" ", b"P", b" ", b"P", b"@"],
                        [b"@", b"@", b"@", b"@", b"@", b"@"],
                    ]
                )
                np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
            self.move_agent("agent-0", [3, 2])
            self.move_agent("agent-2", [2, 2])
            # move this agent out of the way
            self.move_agent("agent-1", [4, 4])
            self.move_agent("agent-3", [3, 3])
        success_percent = agent_2_success / 100.0
        within_bounds = (0.4 < success_percent) and (success_percent < 0.6)
        self.assertTrue(within_bounds)

        # a counterclockwise rotation of a square of agents should work
        # properly
        self.move_agent("agent-0", [3, 2])
        self.move_agent("agent-2", [2, 2])
        # move this agent out of the way
        self.move_agent("agent-1", [2, 3])
        self.move_agent("agent-3", [3, 3])
        self.env.step(
            {
                "agent-0": ACTION_MAP["MOVE_UP"],
                "agent-1": ACTION_MAP["MOVE_DOWN"],
                "agent-2": ACTION_MAP["MOVE_RIGHT"],
                "agent-3": ACTION_MAP["MOVE_LEFT"],
            }
        )
        self.assertTrue(self.env.agents["agent-0"].pos.tolist() == [2, 2])
        self.assertTrue(self.env.agents["agent-1"].pos.tolist() == [3, 3])
        self.assertTrue(self.env.agents["agent-2"].pos.tolist() == [2, 3])
        self.assertTrue(self.env.agents["agent-3"].pos.tolist() == [3, 2])

        # do a check that the conflict resolution still works right
        # if one of the agents is trying to walk through a wall
        self.move_agent("agent-0", [2, 1])
        self.move_agent("agent-1", [1, 1])
        # move these agent out of the way
        self.move_agent("agent-2", [4, 4])
        self.move_agent("agent-3", [3, 3])
        curr_map = get_env_test_map(self.env).copy()
        self.env.step({"agent-0": ACTION_MAP["MOVE_UP"], "agent-1": ACTION_MAP["MOVE_LEFT"]})
        np.testing.assert_array_equal(get_env_test_map(self.env), curr_map)

    def move_agent(self, agent_id, new_pos):
        self.remove_agents_from_color_map()
        self.env.agents[agent_id].set_pos(new_pos)
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.full_map = map_with_agents
        self.env.agents[agent_id].update_agent_pos(new_pos)
        self.add_agents_to_color_map()

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = DummyMapEnv(map, extra_actions={}, view_len=2, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        map_with_agents = env.get_map_with_agents()
        self.env.agents[agent_id] = DummyAgent(
            agent_id, start_pos, start_orientation, map_with_agents, view_len, view_len
        )
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.full_map = map_with_agents
        self.env.agent_pos.append(start_pos)
        self.add_agents_to_color_map()

    def remove_agents_from_color_map(self):
        for agent in self.env.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            self.env.single_update_world_color_map(row, col, self.env.world_map[row, col])

    def add_agents_to_color_map(self):
        # Add agents to color map
        for agent in self.env.agents.values():
            row, col = agent.pos[0], agent.pos[1]
            # Firing beams have priority over agents and should cover them
            if self.env.world_map[row, col] not in [b"F", b"C"]:
                self.env.single_update_world_color_map(row, col, agent.get_char_id())


class TestHarvestEnv(unittest.TestCase):
    def tearDown(self):
        """Remove the env"""
        self.env = None

    def test_step(self):
        """Just check that the step method works at all for all possible actions"""
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)
        self.env.reset()
        action_dim = self.env.action_space.n
        for i in range(action_dim):
            self.env.step({"agent-0": i})

    def test_reset(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=0)
        self.env.reset()
        # check that the map is full of apples
        test_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(get_env_test_map(self.env), test_map)

    def test_apple_spawn(self):
        # render apples a bunch of times and check that the probabilities are within
        # a bound of what you expect. This test fill fail w/ <INSERT> probability
        self.env = HarvestEnv(MINI_HARVEST_MAP, num_agents=0)
        self.env.reset()
        self.env.world_map = TEST_MAP_2.copy()

        # First test, if we step 300 times, are there five apples there?
        # This should fail maybe one in 1000000 times
        for i in range(300):
            self.env.step({})
        num_apples = self.env.count_apples(get_env_test_map(self.env))
        self.assertEqual(num_apples, 5)

        # Now, if a point is temporarily obscured by a beam but an apple should spawn there
        # check that the apple still spawns there
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()
        self.move_agent("agent-0", [3, 1])
        self.move_agent("agent-1", [3, 3])
        self.rotate_agent("agent-0", "UP")
        self.rotate_agent("agent-1", "UP")
        self.env.step({"agent-1": HARVEST_ACTION_MAP["FIRE"]})
        self.env.update_map([[2, 1, b"A"]])
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"A", b" ", b"A", b"A", b"@"],
                [b"@", b"P", b" ", b"P", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # If an agent is temporarily obscured by a beam, and an apple attempts to spawn there
        # no apple should spawn
        self.env.step({"agent-1": HARVEST_ACTION_MAP["FIRE"]})
        self.env.update_map([[3, 1, b"A"]])
        self.env.step({})

        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"A", b" ", b"A", b"A", b"@"],
                [b"@", b"P", b" ", b"P", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

    def test_agent_actions(self):
        # set up the map
        agent_id = "agent-0"
        self.construct_map(TEST_MAP_1.copy(), agent_id, [2, 2], "LEFT")
        # test firing
        self.move_agent(agent_id, [3, 2])
        self.env.step({agent_id: HARVEST_ACTION_MAP["FIRE"]})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [
                [b"@"] + [b" "] * 4,
                [b"@"] + [b"F"] * 2 + [b" "] * 2,
                [b"@"] + [b"F"] + [b"1"] + [b" "] * 2,
                [b"@"] + [b"F"] * 2 + [b" "] * 2,
                [b"@"] + [b" "] * 4,
            ]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # clean up the map and then check if firing looks right
        self.env.step({})
        self.rotate_agent(agent_id, "RIGHT")
        self.move_agent(agent_id, [3, 2])
        self.env.step({agent_id: HARVEST_ACTION_MAP["FIRE"]})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [
                [b"@"] + [b" "] * 4,
                [b"@"] + [b" "] + [b"F"] * 3,
                [b"@"] + [b" "] + [b"1"] + [b"F"] * 2,
                [b"@"] + [b" "] + [b"F"] * 3,
                [b"@"] + [b" "] * 4,
            ]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

        # Check that agents walking over apples makes them go away
        np.random.seed(10)
        self.construct_map(MINI_HARVEST_MAP.copy(), agent_id, [3, 2], "RIGHT")
        self.env.step({agent_id: HARVEST_ACTION_MAP["MOVE_UP"]})
        self.env.step({agent_id: HARVEST_ACTION_MAP["MOVE_DOWN"]})
        agent_view = self.env.agents[agent_id].get_state()
        expected_view = np.array(
            [
                [b"@", b" ", b" ", b" ", b" "],
                [b"@", b" ", b" ", b"A", b"A"],
                [b"@", b" ", b"1", b" ", b"A"],
                [b"@", b" ", b" ", b"A", b" "],
                [b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_view, agent_view)

    def test_agent_rewards(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()
        self.move_agent("agent-0", [2, 2])
        self.move_agent("agent-1", [3, 2])
        self.rotate_agent("agent-0", "UP")
        self.rotate_agent("agent-1", "UP")
        # walk over an apple
        _, rew, _, _ = self.env.step(
            {
                "agent-0": HARVEST_ACTION_MAP["MOVE_RIGHT"],
                "agent-1": HARVEST_ACTION_MAP["MOVE_RIGHT"],
            }
        )
        self.assertTrue(rew["agent-0"] == 1)
        self.assertTrue(rew["agent-1"] == 1)
        # fire a beam from agent 1 to 2
        _, rew, _, _ = self.env.step({"agent-1": HARVEST_ACTION_MAP["FIRE"]})
        self.assertTrue(rew["agent-0"] == -50)
        self.assertTrue(rew["agent-1"] == -1)

    def test_agent_conflict(self):
        """Test that agent conflicts are correctly resolved"""

        # test that if there are two agents and two spawning points, they hit both of them
        self.env = HarvestEnv(ascii_map=BASE_MAP_2, num_agents=2)
        self.env.reset()
        np.testing.assert_array_equal(self.env.base_map, get_env_test_map(self.env))

        # test that agents can't walk into other agents
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [3, 4])
        self.rotate_agent("agent-0", "UP")
        self.rotate_agent("agent-1", "LEFT")

        # test that if an agents firing beam hits another agent it gets covered
        self.env.step({"agent-0": HARVEST_ACTION_MAP["MOVE_LEFT"]})
        self.env.step({"agent-1": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"F", b"F", b"F", b"F", b"@"],
                [b"@", b" ", b"F", b"F", b"P", b"@"],
                [b"@", b"F", b"F", b"F", b"F", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        # but by the next step, the agent is visible again
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b"P", b" ", b"P", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # test that if two agents fire on each other than they're still there after
        self.env.agents["agent-0"].update_agent_rot("RIGHT")
        self.env.step({"agent-0": HARVEST_ACTION_MAP["FIRE"], "agent-1": HARVEST_ACTION_MAP["FIRE"]})
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b"P", b" ", b"P", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

    def test_beam_conflict(self):
        """Test that after the beam is fired, obscured apples and agents are returned"""
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=2)
        self.env.reset()

        # test that agents can't walk into other agents
        self.move_agent("agent-0", [4, 2])
        self.move_agent("agent-1", [4, 4])
        self.env.agents["agent-0"].update_agent_rot("UP")
        self.env.agents["agent-1"].update_agent_rot("LEFT")
        # test that if an agents firing beam hits another agent it gets covered
        self.env.step({"agent-1": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b"F", b"F", b"F", b"F", b"@"],
                [b"@", b" ", b"F", b"F", b"P", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        # test that by the next step it will be returned
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b"P", b"A", b"P", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

    def map_to_colors(self, orientation, unrotated_view, equal_array):
        env = DummyMapEnv(BASE_MAP_1, extra_actions={}, view_len=1, num_agents=0)
        color_map = dict()
        for i in range(9):
            color_map[i] = int(i)

        rgb_array = np.full((3, 3, 1), fill_value=-1)
        env.map_to_colors(unrotated_view, color_map, rgb_array, orientation=orientation)
        rgb_array = rgb_array.reshape((3, 3))

        np.testing.assert_array_equal(rgb_array, equal_array)

    def test_rotations(self):
        unrotated_view = np.array(range(9)).reshape((3, 3))
        self.map_to_colors("UP", unrotated_view, np.rot90(unrotated_view, k=0))
        self.map_to_colors("LEFT", unrotated_view, np.rot90(unrotated_view, k=1))
        self.map_to_colors("DOWN", unrotated_view, np.rot90(unrotated_view, k=2))
        self.map_to_colors("RIGHT", unrotated_view, np.rot90(unrotated_view, k=3))

    def clear_agents(self):
        self.env.agents = {}

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        map_with_agents = env.get_map_with_agents()
        self.env.agents[agent_id] = HarvestAgent(
            agent_id, start_pos, start_orientation, map_with_agents, view_len
        )
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.full_map = map_with_agents
        self.env.agent_pos.append(start_pos)

    def move_agent(self, agent_id, new_pos):
        self.env.agents[agent_id].update_agent_pos(new_pos)
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.full_map = map_with_agents

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = HarvestEnv(map, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)

    def test_firing_range(self):
        # check that the firing beam extends as far as expected
        self.env = HarvestEnv(ascii_map=FIRE_RANGE_MAP, num_agents=0)
        self.env.reset()
        self.add_agent("agent-0", [2, 2], "UP", self.env, 5)
        self.move_agent("agent-0", [6, 6])
        # TODO(@evinitssky) should figure out a way to shrink this
        self.rotate_agent("agent-0", "UP")
        self.env.step({"agent-0": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b"F", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"P", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        self.rotate_agent("agent-0", "DOWN")
        self.env.step({"agent-0": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"P", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b"F", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        self.rotate_agent("agent-0", "RIGHT")
        self.env.step({"agent-0": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b"F", b"F", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b"P", b"F", b"F", b"F", b"F", b"F", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b"F", b"F", b"F", b"F", b"F", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        self.rotate_agent("agent-0", "LEFT")
        self.env.step({"agent-0": HARVEST_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b"F", b"F", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"F", b"F", b"F", b"F", b"F", b"P", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b"F", b"F", b"F", b"F", b"F", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))


class TestCleanupEnv(unittest.TestCase):
    def test_parameters(self):
        self.env = CleanupEnv(num_agents=0)
        self.assertEqual(self.env.potential_waste_area, 119)

    def test_reset(self):
        self.env = CleanupEnv(ascii_map=MINI_CLEANUP_MAP, num_agents=0)
        self.env.reset()
        # check that the map has no apples
        test_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b" ", b" ", b" ", b"@"],
                [b"@", b"R", b" ", b" ", b" ", b"@"],
                [b"@", b"S", b" ", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(get_env_test_map(self.env), test_map)

    def test_cleanup_beam(self):
        self.env = CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=2)
        self.env.reset()
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [4, 2])
        self.rotate_agent("agent-0", "LEFT")
        # check that cleanup beam does four things
        # 1. Cleans waste cells correctly
        # 2. Is blocked by the first waste cell it encounters
        # 3. Obscures agents when fired, doesn't remove them when cleaned
        # 4. Is blocked by agents
        self.env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b"C", b"C", b" ", b"@"],
                [b"@", b"R", b"C", b"P", b" ", b"@"],
                [b"@", b"H", b"C", b"C", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        np.random.seed(12)
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b"R", b" ", b" ", b"@"],
                [b"@", b"R", b"R", b"P", b" ", b"@"],
                [b"@", b"H", b"P", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # check that the cleanup beam doesn't remove apples
        self.env.reset()
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [4, 2])
        self.env.update_map([[3, 4, b"A"]])
        self.rotate_agent("agent-0", "RIGHT")
        self.env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"]})
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b"H", b" ", b" ", b"@"],
                [b"@", b"R", b"H", b"P", b"A", b"@"],
                [b"@", b"H", b"P", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # check that you can clean up waste that an agent is standing on
        self.move_agent("agent-1", [2, 2])
        self.move_agent("agent-0", [1, 3])
        self.rotate_agent("agent-0", "DOWN")
        random.seed(6)
        self.env.step({"agent-0": CLEANUP_ACTION_MAP["CLEAN"]})
        self.assertTrue(self.env.world_map[2, 2] == b"R")

        # check that the beams add constructively, i.e. that if one beam clears
        # some waste then the next agents beam is not blocked by it and can hit
        # formerly blocked cells
        random.seed(7)
        self.move_agent("agent-1", [2, 3])
        self.move_agent("agent-0", [4, 3])
        # put some waste back where it's needed
        self.env.update_map([[2, 2, b"H"]])
        self.env.update_map([[3, 1, b"H"]])
        self.rotate_agent("agent-0", "LEFT")
        self.rotate_agent("agent-1", "LEFT")
        self.env.step(
            {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["CLEAN"]}
        )
        self.assertTrue(self.env.world_map[3, 1] == b"R")

    def test_firing_beam(self):
        self.env = CleanupEnv(ascii_map=FIRING_CLEANUP_MAP, num_agents=2)
        self.env.reset()
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [4, 2])
        self.rotate_agent("agent-0", "LEFT")

        # check that firing beam does not clean anything and is not blocked
        # by anything
        self.env.step({"agent-0": CLEANUP_ACTION_MAP["FIRE"]})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"F", b"F", b"F", b" ", b"@"],
                [b"@", b"F", b"F", b"P", b" ", b"@"],
                [b"@", b"H", b"F", b"F", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))
        # check that the firing beam is removed correctly after one step
        # it should not remove any waste, rivers, or agents
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b"H", b" ", b" ", b"@"],
                [b"@", b"R", b"H", b"P", b" ", b"@"],
                [b"@", b"H", b"P", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

        # check that the cleanup beam doesn't remove apples
        self.env.reset()
        self.move_agent("agent-0", [3, 3])
        self.move_agent("agent-1", [4, 2])
        self.env.update_map([[3, 4, b"A"]])
        self.rotate_agent("agent-0", "RIGHT")
        self.env.step({"agent-0": CLEANUP_ACTION_MAP["FIRE"]})
        self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b" ", b" ", b" ", b"@"],
                [b"@", b"H", b"H", b" ", b" ", b"@"],
                [b"@", b"R", b"H", b"P", b"A", b"@"],
                [b"@", b"H", b"P", b" ", b" ", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

    def test_apple_spawn(self):
        """Confirm that apples spawn correctly in cleanup"""
        self.env = CleanupEnv(ascii_map=APPLE_SPAWN_MAP_CLEANUP, num_agents=2)
        self.env.reset()
        for i in range(500):
            self.env.step({})
        expected_map = np.array(
            [
                [b"@", b"@", b"@", b"@", b"@", b"@"],
                [b"@", b" ", b"P", b" ", b" ", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b"A", b"@"],
                [b"@", b" ", b" ", b"A", b"P", b"@"],
                [b"@", b"@", b"@", b"@", b"@", b"@"],
            ]
        )
        np.testing.assert_array_equal(expected_map, get_env_test_map(self.env))

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
        np.random.seed(1)
        self.rotate_agent("agent-0", "LEFT")
        self.rotate_agent("agent-1", "LEFT")
        self.env.step(
            {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["CLEAN"]}
        )
        self.assertTrue(np.isclose(self.env.current_waste_spawn_prob, 0.5))

        # check that the apple spawn probability is computed correctly
        while True:
            self.env.step(
                {"agent-0": CLEANUP_ACTION_MAP["CLEAN"], "agent-1": CLEANUP_ACTION_MAP["CLEAN"]}
            )
            if self.env.compute_permitted_area() == 4:
                break
        self.env.compute_probabilities()
        self.assertTrue(np.isclose(self.env.current_apple_spawn_prob, 0.025))

        # test that you can spawn waste under an agent
        self.move_agent("agent-0", [2, 1])
        random.seed(5)
        self.env.step({})
        self.assertTrue(self.env.world_map[2, 1] == b"H")

    def test_past_bugs(self):
        """This function is used to check that previous bugs do not regress"""
        pass

    def clear_agents(self):
        self.env.agents = {}

    def add_agent(self, agent_id, start_pos, start_orientation, env, view_len):
        map_with_agents = env.get_map_with_agents()
        self.env.agents[agent_id] = CleanupAgent(
            agent_id, start_pos, start_orientation, map_with_agents, view_len
        )
        map_with_agents = env.get_map_with_agents()

        for agent in env.agents.values():
            # Update each agent's view of the world
            agent.full_map = map_with_agents
        self.env.agent_pos.append(start_pos)

    def move_agent(self, agent_id, new_pos):
        self.env.agents[agent_id].update_agent_pos(new_pos)
        map_with_agents = self.env.get_map_with_agents()
        agent = self.env.agents[agent_id]
        agent.full_map = map_with_agents

    def rotate_agent(self, agent_id, new_rot):
        self.env.agents[agent_id].update_agent_rot(new_rot)

    def construct_map(self, map, agent_id, start_pos, start_orientation):
        # overwrite the map for testing
        self.env = CleanupEnv(map, num_agents=0)
        self.env.reset()

        # replace the agents with agents with smaller views
        self.add_agent(agent_id, start_pos, start_orientation, self.env, 2)


if __name__ == "__main__":
    unittest.main()
