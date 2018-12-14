"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (MapEnv)
            a reference to the containing environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = start_pos
        self.orientation = start_orientation
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def set_pos(self, new_pos):
        self.pos = new_pos

    def get_pos(self):
        return self.pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid.map


# use keyword names so that it's easy to understand what the agent is calling
HARVEST_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                   1: 'MOVE_RIGHT',  # Move right
                   2: 'MOVE_UP',  # Move up
                   3: 'MOVE_DOWN',  # Move down
                   4: 'STAY',  # don't move
                   5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                   6: 'TURN_COUNTERCLOCKWISE',  # Rotate clockwise
                   7: 'FIRE'}  # Fire forward

class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid):
        # FIXME(ev) put in the right sizes
        # FIXME(ev) magic number bruh
        super().__init__(agent_id, start_pos, start_orientation, grid, 3, 3)

    @property
    def action_space(self):
        return Discrete(8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        # FIXME(ev) put in the right sizes
        return Box(low=0.0, high=0.0, shape=(3, 3, 3), dtype=np.float32)

    def get_state(self):
        return self.grid.return_view(self.pos, self.row_size, self.col_size)

    def get_reward(self):
        # FIXME(ev) put in the actual reward
        return 1

    def get_done(self):
        # FIXME(ev) put in the actual computation
        return False


class CleanupAgent(Agent):
    pass
