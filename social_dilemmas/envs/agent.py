"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box
from gym.spaces import Discrete

class Agent(object):

    def __init__(self, agent_id, start_pos, grid, row_size, col_size):
        self.agent_id = agent_id
        self.pos = start_pos
        # FIXME(ev) who should hold positions, grid or agent?
        # There's an argument for both sides
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

    def possible_actions(self):
        """Returns a mapping between numbers and """
        pass

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def update_pos(self, new_pos):
        self.pos = new_pos

    def get_pos(self):
        return self.pos

class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, grid):
        # FIXME(ev) put in the right sizes
        super.__init__(agent_id, start_pos, grid, 10, 10)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(10, 10, 3), dtype=np.float32)

    def get_state(self):
        return self.grid.return_view(self.pos, self.row_size, self.col_size)

    def compute_reward(self):
        # FIXME(ev) put in the actual reward
        return 1

class CleanupAgent(Agent):
    pass