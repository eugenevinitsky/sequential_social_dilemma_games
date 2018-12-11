"""Base class for an agent that defines the possible actions. """


class Agent(object):

    def __init__(self, agent_id, start_pos, grid, row_size, col_size):
        self.agent_id = agent_id
        self.pos = start_pos
        # FIXME(ev) who should hold positions, grid or agent?
        # There's an argument for both sides
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size

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


class HarvestAgent(Agent):
    def get_state(self):
        return self.grid.return_view(self.pos, self.row_size, self.col_size)

    def compute_reward(self):
        # FIXME(ev) put in the actual reward
        return 1

class CleanupAgent(Agent):
    pass