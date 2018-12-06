"""Base class for an agent that defines the possible actions. """


class Agent(object):

    def __init__(self, agent_id, grid_pos):
        self.agent_id = agent_id
        self.grid_pos = grid_pos

    def action_map(self, action_number):
        """Maps action_number to a desired action in the maps"""
        raise NotImplementedError

    def possible_actions(self):
        """Returns a mapping between numbers and """

    def get_state(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError
