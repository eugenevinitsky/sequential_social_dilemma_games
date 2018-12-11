from gym.spaces import Box
import numpy as np
import six

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv


APPLE_RADIUS = 2

COLOURS = {' ': [0, 0, 0],        # Black background
           '@': [764, 0, 999],    # Board walls
           'A': [0, 999, 0],      # Green apples
           'P': [0, 999, 999]}    # Player


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super.__init__(ascii_map, num_agents, render)

    @property
    def action_space(self):
        pass

    @property
    def observation_space(self):
        pass

    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            self.agents[agent_id] = self.create_agent(agent_id)

    def update_map(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions

        Returns
        -------
        new_map: numpy ndarray
            the updated map to store
        agent_pos: dict of tuples with keys as agent ids
        """

        # Move the agents
        base_map = self.ascii_map


        # spawn the apples
        raise NotImplementedError

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent"""
        return HarvestAgent(agent_id, )

    def spawn_apples(self):
        raise NotImplementedError

    def spawn_point(self):
        """Returns a randomly selected spawn point"""
        pass