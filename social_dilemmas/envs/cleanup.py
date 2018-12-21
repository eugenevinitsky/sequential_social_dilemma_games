import numpy as np

from social_dilemmas.envs.agent import CleanupAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS, ORIENTATIONS

# TODO(ev) add waste colors
COLOURS = {' ': [0, 0, 0],  # Black background
           '': [195, 0, 255],  # Board walls
           '@': [195, 0, 255],  # Board walls
           'A': [0, 255, 0],  # Green apples
           'P': [0, 255, 255],  # Player #FIXME(ev) agents need to have different colors
           'F': [255, 255, 0]}  # Yellow firing beam

# Add custom actions to the agent
ACTIONS['FIRE'] = 5

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

class CleanupEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, COLOURS, num_agents, render)

    def custom_reset(self):
        """Reset custom elements of the map"""
        raise NotImplementedError


    def custom_action(self, agent):
        """Allows agents to take actions that are not move or turn"""
        raise NotImplementedError


    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        raise NotImplementedError


    def clean_map(self):
        """Clean map of elements that should be removed. Executed every step/"""
        raise NotImplementedError


    def execute_custom_reservations(self):
        """Execute reserved slots that do not have to do with moving"""
        raise NotImplementedError


    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        raise NotImplementedError