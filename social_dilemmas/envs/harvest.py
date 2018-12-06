from gym.spaces import Box
import numpy as np
import six
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.constants import HARVEST_MAP

class HarvestEnv(MapEnv):

    def __init__(self):
        self.row_size = HARVEST_MAP.shape[0]
        self.col_size = HARVEST_MAP
        self.map = self.ascii_to_matrix(HARVEST_MAP)

    @property
    def action_space(self):
        pass

    @property
    def observation_space(self):
        pass

    def step(self):
        pass

    def reset(self):
        pass

    def update_map(self):
        pass

    def spawn_apples(self):
        pass

