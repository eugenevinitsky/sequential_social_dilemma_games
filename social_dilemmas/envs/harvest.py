from gym.spaces import Box, Dict, Discrete
import numpy as np

from social_dilemmas.envs.agent import HarvestAgent  # HARVEST_VIEW_SIZE
from social_dilemmas.maps import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

HARVEST_VIEW_SIZE = 7


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False, return_agent_actions=False):
        super().__init__(ascii_map, num_agents, render,  return_agent_actions=return_agent_actions)
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

        self.view_len = HARVEST_VIEW_SIZE

    @property
    def observation_space(self):
        if self.return_agent_actions:
            # We will append on some extra values to represent the actions of other agents
            return Dict({"curr_obs": Box(low=-np.infty, high=np.infty, shape=(2 * self.view_len + 1,
                                                 2 * self.view_len + 1, 3), dtype=np.float32),
                         "other_agent_actions": Box(low=0, high=len(ACTIONS), shape=(self.num_agents - 1, ), dtype=np.int32,),
                         "visible_agents": Box(low=0, high=self.num_agents, shape=(self.num_agents - 1,), dtype=np.int32)})
        else:
            return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                                 2 * self.view_len + 1, 3), dtype=np.float32)

    @property
    def action_space(self):
        return Discrete(8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j**2 + k**2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples
