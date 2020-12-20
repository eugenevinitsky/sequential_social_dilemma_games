import numpy as np
import random

from social_dilemmas.constants import CLEANUP_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS
from social_dilemmas.envs.agent import CleanupAgent  # CLEANUP_VIEW_SIZE

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing beam
ACTIONS['CLEAN'] = 5  # length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {'C': [100, 255, 255],  # Cyan cleaning beam
                  'S': [99, 156, 194],  # Light grey-blue stream cell
                  'H': [113, 75, 24],  # brown waste cells
                  'R': [99, 156, 194]}  # Light grey-blue river cell

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

cleanup_params_default = {'thresholdDepletion': 0.4,
                          'thresholdRestoration': 0.0,
                          'wasteSpawnProbability': 0.5,
                          'appleRespawnProbability': 0.05}

class CleanupEnv(MapEnv):

    def __init__(self, ascii_map=CLEANUP_MAP, num_agents=1, render=False,
                 shuffle_spawn=True, global_ref_point=None,
                 view_size=7, random_orientation=True,
                 cleanup_params=cleanup_params_default,
                 beam_width=3):
        self.global_ref_point = global_ref_point
        self.view_size = view_size
        super().__init__(ascii_map, num_agents, render,
                         shuffle_spawn=shuffle_spawn,
                         random_orientation=random_orientation,
                         beam_width=beam_width)

        self.thresholdDepletion = cleanup_params['thresholdDepletion']
        self.thresholdRestoration = cleanup_params['thresholdRestoration']
        self.wasteSpawnProbability = cleanup_params['wasteSpawnProbability']
        self.appleRespawnProbability = cleanup_params['appleRespawnProbability']

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get('H', 0) + counts_dict.get('R', 0)
        self.current_apple_spawn_prob = self.appleRespawnProbability
        self.current_waste_spawn_prob = self.wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'B':
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == 'S':
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == 'H':
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] == 'H' or self.base_map[row, col] == 'R':
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == 'R':
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        # FIXME(ev) this is an information leak
        agents = list(self.agents.values())
        return agents[0].observation_space

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.world_map[waste_start_point[0], waste_start_point[1]] = 'H'
        for river_point in self.river_points:
            self.world_map[river_point[0], river_point[1]] = 'R'
        for stream_point in self.stream_points:
            self.world_map[stream_point[0], stream_point[1]] = 'S'
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == 'FIRE':
            agent.fire_beam('F')
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(), ACTIONS['FIRE'],
                                           fire_char='F')
        elif action == 'CLEAN':
            agent.fire_beam('C')
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(),
                                           ACTIONS['FIRE'],
                                           fire_char='C',
                                           cell_types=['H'],
                                           update_char=['R'],
                                           blocking_cells=['H'])
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(agent_id, spawn_point, rotation, map_with_agents,
                                 global_ref_point=self.global_ref_point,
                                 view_len=self.view_size)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, 'A'))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != 'H':
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, 'H'))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= self.thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = self.wasteSpawnProbability
            if waste_density <= self.thresholdRestoration:
                self.current_apple_spawn_prob = self.appleRespawnProbability
            else:
                spawn_prob = (1 - (waste_density - self.thresholdRestoration)
                              / (self.thresholdDepletion - self.thresholdRestoration)) \
                             * self.appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get('H', 0)
        free_area = self.potential_waste_area - current_area
        return free_area
