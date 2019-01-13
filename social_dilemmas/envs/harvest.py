from gym.spaces import Discrete
import numpy as np

from social_dilemmas.envs.agent import HarvestAgent, HARVEST_VIEW_SIZE
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS, ORIENTATIONS
import utility_funcs as util


APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

# FIXME(ev) this whole thing is in serious need of some abstraction
# FIXME(ev) switching betewen types and lists in a pretty arbitrary manner


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, num_agents, render)
        self.no_update_cells = ['F']
        self.firing_points = []
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

    # FIXME(ev) action_space should really be defined in the agents
    @property
    def action_space(self):
        return Discrete(8)

    @property
    def observation_space(self):
        # FIXME(ev) this is an information leak
        agents = list(self.agents.values())
        return agents[0].observation_space

    # TODO(ev) this can probably be moved into the superclass
    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = util.return_view(map_with_agents, spawn_point,
                                    HARVEST_VIEW_SIZE, HARVEST_VIEW_SIZE)
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        self.firing_points = []
        self.update_map_apples(self.apple_points)

    def custom_action(self, agent, action):
        agent.fire_beam()
        self.reserved_slots += self.update_map_fire(agent.get_pos().tolist(),
                                                    agent.get_orientation())

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        if len(new_apples) > 0:
            self.reserved_slots += new_apples

    def execute_custom_reservations(self):
        """Execute firing and then apple spawning"""
        apple_pos = []
        firing_pos = []
        for slot in self.reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'A':
                apple_pos.append([row, col])
            elif slot[2] == 'F':
                firing_pos.append([row, col])
        for pos in firing_pos:
            row, col = pos
            self.map[row, col] = 'F'

        # update the apples
        self.update_map_apples(apple_pos)

    def append_hiddens(self, new_pos, old_char, new_char=None):
        """Add hidden cells to self.hidden_cells that should be put back when cleaning"""
        # an apple is gone once an agent walks over it

        if old_char == 'A' and new_char == 'P':
            self.hidden_cells.append(new_pos + [' '])
        else:
            self.hidden_cells.append(new_pos + [old_char])

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
            if self.map[row, col] != 'P' and self.map[row, col] != 'A':
                window = util.return_view(self.map, self.apple_points[i], APPLE_RADIUS, APPLE_RADIUS)
                num_apples = self.count_apples(window)
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

    def update_map_apples(self, new_apple_points):
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        for i in range(len(new_apple_points)):
            row, col = new_apple_points[i]
            if self.map[row, col] != 'P' and self.map[row, col] != 'F':
                self.map[row, col] = 'A'
            # you can't spawn apples if an agent is there but hidden by a beam,
            elif self.map[row, col] == 'F' and [row, col] not in curr_agent_pos:
                self.append_hiddens([row, col], 'A')

    # TODO(ev) this is in two classes already
    def update_map_fire(self, firing_pos, firing_orientation):
        num_fire_cells = ACTIONS['FIRE']
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        firing_pos = [start_pos, start_pos + right_shift - firing_direction,
                      start_pos - right_shift - firing_direction]
        firing_points = []
        for pos in firing_pos:
            for i in range(num_fire_cells):
                next_cell = pos + firing_direction
                if self.test_if_in_bounds(next_cell) and self.map[next_cell[0], next_cell[1]] != '@':
                    char = self.map[next_cell[0], next_cell[1]]
                    self.append_hiddens([next_cell[0], next_cell[1]], char, 'F')
                    firing_points.append((next_cell[0], next_cell[1], 'F'))
                    pos += firing_direction
                else:
                    break
        return firing_points
