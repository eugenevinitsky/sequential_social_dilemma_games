import numpy as np

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS, ORIENTATIONS

APPLE_RADIUS = 2

COLOURS = {' ': [0, 0, 0],  # Black background
           '': [195, 0, 255],  # Board walls
           '@': [195, 0, 255],  # Board walls
           'A': [0, 255, 0],  # Green apples
           'P': [0, 255, 255],  # Player #FIXME(ev) agents need to have different colors
           'F': [255, 255, 0]}  # Yellow firing beam

# Add custom actions to the agent
ACTIONS['FIRE'] = 5

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

# FIXME(ev) this whole thing is in serious need of some abstraction
# FIXME(ev) switching betewen types and lists in a pretty arbitrary manner


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, COLOURS, num_agents, render)
        # set up the list of spawn points
        self.apple_points = []
        self.wall_points = []
        self.firing_points = []
        self.hidden_apples = []
        self.hidden_agents = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        # TODO(ev) this call should be in the superclass
        self.setup_agents()

    # FIXME(ev) action_space should really be defined in the agents
    @property
    def action_space(self):
        pass

    @property
    def observation_space(self):
        pass

    # TODO(ev) this can probably be moved into the superclass
    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            agent = HarvestAgent(agent_id, self.spawn_point(), self.spawn_rotation(), self, 3)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        self.firing_points = []
        self.build_walls()
        self.update_map_apples(self.apple_points)

    def custom_action(self, agent):
        agent.fire_beam()
        self.reserved_slots += self.update_map_fire(agent.get_pos().tolist(),
                                                    agent.get_orientation())

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        if len(new_apples) > 0:
            self.reserved_slots += new_apples

    def clean_map(self):
        """Put back agents and apples that were hidden by the fining beam"""
        agent_pos = []
        for agent in self.agents.values():
            agent_pos.append(agent.get_pos().tolist())
        for i in range(len(self.firing_points)):
            row, col = self.firing_points[i]
            if self.firing_points[i] in self.hidden_apples:
                self.map[row, col] = 'A'
            elif [row, col] in agent_pos:
                # put the agent back if they were temporarily obscured by the firing beam
                self.map[row, col] = 'P'
            else:
                self.map[row, col] = ' '
        self.hidden_apples = []
        self.firing_points = []
        self.hidden_agents = []

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
            self.firing_points.append([row, col])

        # update the apples
        self.update_map_apples(apple_pos)

    # TODO(ev) this is almost certainly used by every environment
    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.map[row, col] = '@'

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
            window = self.return_view(self.apple_points[i], APPLE_RADIUS, APPLE_RADIUS)
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

    def update_map_fire(self, firing_pos, firing_orientation):
        num_fire_cells = 5
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        firing_points = []
        for i in range(num_fire_cells):
            next_cell = start_pos + firing_direction
            if self.test_if_in_bounds(next_cell) and self.map[next_cell[0], next_cell[1]] != '@':
                if self.map[next_cell[0], next_cell[1]] == 'A':
                    self.hidden_apples.append([next_cell[0], next_cell[1]])
                elif self.map[next_cell[0], next_cell[1]] == 'P':
                    self.hidden_agents.append([next_cell[0], next_cell[1]])
                self.map[next_cell[0], next_cell[1]] = 'F'
                firing_points.append((next_cell[0], next_cell[1], 'F'))
                start_pos += firing_direction
            else:
                break
        return firing_points

    def update_map_apples(self, new_apple_points):
        for i in range(len(new_apple_points)):
            row, col = new_apple_points[i]
            if self.map[row, col] != 'P' and self.map[row, col] != 'F':
                self.map[row, col] = 'A'
            elif self.map[row, col] == 'F' and [row, col] not in self.hidden_agents:
                self.hidden_apples.append([row, col])
