import numpy as np

from social_dilemmas.constants import CLEANUP_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS, ORIENTATIONS
from social_dilemmas.envs.agent import CleanupAgent, CLEANUP_VIEW_SIZE
import utility_funcs as util

# Add custom actions to the agent
ACTIONS['FIRE'] = 5

# Custom colour dictionary
CLEANUP_COLORS = {'C': [100, 255, 255],  # Cyan cleaning beam
                  'S': [99, 156, 194],  # Light grey-blue stream cell
                  'H': [113, 75, 24],  # brown waste cells
                  'R': [99, 156, 194]}  # Light grey-blue river cell

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class CleanupEnv(MapEnv):

    def __init__(self, ascii_map=CLEANUP_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, num_agents, render)

        self.no_update_cells = ['F']

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get('H', 0) + counts_dict.get('R', 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.firing_points = []
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
        self.firing_points = []
        self.update_map_waste(self.waste_start_points)
        self.update_map_river(self.river_points)
        self.update_map_stream(self.stream_points)

    def custom_action(self, agent):
        """Allows agents to take actions that are not move or turn"""
        agent.fire_beam()
        self.reserved_slots += self.update_map_fire(agent.get_pos().tolist(),
                                                    agent.get_orientation())

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        # spawn the apples
        self.compute_probabilities()
        new_apples_and_waste = self.spawn_apples_and_waste()
        if len(new_apples_and_waste) > 0:
            self.reserved_slots += new_apples_and_waste

    def execute_custom_reservations(self):
        """Execute firing and then apple spawning"""
        apple_pos = []
        firing_pos = []
        waste_pos = []
        for slot in self.reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'A':
                apple_pos.append([row, col])
            elif slot[2] == 'H':
                waste_pos.append([row, col])
            elif slot[2] == 'F':
                firing_pos.append([row, col])
        for pos in firing_pos:
            row, col = pos
            self.map[row, col] = 'F'
            self.firing_points.append([row, col])

        # update the apples
        self.update_map_apples(apple_pos)
        self.update_map_waste(waste_pos)

    def append_hiddens(self, new_pos, old_char, new_char=None):
        """Add hidden cells to self.hidden_cells that should be put back when cleaning"""
        # an apple is gone once an agent walks over it
        if old_char == 'A' and new_char == 'P':
            self.hidden_cells.append(new_pos + [' '])
        # a waste cell is gone if a firing cell hits it
        if old_char == 'H' and new_char == 'F':
            self.hidden_cells.append(new_pos + ['R'])
        else:
            self.hidden_cells.append(new_pos + [old_char])

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = util.return_view(map_with_agents, spawn_point,
                                    CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            if self.map[row, col] != 'P' and self.map[row, col] != 'A':
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, 'A'))

        # spawn one waste point
        if not np.isclose(self.current_waste_spawn_prob, 0):
            while True:
                spawn_index = np.random.randint(0, len(self.waste_points))
                row, col = self.waste_points[spawn_index]
                if self.map[row, col] != 'P' and self.map[row, col] != 'H':
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, 'H'))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                coeff = appleRespawnProbability / (thresholdDepletion - thresholdRestoration)
                spawn_prob = (1 - (waste_density - thresholdRestoration)) * coeff
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get('H', 0)
        free_area = self.potential_waste_area - current_area
        return free_area

    def update_map_waste(self, new_waste_points):
        for i in range(len(new_waste_points)):
            row, col = new_waste_points[i]
            # TODO(ev) can waste spawn where an agent or  beam is?
            if self.map[row, col] != 'P' and self.map[row, col] != 'F':
                self.map[row, col] = 'H'

    def update_map_apples(self, new_apple_points):
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        for i in range(len(new_apple_points)):
            row, col = new_apple_points[i]
            if self.map[row, col] != 'P' and self.map[row, col] != 'F':
                self.map[row, col] = 'A'
            # you can't spawn apples if an agent is there but hidden by a beam,
            elif self.map[row, col] == 'F' and [row, col] not in curr_agent_pos:
                self.append_hiddens([row, col], 'A')

    def update_map_river(self, new_river_points):
        for i in range(len(new_river_points)):
            row, col = new_river_points[i]
            self.map[row, col] = 'R'

    def update_map_stream(self, new_stream_points):
        for i in range(len(new_stream_points)):
            row, col = new_stream_points[i]
            self.map[row, col] = 'S'

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
