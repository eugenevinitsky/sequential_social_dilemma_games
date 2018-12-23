import numpy as np

from social_dilemmas.constants import CLEANUP_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS, ORIENTATIONS
from social_dilemmas.envs.agent import CleanupAgent

# TODO(ev) add waste colors
COLOURS = {' ': [0, 0, 0],  # Black background
           '': [195, 0, 255],  # Board walls
           '@': [195, 0, 255],  # Board walls
           'A': [0, 255, 0],  # Green apples
           'P': [0, 255, 255],  # Yellow player
           'F': [255, 255, 0],  # Light blue firing beam
           'S': [0, 0, 255],  # Dark blue stream cell
           'H': [17, 56, 100],  # brown waste cells
           'R': [255, 140, 0]}  # red river cell # CHANGE COLORS

# Add custom actions to the agent
ACTIONS['FIRE'] = 5

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class CleanupEnv(MapEnv):

    def __init__(self, ascii_map=CLEANUP_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, COLOURS, num_agents, render)

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get('H', 0) + counts_dict.get('R', 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        self.wall_points = []
        self.firing_points = []
        self.hidden_apples = []
        self.hidden_river = []
        self.hidden_stream = []
        self.hidden_agents = []
        self.hidden_waste = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'B':
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
                elif self.base_map[row, col] == 'S':
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == 'H':
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] == 'H' or self.base_map[row, col] == 'R':
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == 'R':
                    self.river_points.append([row, col])

    def custom_reset(self):
        """Initialize the walls and the waste"""
        self.firing_points = []
        self.hidden_apples = []
        self.hidden_river = []
        self.hidden_stream = []
        self.hidden_agents = []
        self.hidden_waste = []
        self.build_walls()
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

    def clean_map(self):
        """Clean map of elements that should be removed. Executed every step"""
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
            elif self.firing_points[i] in self.hidden_river:
                self.map[row, col] = 'R'
            elif self.firing_points[i] in self.hidden_stream:
                self.map[row, col] = 'S'
            else:
                self.map[row, col] = ' '
        self.hidden_apples = []
        self.firing_points = []
        self.hidden_agents = []
        self.hidden_river = []

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

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            agent = CleanupAgent(agent_id, self.spawn_point(), self.spawn_rotation(), self)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_apple_spawn_prob:
                spawn_points.append((row, col, 'A'))
        for i in range(len(self.waste_points)):
            row, col = self.waste_points[i]
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_waste_spawn_prob:
                spawn_points.append((row, col, 'H'))
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
        for i in range(len(new_apple_points)):
            row, col = new_apple_points[i]
            if self.map[row, col] != 'P' and self.map[row, col] != 'F':
                self.map[row, col] = 'A'
            elif self.map[row, col] == 'F' and [row, col] not in self.hidden_agents:
                self.hidden_apples.append([row, col])

    def update_map_river(self, new_river_points):
        for i in range(len(new_river_points)):
            row, col = new_river_points[i]
            self.map[row, col] = 'R'

    def update_map_stream(self, new_stream_points):
        for i in range(len(new_stream_points)):
            row, col = new_stream_points[i]
            self.map[row, col] = 'S'

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
                    if self.map[next_cell[0], next_cell[1]] == 'A':
                        self.hidden_apples.append([next_cell[0], next_cell[1]])
                    elif self.map[next_cell[0], next_cell[1]] == 'P':
                        self.hidden_agents.append([next_cell[0], next_cell[1]])
                    elif self.map[next_cell[0], next_cell[1]] == 'R':
                        self.hidden_river.append([next_cell[0], next_cell[1]])
                    elif self.map[next_cell[0], next_cell[1]] == 'S':
                        self.hidden_stream.append([next_cell[0], next_cell[1]])
                    self.map[next_cell[0], next_cell[1]] = 'F'
                    firing_points.append((next_cell[0], next_cell[1], 'F'))
                    pos += firing_direction
                else:
                    break
        return firing_points

    # TODO(ev) this is duplicated in Harvest so it should be moved into MapEnv
    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.map[row, col] = '@'
