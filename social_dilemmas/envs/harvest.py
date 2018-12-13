from gym.spaces import Box
import numpy as np
import six

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv

APPLE_RADIUS = 2

COLOURS = {' ': [0, 0, 0],  # Black background
           '@': [764, 0, 999],  # Board walls
           'A': [0, 999, 0],  # Green apples
           'P': [0, 999, 999],  # Player #FIXME(ev) agents need to have different colors
           'F': [999, 999, 0]}  # Yellow firing beam

# use keyword names so that it's easy to understand what the agent is calling
ACTIONS = {'MOVE_LEFT': [-1, 0],  # Move left
           'MOVE_RIGHT': [1, 0],  # Move right
           'MOVE_UP': [0, 1],  # Move up
           'MOVE_DOWN': [0, -1],  # Move down
           'STAY': [0, 0],  # don't move
           'TURN_CLOCKWISE': [[0, 1], [-1, 0]],  # Rotate counter clockwise
           'TURN_COUNTERCLOCKWISE': [[0, -1], [1, 0]],  # Rotate clockwise
           'FIRE': 5}  # Fire 5 squares forward #FIXME(ev) is the firing in a straight line?

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, 1],
                'DOWN': [0, -1]}


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, COLOURS, num_agents, render)
        # set up the list of spawn points
        self.spawn_points = []
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

    # FIXME(ev) action_space should really be defined in the agents
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

    def setup_map(self):
        self.spawn_apples()
        # self.place_agents()
        # FIXME(eugene) move what is below into place agents
        for agent in self.agents.values():
            new_pos = self.spawn_point()
            agent.set_pos(new_pos)

    def update_map(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions

        Returns
        -------
        new_map: numpy ndarray
            the updated map to store
        agent_pos: dict of tuples with keys as agent ids
        """

        # Move the agents
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) updating the agents has to be synchronous? I think?
            # TODO(ev) do we overlay firing over the agent or what?
            if 'MOVE' or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                self.update_map_agent_pos(agent.get_pos(), new_pos)
                self.agents[agent_id].update_pos(new_pos)
            elif 'TURN' in action:
                agent_rot = ORIENTATIONS[agent.get_orientation()]
                new_rot = np.dot(ACTIONS[action], agent_rot)
                self.update_map_agent_rot(agent.get_pos, new_rot)
            else:
                self.update_map_fire(agent.get_pos(), agent.get_orientation())

        # TODO(ev) there should be an empty map that we add all new actions to
        # TODO(ev) doing agents, than fire, than apples is 3x as slow
        # FIXME(EV) define a sum operation on these numpy matrices that let us "add" the strings

        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map_apples(new_apples)

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent"""
        return HarvestAgent(agent_id, self.spawn_point(), self.spawn_rotation, self.map)

    def spawn_apples(self):
        # iterate over the spawn points in self.ascii_map and compare it with
        # current points in self.map

        # FIXME(ev) magic number
        l2_dist = 2
        # first pad the matrix so that we can iterate through nicely
        pad_mat = self.pad_matrix(2, 2, 2, 2, self.map, ' ')
        new_map = np.zeros(self.map.shape)
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            window = pad_mat[row - l2_dist + 1:row + l2_dist,
                     col - l2_dist + 1:col + l2_dist]
            # compute how many apples are in window
            unique, counts = np.unique(window, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            num_apples = counts_dict['A']
            spawn_prob = SPAWN_PROB[min(num_apples, 3)]
            rand_num = np.random.rand(1)[0]
            if rand_num < spawn_prob:
                new_map[row, col] = 'A'
        return new_map

    def spawn_point(self):
        """Returns a randomly selected spawn point"""

        # select a spawn point
        num_ints = len(self.spawn_points)
        rand_int = np.random.randint(num_ints)
        return self.spawn_points[rand_int]

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def update_map_agent_pos(self, old_pos, new_pos):
        self.map[old_pos] = ' '
        self.map[new_pos] = 'P'

    def update_map_agent_rot(self, old_pos, new_rot):
        self.map[old_pos] = ' '
        # FIXME(ev) once we have a color scheme worked out we need to convert rotation
        # into a color
        self.map[old_pos] = 'P'

    def update_map_fire(self, firing_pos, firing_orientation):
        num_fire_cells = 5
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        for i in range(num_fire_cells):
            # FIXME(ev) this needs to be passed a set of indices
            next_cell = start_pos + firing_direction
            self.map[next_cell[0], next_cell[1]] = 'F'
            start_pos += firing_direction

    def update_map_apples(self, new_apple_map):
        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):
                if new_apple_map[row, col] == 'A' and self.map[row, col] != 'P':
                    # FIXME(ev) what if a firing beam is here at this time?
                    self.map[row, col] = 'A'

    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(ACTIONS['ROTATE_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['ROTATE_CLOCKWISE'], action_vec)
