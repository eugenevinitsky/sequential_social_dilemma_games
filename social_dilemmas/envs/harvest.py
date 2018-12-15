from gym.spaces import Box
import numpy as np
import six

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv

APPLE_RADIUS = 2

COLOURS = {' ': [0, 0, 0],  # Black background
           '': [764, 0, 999],  # Board walls
           '@': [764, 0, 999],  # Board walls
           'A': [0, 999, 0],  # Green apples
           'P': [0, 999, 999],  # Player #FIXME(ev) agents need to have different colors
           'F': [999, 999, 0]}  # Yellow firing beam

# use keyword names so that it's easy to understand what the agent is calling
ACTIONS = {'MOVE_LEFT':             [-1, 0],  # Move left
           'MOVE_RIGHT':            [1, 0],   # Move right
           'MOVE_UP':               [0, 1],   # Move up
           'MOVE_DOWN':             [0, -1],  # Move down
           'STAY':                  [0, 0],   # don't move
           'TURN_CLOCKWISE':        [[0, 1], [-1, 0]],  # Rotate counter clockwise
           'TURN_COUNTERCLOCKWISE': [[0, -1], [1, 0]],   # Move right
           'FIRE': 5}               # Fire 5 squares forward #FIXME(ev) is the firing in a straight line?

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, 1],
                'DOWN': [0, -1]}

# FIXME(ev) this whole thing is in serious need of some abstraction
# FIXME(ev) switching betewen types and lists in a pretty arbitrary manner


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, COLOURS, num_agents, render)
        # set up the list of spawn points
        self.spawn_points = []
        self.apple_points = []
        self.wall_points = []
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
            self.agents[agent_id] = self.create_agent(agent_id)

    # TODO(ev) this can probably be moved into the superclass
    def reset_map(self):
        self.map = np.full((len(self.base_map), len(self.base_map[0])), ' ')

        self.build_walls()
        self.update_map_apples(self.spawn_apples())

        for agent in self.agents.values():
            agent.update_map_agent_pos(self.spawn_point())
            agent.update_map_agent_rot(self.spawn_rotation())

    # FIXME(ev) most of this is general and can be moved, only apples need to be done here
    def update_map(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        Returns
        -------
        new_map: numpy ndarray
            the updated map to store
        agent_pos: list of tuples with keys as agent ids
        """

        # FIXME(ev) walls are not showing up in the map
        # Move the agents
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) updating the agents has to be synchronous?
            # TODO(ev) for example, an agent may try to walk through another agent
            # TODO(ev) which is fine if the other agent is going to move
            # TODO(ev) do we overlay firing over the agent or what?
            if 'MOVE' in action or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                agent.update_map_agent_pos(new_pos)
            elif 'TURN' in action:
                # FIXME(ev) move into a utility method
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_map_agent_rot(new_rot)
            else:
                self.update_map_fire(agent.get_pos(), agent.get_orientation())

        # TODO(ev) there should be an empty map that we add all new actions to
        # TODO(ev) doing agents, than fire, than apples is 3x as slow
        # FIXME(EV) define a sum operation on these numpy matrices that let us "add" the strings

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map_apples(new_apples)

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent"""
        # FIXME(ev) the agent window is currently a magic number
        return HarvestAgent(agent_id, self.spawn_point(), self.spawn_rotation(), self, 3)

    def spawn_apples(self):
        # iterate over the spawn points in self.ascii_map and compare it with
        # current points in self.map

        # first pad the matrix so that we can iterate through nicely
        # FIXME(ev) you shouldn't be doing the padding yourself here, this should be done
        # by a utility method
        pad_mat= self.pad_matrix(*[APPLE_RADIUS]*4, self.map)
        new_map = np.zeros(self.map.shape)
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            if self.base_map[row, col] == 'A':
                # FIXME(ev) this padding probably needs to be moved into a method
                row += APPLE_RADIUS
                row += APPLE_RADIUS
                window = pad_mat[row - APPLE_RADIUS:row + APPLE_RADIUS,
                         col - APPLE_RADIUS:col + APPLE_RADIUS]
                # compute how many apples are in window
                unique, counts = np.unique(window, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                num_apples = counts_dict.get('A', 0)
                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_map[row, col] = 'A'
        return new_map

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.map[row, col] = '@'

    # FIXME(ev) this is probably shared by every env
    def spawn_point(self):
        """Returns a randomly selected spawn point"""
        not_occupied = False
        rand_int = 0
        # select a spawn point
        while not not_occupied:
            num_ints = len(self.spawn_points)
            rand_int = np.random.randint(num_ints)
            spawn_point = self.spawn_points[rand_int]
            # FIXME(ev) this will break when we implement rotation colors
            if self.map[spawn_point[0], spawn_point[1]] != 'P':
                not_occupied = True
        return np.array(self.spawn_points[rand_int])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def update_map_fire(self, firing_pos, firing_orientation):
        num_fire_cells = 5
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        for i in range(num_fire_cells):
            next_cell = start_pos + firing_direction
            if self.test_if_in_bounds(next_cell):
                self.map[next_cell[0], next_cell[1]] = 'F'
                start_pos += firing_direction
            else:
                break

    # def update_map(self, points_list):
    #     """Takes in a list of tuples consisting of ('row', 'col', 'new_ascii_char' and makes a new map"""

    def update_map_apples(self, new_apple_map):
        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):
                if new_apple_map[row, col] == 'A' and self.map[row, col] != 'P':
                    # FIXME(ev) what if a firing beam is here at this time?
                    self.map[row, col] = 'A'

    # FIXME(ev) this can be a general property of map_env or a util
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
        return np.dot(ACTIONS['TURN_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['TURN_CLOCKWISE'], action_vec)

    # FIXME(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    # FIXME(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.map.shape[1]:
            return False
        else:
            return True