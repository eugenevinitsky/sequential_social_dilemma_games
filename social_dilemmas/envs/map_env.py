"""Base map class that defines the rendering process


Code partially adapted from PyColab: https://github.com/deepmind/pycolab
"""
from ray.rllib.env import MultiAgentEnv
import numpy as np
import matplotlib.pyplot as plt

ACTIONS = {'MOVE_LEFT': [-1, 0],  # Move left
           'MOVE_RIGHT': [1, 0],  # Move right
           'MOVE_UP': [0, -1],  # Move up
           'MOVE_DOWN': [0, 1],  # Move down
           'STAY': [0, 0],  # don't move
           'TURN_CLOCKWISE': [[0, -1], [1, 0]],  # Rotate counter clockwise
           'TURN_COUNTERCLOCKWISE': [[0, 1], [-1, 0]]}  # Move right

ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, -1],
                'DOWN': [0, 1]}

# the axes look like
# graphic is here to help me get my head in order
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
#         ^
#         |
#         U
#         P
# <--LEFT*RIGHT---->
#         D
#         O
#         W
#         N
#         |


class MapEnv(MultiAgentEnv):

    def __init__(self, ascii_map, color_map, num_agents=1, render=True):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        color_map: dict
            Specifies how to convert between ascii chars and colors
        num_agents: int
            Number of agents to have in the system.
        render: bool
            Whether to render the environment
        """
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        # keeps track of positions that agents have said they want to move to
        # as well as the intended action in that slot
        self.reserved_slots = []
        self.agents = {}
        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.render = render
        self.color_map = color_map
        self.spawn_points = []  # where agents can appear

    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array


        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """
        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        self.update_map(agent_actions)

        self.custom_map_update()
        self.execute_reservations()
        self.reserved_slots = []

        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
        dones["__all__"] = np.any(dones.items())
        return observations, rewards, dones, info

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        self.reserved_slots = []
        self.reset_map()
        self.custom_map_update()

        observations = {}
        for agent in self.agents.values():
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
        return observations

    def map_to_colors(self, map=None, color_map=None):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if map is None:
            map = self.map
        if color_map is None:
            color_map = self.color_map

        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]
        return rgb_arr

    def render_map(self):
        if self.render:
            rgb_arr = self.map_to_colors()
            plt.imshow(rgb_arr, interpolation='nearest')
            plt.show()

    def update_map(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """

        # TODO(ev) split into three methods: clean(), update_map, custom_update_map
        self.clean_map()

        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) these two parts of the actions
            if 'MOVE' in action or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                self.reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_map_agent_rot(new_rot)
            else:
                self.custom_action(agent)

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.setup_agents()
        self.custom_reset()

    def custom_reset(self):
        """Reset custom elements of the map"""
        pass

    def custom_action(self, agent):
        """Allows agents to take actions that are not move or turn"""
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def clean_map(self):
        """Clean map of elements that should be removed"""
        pass

    def execute_reservations(self):
        """Takes all the reserved slots and decides which move has priority"""
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        agent_by_pos = {tuple(agent.get_pos()): agent.agent_id for agent in self.agents.values()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in self.reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        # First, resolve conflicts between two agents that want the same spot
        if len(agent_to_slot) > 0:

            # a random agent will win the slot
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)
            # if there are any conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        self.agents[agent_to_slot[index]].update_map_agent_pos(move)
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos().tolist()

            for agent_id, move in agent_moves.items():
                if move in curr_agent_pos:
                    # find the agent that is currently at that spot, check where they will be next
                    # if they're going to move away, go ahead and move into their spot
                    conflicting_agent_id = agent_by_pos[tuple(move)]
                    # a STAY command has been issued or the other agent hasn't been issued a command,
                    # don't do anything
                    if agent_id == conflicting_agent_id or \
                            conflicting_agent_id not in agent_moves.keys():
                        continue
                    elif agent_moves[conflicting_agent_id] != move:
                        self.agents[agent_id].update_map_agent_pos(move)
                else:
                    self.agents[agent_id].update_map_agent_pos(move)

        self.execute_custom_reservations()
        self.reserved_slots = []

    def execute_custom_reservations(self):
        """Execute reserved slots that do not have to do with moving"""
        raise NotImplementedError

    def setup_agents(self):
        raise NotImplementedError

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent.

        Parameters
        ----------
        agent_id: str
            name that they agent should be assigned

        Returns
        -------
        agent: Agent
            constructed agent

        """
        raise NotImplementedError

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        not_occupied = False
        rand_int = 0
        # select a spawn point
        # replace this with an operation over a set
        while not not_occupied:
            num_ints = len(self.spawn_points)
            rand_int = np.random.randint(num_ints)
            spawn_point = self.spawn_points[rand_int]
            if self.map[spawn_point[0], spawn_point[1]] != 'P':
                not_occupied = True
        return np.array(self.spawn_points[rand_int])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    ########################################
    # Utility methods, move these eventually
    ########################################

    def return_view(self, pos, row_size, col_size):
        """Given an  position and view window, returns correct map part

        Note, if the agent asks for a view that exceeds the map bounds,
        it is padded with zeros

        Parameters
        ----------
        pos: list
            list consisting of row and column at which to search
        row_size: int
            how far the view should look in the row dimension
        col_size: int
            how far the view should look in the col dimension

        Returns
        -------
        view: (np.ndarray) - a slice of the map for the agent to see
        """
        x, y = pos
        left_edge = x - col_size
        right_edge = x + col_size
        top_edge = y - row_size
        bot_edge = y + row_size
        pad_mat, left_pad, top_pad = self.pad_if_needed(left_edge, right_edge,
                                                        top_edge, bot_edge, self.map)
        x += left_pad
        y += top_pad
        view = pad_mat[x - col_size: x + col_size + 1,
                       y - row_size: y + row_size + 1]
        return view

    def pad_if_needed(self, left_edge, right_edge, top_edge, bot_edge, matrix):
        # FIXME(ev) something is broken here, I think x and y are flipped
        row_dim = matrix.shape[0]
        col_dim = matrix.shape[1]
        left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
        if left_edge < 0:
            left_pad = abs(left_edge)
        if right_edge > row_dim - 1:
            right_pad = right_edge - (row_dim - 1)
        if top_edge < 0:
            top_pad = abs(top_edge)
        if bot_edge > col_dim - 1:
            bot_pad = bot_edge - (col_dim - 1)

        return self.pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, \
            top_pad

    def pad_matrix(self, left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
        pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                         'constant', constant_values=(const_val, const_val))
        return pad_mat

    # TODO(ev) this can be a general property of map_env or a util
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

    # TODO(ev) this should be an agent property
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

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.map.shape[1]:
            return False
        else:
            return True
