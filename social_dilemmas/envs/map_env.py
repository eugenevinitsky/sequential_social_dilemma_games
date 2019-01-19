"""Base map class that defines the rendering process


Code partially adapted from PyColab: https://github.com/deepmind/pycolab
"""
from ray.rllib.env import MultiAgentEnv
import numpy as np
import matplotlib.pyplot as plt
import utility_funcs as util

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

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

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

    def __init__(self, ascii_map, num_agents=1, render=True, color_map=None):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        num_agents: int
            Number of agents to have in the system.
        render: bool
            Whether to render the environment
        color_map: dict
            Specifies how to convert between ascii chars and colors
        """
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.map = np.full((len(self.base_map), len(self.base_map[0])), ' ')

        # keeps track of positions that agents have said they want to move to
        # as well as the intended action in that slot
        self.reserved_slots = []
        self.agents = {}
        # keeps track of cell types that should not be over-ridden by agents
        self.no_update_cells = []

        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS
        self.spawn_points = []  # where agents can appear

        # map that is revealed after cleaning
        self.post_clean_map = None

        # cells hidden by agents or other actions, elements are [row, pos, str]
        self.hidden_cells = []
        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        self.setup_agents()

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

        # move
        self.clean_map()
        self.post_clean_map = self.map.copy()
        self.update_moves(agent_actions)
        self.execute_reservations()

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)
        self.execute_reservations()

        # execute spawning events
        self.custom_map_update()
        self.execute_reservations()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            agent.grid = util.return_view(map_with_agents, agent.pos,
                                          agent.row_size, agent.col_size)
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
            rewards[agent.agent_id] = agent.compute_reward()
            dones[agent.agent_id] = agent.get_done()
        dones["__all__"] = np.any(list(dones.values()))
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
        self.hidden_cells = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.post_clean_map = self.map.copy()
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        for agent in self.agents.values():
            agent.grid = util.return_view(map_with_agents, agent.pos,
                                          agent.row_size, agent.col_size)
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
        return observations

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.map)

        for agent_id, agent in self.agents.items():
            char_id = str(int(agent_id[-1]) + 1)

            # If agent is not within map, skip.
            if not(agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                    agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            # make sure that we don't override things like firing
            # cells with agents
            if self.map[agent.pos[0], agent.pos[1]] not in \
                    self.no_update_cells:
                grid[agent.pos[0], agent.pos[1]] = char_id

        return grid

    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[str(i+1)] != 1:
                print('Error! Wrong number of agent', i, 'in map!')
                return False
        return True

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

    def render(self, filename=None):
        """ Creates an image of the map to plot or save.

        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents)
        plt.imshow(rgb_arr, interpolation='nearest')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """

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
                agent.update_agent_rot(new_rot)

    def update_custom_moves(self, agent_actions):
        for agent_id, action in agent_actions.items():
            # check its not a move based action
            if 'MOVE' not in action and 'STAY' not in action and 'TURN' not in action:
                agent = self.agents[agent_id]
                self.custom_action(agent, action)

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        for agent in self.agents.values():
            pos = agent.get_pos()
            row, col = pos
            # TODO(ev) this rendering logic should not be done here)
            self.map[row, col] = 'P'
            self.append_hiddens(pos.tolist(), ' ', 'P')
        self.build_walls()
        self.custom_reset()

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass

    def custom_action(self, agent, action):
        """Add reservations to self.reserved_slots for actions that are not move or turn.
        For example, if an agent can fire, you can add (row, col, 'F')
        to indicate that F should be placed at that point"""
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions. For example, you can add
        (row, col, 'A') to env.reserved_slots to indicate an apple should be placed at that point"""
        pass

    def clean_map(self):
        """Place back all hidden cells that are not currently blocked by an agent"""
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        hidden_pos = [hidden[0:2] for hidden in self.hidden_cells]
        hidden_char = [hidden[2] for hidden in self.hidden_cells]
        for i, hidden in enumerate(hidden_pos):
            # you can't put back hidden cells that an agent is on unless it is an agent that is
            # hidden
            if hidden not in curr_agent_pos or hidden_char[i] == 'P':
                row, col = hidden
                self.map[row, col] = hidden_char[i]
                index = self.hidden_cells.index(hidden + [hidden_char[i]])
                del self.hidden_cells[index]

    def execute_custom_reservations(self):
        """Execute reserved slots that do not have to do with moving agents. For example,
        placing apples or placing the fired beam. """
        raise NotImplementedError

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    def append_hiddens(self, new_pos, old_char, new_char):
        """Add hidden cells to self.hidden_cells that should be put back

        Parameters
        ----------
        new_pos: list
            the position the new char is going to be placed at
        old_char: str
            the character that will be hidden
        new_char: str
            the character that will replace it
        """
        raise NotImplementedError

    def execute_reservations(self):
        """Takes all the reserved slots and make the moves. Handles conflicting reservations.

           This method works by finding all conflicts over a cell and randomly assigning them
           to one of the agents that desires the slot. It then sets all of the other agents
           that wanted the cell to have a move of staying. For moves that do not directly
           conflict with another agent for a cell, but may not be temporarily resolvable
           due to an agent currently being in the desired cell, we continually loop through
           the actions until all moves have been satisfied or deemed impossible.
           For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
           Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
           first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
           then be able to move into [2,2].
        """
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

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:

            # first we will resolve all slots over which multiple agents
            # want the slot

            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        hidden_pos = [hidden_cell[0: 2] for hidden_cell in self.hidden_cells]
                        hidden_char = [hidden_cell[2] for hidden_cell in self.hidden_cells]

                        # TODO(ev) this should be a method from ---- to -----
                        # -------------------------------------
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in curr_agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].get_pos().tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id].\
                                    get_pos().tolist()
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent_id not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move.tolist() == self.agents[conflicting_agent_id]\
                                            .get_pos().tolist():
                                        conflict_cell_free = False

                        if conflict_cell_free:
                            new_pos, old_pos = \
                                self.agents[agent_to_slot[index]].update_agent_pos(move)
                            new_pos = new_pos.tolist()
                            old_pos = old_pos.tolist()
                            hidden_pos_arr = np.array(hidden_pos)
                            search_rows = np.where((hidden_pos_arr == old_pos)
                                                   .all(axis=1))[0].tolist()
                            # only put back and delete elements that are not 'P'
                            found_index = 0
                            for index in search_rows:
                                if hidden_char[index] != 'P':
                                    found_index = index
                                    break
                            self.map[old_pos[0], old_pos[1]] = self.hidden_cells[found_index][2]
                            del self.hidden_cells[found_index]
                            char = self.map[new_pos[0], new_pos[1]]
                            # FIXME(ev) this can cause you
                            if char != 'P':
                                self.append_hiddens(new_pos, char, 'P')
                            self.map[new_pos[0], new_pos[1]] = 'P'
                            # update the positions for resolving future conflicts
                            curr_agent_pos = [agent.get_pos().tolist() for
                                              agent in self.agents.values()]
                            agent_by_pos = {tuple(agent.get_pos()):
                                            agent.agent_id for agent in self.agents.values()}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos().tolist()

            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    hidden_pos = [hidden_cell[0: 2] for hidden_cell in self.hidden_cells]
                    hidden_char = [hidden_cell[2] for hidden_cell in self.hidden_cells]
                    if move in curr_agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].get_pos().tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].get_pos().tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if agent_moves[conflicting_agent_id] == curr_pos and \
                                    move == self.agents[conflicting_agent_id].get_pos().tolist():
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        new_pos, old_pos = self.agents[agent_id].update_agent_pos(move)
                        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
                        new_pos = new_pos.tolist()
                        old_pos = old_pos.tolist()

                        # Don't delete and update your old cell and hidden info
                        # if there's an agent there, they should take over the hidden info
                        if old_pos not in curr_agent_pos:
                            hidden_pos_arr = np.array(hidden_pos)
                            search_rows = np.where((hidden_pos_arr == old_pos).all(axis=1))[0].tolist()
                            # only put back and delete elements that are not 'P'
                            found_index = 0
                            for index in search_rows:
                                if hidden_char[index] != 'P':
                                    found_index = index
                                    break
                            self.map[old_pos[0], old_pos[1]] = self.hidden_cells[found_index][2]
                            del self.hidden_cells[found_index]
                            char = self.map[new_pos[0], new_pos[1]]
                            self.append_hiddens(new_pos, char, 'P')
                        self.map[new_pos[0], new_pos[1]] = 'P'
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

        self.execute_custom_reservations()
        self.reserved_slots = []

    def clear_impossible_moves(self):
        """Given a proposed move, check if the cell that will be moved into is possible
           If not,
        """

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        not_occupied = False
        rand_int = 0
        # select a spawn point
        # replace this with an operation over a set
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        while not not_occupied:
            # TODO(ev), this is lazy, only spawn numbers that are valid
            num_ints = len(self.spawn_points)
            rand_int = np.random.randint(num_ints)
            spawn_point = self.spawn_points[rand_int]
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                not_occupied = True
        return np.array(self.spawn_points[rand_int])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.map[row, col] = '@'

    ########################################
    # Utility methods, move these eventually
    ########################################

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
