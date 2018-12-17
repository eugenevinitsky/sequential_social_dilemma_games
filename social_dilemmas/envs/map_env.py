"""Base map class that defines the rendering process


Code partially adapted from PyColab: https://github.com/deepmind/pycolab
"""

from gym import Env
import numpy as np
from renderer import CursesUi
import matplotlib.pyplot as plt


class MapEnv(Env):

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
        self.agents = {}
        self.render = render
        self.color_map = color_map
        if render:
            self.renderer = CursesUi({}, 1)

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

    def reset_map(self):
        raise NotImplementedError

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

        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
            rewards[agent.agent_id] = agent.get_reward()
            dones[agent.agent_id] = agent.get_done()
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
        self.reset_map()
        self.custom_map_update()
        # TODO(ev) the agent pos and orientation setting code is duplicated

        observations = {}
        for agent in self.agents.values():
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
        return observations

    def map_to_colors(self, map, color_map):
        """Converts a map to an array of RGB values"""
        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3))
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]
        return rgb_arr

    def render_map(self, mode='human'):
        if self.render:
            rgb_arr = self.map_to_colors(self.map, self.color_map)
            plt.imshow(rgb_arr, interpolation='nearest')
            plt.show()

    def update_map(self, agent_actions):
        """Converts agent action tuples into a new map and new agewnt positions

        Returns
        -------
        new_map: numpy ndarray
            the updated map to store
        agent_pos: dict of tuples with keys as agent ids
        """
        raise NotImplementedError

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        raise NotImplementedError

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent"""
        raise NotImplementedError

    ########################################
    # Utility methods, move these eventually
    ########################################

    def return_view(self, agent_pos, row_size, col_size):
        """Given an agent position and view window, returns correct map part

        Note, if the agent asks for a view that exceeds the map bounds,
        it is padded with zeros

        Parameters
        ----------

        Returns
        -------
        view: (np.ndarray) - a slice of the map for the agent to see
        """
        # FIXME(ev) this might be transposed
        x, y = agent_pos
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
        row_dim = matrix.shape[0]
        col_dim = matrix.shape[1]
        left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
        # TODO(ev) this can be more elegantly written in terms of mins and maxes
        if left_edge < 0:
            left_pad = abs(left_edge)
        if right_edge > col_dim - 1:
            right_pad = right_edge - (col_dim - 1)
        if top_edge < 0:
            top_pad = abs(top_edge)
        if bot_edge > row_dim - 1:
            bot_pad = bot_edge - (row_dim - 1)

        return self.pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, \
            top_pad

    def pad_matrix(self, left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
        pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                         'constant', constant_values=(const_val, const_val))
        return pad_mat