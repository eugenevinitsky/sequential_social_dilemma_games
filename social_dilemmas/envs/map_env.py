"""Base map class that defines the rendering process"""

from gym.spaces import Box
from gym import Env
import numpy as np
import six


class MapEnv(Env):

    def __init__(self, row_size, col_size, map, num_agents=1):
        self.row_size = row_size
        self.col_size = col_size
        self.num_agents = num_agents
        self.map = map
        self.agents = []
        for i in range(self.num_agents):
            self.agents.append(self.create_agent('agent-'+str(i)))

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def ascii_to_matrix(self, ascii_map):
        """Construct a numpy array of dtype `uint8` from an ASCII art diagram.
        This function takes ASCII art diagrams (expressed as lists or tuples of
        equal-length strings) and derives 2-D numpy arrays with dtype `uint8`.
        Returns:
          A 2-D numpy array as described.
        Raises:
          ValueError: self.map wasn't an ASCII art diagram, as described; this could be
            because the strings it is made of contain non-ASCII characters, or do not
            have constant length.
          TypeError: self.amp was not a list of strings.
        """
        error_text = (
            'the argument to ascii_art_to_uint8_nparray must be a list (or tuple) '
            'of strings containing the same number of strictly-ASCII characters.')
        try:
            mat = np.vstack(np.fromstring(line, dtype=np.uint8) for line in ascii_map)
        except ValueError as e:
            raise ValueError('{} (original error from numpy: {})'.format(error_text, e))
        except TypeError as e:
            if isinstance(self.map, (list, tuple)) and not all(
                    isinstance(row, six.string_types) for row in ascii_map):
                error_text += ' Did you pass a list of list of single characters?'
            raise TypeError('{} (original error from numpy: {})'.format(error_text, e))
        if np.any(self.map > 127): raise ValueError(error_text)
        return mat

    def step(self, actions):
        """Takes in a list of actions and converts them to a map update

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """
        agent_actions = []
        for agent, action in zip(self.agents, actions):
            agent_action = agent.action_map(action)
            agent_actions.append((agent.agent_id, agent_action))
        self.update_map(agent_actions)
        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents:
            observations[agent.agent_id] = agent.get_state()
            rewards[agent.agent_id] = agent.get_state()
            dones[agent.agent_id] = agent.get_done()
        return observations, rewards, done, info

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
        pass

    def update_map(self, agent_actions):
        """Takes in a list of agent_action tuples and returns a new map """
        pass

    def create_agent(self, agent_id):
        raise NotImplementedError


