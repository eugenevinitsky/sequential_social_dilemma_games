from gym.spaces import Box
import numpy as np
import six

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.constants import HARVEST_MAP
from social_dilemmas.envs.map_env import MapEnv


APPLE_RADIUS = 2

COLOURS = {' ': [0, 0, 0],        # Black background
           '@': [764, 0, 999],    # Board walls
           'A': [0, 999, 0],      # Green apples
           'P': [0, 999, 999],    # Player
           'F': [999, 999, 0]}    # Yellow firing beam

# use keyword names so that it's easy to understand what the agent is calling
ACTIONS = {'MOVE_LEFT':  [-1, 0],  # Move left
           'MOVE_RIGHT': [1, 0],   # Move right
           'MOVE_UP':    [0, 1],   # Move up
           'MOVE_DOWN':  [0, -1],  # Move down
           'STAY':       [0, 0],   # don't move
           'FIRE_LEFT':  [-1, 0],  # Move left
           'FIRE_RIGHT': [1, 0],   # Move right
           'FIRE_UP':    [0, 1],   # Move up
           'FIRE_DOWN':  [0, -1],  # Move down
}


class HarvestEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False):
        super.__init__(ascii_map, num_agents, render)

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
        self.place_agents()
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
            # TODO(ev) updating the agents has to be synchronous
            # TODO(ev) do we overlay firing over the agent or what?
            if 'MOVE' or 'STAY' in action:
                new_pos = agent.get_pos() + selected_action
                self.update_map_agent_pos(agent.get_pos(), new_pos)
                self.agents[agent_id].update_pos(new_pos)
            else:
                self.update_map_fire(agent.get_pos(), selected_action)

        # spawn the apples
        # FIXME(ev) there should be an empty map that we add all new actions to
        # rather than repeating
        new_apples = self.spawn_apples()
        self.update_map_apples(new_apples)

    def create_agent(self, agent_id, *args):
        """Takes an agent id and agents args and returns an agent"""
        return HarvestAgent(agent_id, self.spawn_point(), self.map)

    def spawn_apples(self):
        raise NotImplementedError

    def spawn_point(self):
        """Returns a randomly selected spawn point"""
        pass

    def update_map_agent_pos(self, old_pos, new_pos):
        pass

    def update_map_fire(self, firing_pos, firing_direction):
        pass

    def update_map_apples(self, new_apple_map):