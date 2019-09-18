import numpy as np

from social_dilemmas.envs.agent import SwitchAgent
from social_dilemmas.constants import SWITCH_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS


# Add custom actions to the agent
ACTIONS['TOGGLE_SWITCH'] = 1  # length of firing range

# Custom colour dictionary
SWITCH_COLORS = {'D': [183, 128, 0],  # Brown closed door
                 'd': [255, 239, 201],  # Light brown opened door
                 'S': [50, 255, 50],  # Green turned-on switch
                 's': [255, 0, 34]}  # Red turned-off switch

GIVE_EXTERNAL_SWITCH_REWARD = int(True)

class SwitchEnv(MapEnv):

    def __init__(self, ascii_map=SWITCH_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, num_agents, render)
        self.initial_map_state = dict()
        self.switch_locations = []
        self.door_locations = []
        self.switch_count = 0
        self.prev_activated_switch_count = 0
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                current_char = self.base_map[row, col]
                if current_char in ['s', 'S', 'd', 'D']:
                    self.initial_map_state[row, col] = current_char
                # Remember switch/door locations for faster access
                if current_char in ['s', 'S']:
                    self.switch_locations.append((row, col))
                    self.switch_count += 1
                if current_char in ['d', 'D']:
                    self.door_locations.append((row, col))

        self.color_map.update(SWITCH_COLORS)

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = SwitchAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the switches/doors"""
        for coordinates, char in self.initial_map_state.items():
            self.world_map[coordinates[0], coordinates[1]] = char
        self.prev_activated_switch_count = 0

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       fire_len=ACTIONS['TOGGLE_SWITCH'],
                                       fire_char='F',
                                       cell_types=['s', 'S'],
                                       update_char=['S', 's'],
                                       beam_width=1)
        return updates

    def custom_map_update(self):
        activated_switch_count = 0
        for row, col in self.switch_locations:
            if self.world_map[row, col] == 'S':
                activated_switch_count += 1

        temp_reward = (activated_switch_count - self.prev_activated_switch_count) * GIVE_EXTERNAL_SWITCH_REWARD
        self.prev_activated_switch_count = activated_switch_count

        for agent in list(self.agents.values()):
            agent.reward_this_turn += temp_reward

        # Open doors if all switches have been activated
        open_doors = activated_switch_count == self.switch_count
        door_char = 'd' if open_doors else 'D'
        updates = []
        for row, col in self.door_locations:
            updates.append((row, col, door_char))
        self.update_map(updates)

    def count_switches(self, window):
        # Compute how many switches are activated
        # Testing function
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_switches = counts_dict.get('S', 0)
        return num_switches
