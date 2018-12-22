from social_dilemmas.constants import CLEANUP_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS
from social_dilemmas.envs.agent import CleanupAgent

# TODO(ev) add waste colors
COLOURS = {' ': [0, 0, 0],  # Black background
           '': [195, 0, 255],  # Board walls
           '@': [195, 0, 255],  # Board walls
           'A': [0, 255, 0],  # Green apples
           'P': [0, 255, 255],  # Yellow player
           'F': [255, 255, 0], # Light blue firing beam
           'S': [0, 0, 255],  # Dark blue stream cell
           'H': [139,69,19]} # brown waste cells

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

    def custom_reset(self):
        """Reset custom elements of the map"""
        raise NotImplementedError

    def custom_action(self, agent):
        """Allows agents to take actions that are not move or turn"""
        raise NotImplementedError

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        # spawn the apples
        new_apples_and_waste = self.spawn_apples_and_waste()
        if len(new_apples_and_waste) > 0:
            self.reserved_slots += new_apples_and_waste

    def clean_map(self):
        """Clean map of elements that should be removed. Executed every step/"""
        raise NotImplementedError

    def execute_custom_reservations(self):
        """Execute reserved slots that do not have to do with moving"""
        raise NotImplementedError

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            agent = CleanupAgent(agent_id, self.spawn_point(), self.spawn_rotation(), self, 3)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        pass