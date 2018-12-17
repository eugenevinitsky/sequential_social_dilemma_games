"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

from social_dilemmas.envs.harvest import HarvestEnv
import numpy as np

# TODO: Agents incorporated and controlled from here. 

class Controller:

    def __init__(self):
        self.env = HarvestEnv(num_agents=1, render=True)
        self.env.reset()

    def rollout(self, horizon=500, render_full_vid=True):
        actions = []
        rewards = []
        observations = []
        
        for i in range(horizon):
            # TODO: use agent policy not just random actions
            rand_action = np.random.randint(8)
            obs, rew, dones, info, = self.env.step({'agent-0': rand_action})

            print("action", rand_action, "reward", rew['agent-0'])
            self.env.render_map()

            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

if __name__=='__main__':
    c = Controller()
    c.rollout()
    
