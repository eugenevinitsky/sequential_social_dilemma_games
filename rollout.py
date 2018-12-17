from social_dilemmas.envs.harvest import HarvestEnv
import numpy as np

if __name__=='__main__':
    horizon = 500
    actions = []
    rewards = []
    observations = []
    env = HarvestEnv(num_agents=1)
    env.reset()
    for i in range(horizon):
        rand_action = np.random.randint(8)
        obs, rew, dones, info, = env.step({'agent-0': rand_action})
        observations.append(obs['agent-0'])
        rewards.append(rew['agent-0'])
