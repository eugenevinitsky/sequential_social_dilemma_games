import unittest

import numpy as np
from pettingzoo.test import api_test, parallel_api_test

from social_dilemmas.envs.pettingzoo_env import MAX_CYCLES
from social_dilemmas.envs.pettingzoo_env import env as aec_env
from social_dilemmas.envs.pettingzoo_env import parallel_env


class PettingZooTest(unittest.TestCase):
    def test_parallel(self):
        env = parallel_env(max_cycles=MAX_CYCLES, env="harvest", num_agents=2)
        env.seed()
        env.reset()
        n_act = env.action_space("agent-0").n
        for _ in range(MAX_CYCLES * env.num_agents):
            actions = {agent: np.random.randint(n_act) for agent in env.agents}
            _, _, _, _ = env.step(actions)
            if not env.agents:
                _ = env.reset()
        parallel_api_test(env, MAX_CYCLES)

    def test_aec(self):
        env = aec_env(max_cycles=MAX_CYCLES, env="harvest", num_agents=2)
        env.seed(0)
        env.reset()
        n_act = env.action_space("agent-0").n
        for agent in env.agent_iter(max_iter=MAX_CYCLES * env.num_agents):
            _, _, _, _ = env.last()
            action = np.random.randint(n_act)
            env.step(action)
            if not env.agents:
                env.reset()
        api_test(env, MAX_CYCLES)


if __name__ == "__main__":
    unittest.main()
