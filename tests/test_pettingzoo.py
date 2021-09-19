from config.default_args import add_default_args
import argparse
import numpy as np
import unittest
from social_dilemmas.envs.pettingzoo_env import MAX_CYCLES, parallel_env

parser = argparse.ArgumentParser()
add_default_args(parser)
args = parser.parse_args()

class PettingZooTest(unittest.TestCase):

    def test(self):
        env = parallel_env(max_cycles=MAX_CYCLES, ssd_args=args)
        env.seed(0)
        env.reset()
        for i in range(MAX_CYCLES):
            agents = env.agents
            actions = {agent: np.random.randint(0, 8) for agent in agents}
            obss, rewss, dones, infos = env.step(actions)
            if not env.agents:
                env.reset()
    
    def test_api(self):
        from pettingzoo.test import parallel_api_test
        env = parallel_env(max_cycles=MAX_CYCLES, ssd_args=args)
        parallel_api_test(env, MAX_CYCLES)


if __name__ == "__main__":
    unittest.main()
