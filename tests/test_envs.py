'''Unit tests for all of the envs'''

from social_dilemmas.envs.harvest import HarvestEnv

MINI_HARVEST_MAP = [
'@@@@@@',
'@ P  @',
'@  A @',
'@ AAA@',
'@  AP@',
'@@@@@@',
]


import unittest

class TestHarvestEnv(unittest.TestCase):
    def setUp(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)

    def test_step(self):
        pass

    def test_reset(self):
        self.env.reset()

    def test_view(self):
        """Confirm that an agent placed at the right point returns the right view"""
        pass

    def test_apple_spawn(self):
        pass

    def test_firing(self):
        pass

    def test_move_agent(self):
        pass

    def test_agent_rewards(self):
        pass

if __name__ == '__main__':
    unittest.main()
