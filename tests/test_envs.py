'''Unit tests for all of the envs'''

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.harvest import COLOURS

MINI_HARVEST_MAP = [
'@@@@@@'
'@ P  @'
'@  A @'
'@ AAA@'
'@  AP@'
'@@@@@@'
]


import unittest

class TestHarvestEnv(unittest.TestCase):
    def test_setup(self):
        self.env = HarvestEnv(ascii_map=MINI_HARVEST_MAP, num_agents=1)

    def test_step(self):
        pass

    def test_reset(self):
        self.env.reset()

if __name__ == '__main__':
    unittest.main()
