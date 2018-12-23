import ray
from ray.tune import run_experiments
import unittest

from run_scripts.run_harvest import setup


class TestRunScripts(unittest.TestCase):
    def test_harvest(self):

        algorithm, env_name, config = setup()
        self.run_exp(algorithm, env_name, config)

    def run_exp(self, algorithm, env_name, config):
        try:
            ray.init(num_cpus=2)
        except Exception:
            pass
        config['train_batch_size'] = 50
        config['horizon'] = 20
        config['sample_batch_size'] = 1
        config['num_workers'] = 1

        run_experiments({
            'test': {
                'run': algorithm,
                'env': env_name,
                'config': {
                    **config
                },
                'checkpoint_freq': 1,
                'stop': {
                    'training_iteration': 1,
                },
            }
        })


if __name__ == '__main__':
    unittest.main()
