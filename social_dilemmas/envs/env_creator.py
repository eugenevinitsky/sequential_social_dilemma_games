from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.switch import SwitchEnv


def get_env_creator(env, num_agents):
    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
    elif env == 'cleanup':
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)
    elif env == 'switch':
        def env_creator(_):
            return SwitchEnv(num_agents=num_agents)

    return env_creator
