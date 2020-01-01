import ray
from ray import tune
from ray.tune.registry import register_env

from social_dilemmas.envs.env_creator import get_env_creator


class Args(object):
    pass


args = Args()
args.num_switches = 6
args.env = "switch"

env_creator = get_env_creator(args.env, 1, args)
env_name = args.env + "_env"
register_env(env_name, env_creator)

ray.init(memory=100000000000)
tune.run(
    "PPO",
    stop={"episode_reward_mean": 1, "timesteps_total": 500000},
    config={
        "env": env_name,
        "horizon": 1000,
        "sample_batch_size": 1000,
        "num_cpus_for_driver": 0,
        "num_gpus": 1,
        "num_gpus_per_worker": 3.0 / 12.0,
        "num_workers": 12,
        "num_envs_per_worker": tune.grid_search([1, 2, 4]),
        "lr": tune.grid_search([0.001, 0.005, 0.01, 0.05]),
        "entropy_coeff": 0,
        "model": {"fcnet_hiddens": [32, 32], "use_lstm": False, "vf_share_layers": True},
        "sgd_minibatch_size": tune.grid_search([250, 500]),
    },
)
