import ray
from ray import tune
from ray.tune.registry import register_env

from social_dilemmas.envs.env_creator import get_env_creator


class Args(object):
    pass


args = Args()
args.num_switches = 1
args.env = "harvest"

env_creator = get_env_creator(args.env, 1, args)
env_name = args.env + "_env"
register_env(env_name, env_creator)

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": env_name,
        "horizon": 100,
        "sample_batch_size": 100,
        "num_gpus": 1,
        "num_workers": 7,
        "num_envs_per_worker": 16,
        "lr": 0.01,
        "eager": False,
        "memory": 800000000,
    },
)
