import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune import run_experiments
from ray.tune.registry import register_env

from social_dilemmas.envs.harvest import HarvestEnv


if __name__ == "__main__":
    ray.init(num_cpus=1)

    # Simple environment with `num_agents` independent cartpole entities
    register_env("harvest_env", lambda _: HarvestEnv())
    single_env = HarvestEnv()
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'shared': (PPOPolicyGraph, obs_space, act_space, {})}

    def policy_mapping_fn(_):
        return 'shared'


    run_experiments({
        "test": {
            "run": "PPO",
            "env": "multi_cartpole",
            "stop": {
                "training_iteration": 100
            },
            "config": {
                "num_workers": 0,
                "log_level": "DEBUG",
                "num_sgd_iter": 10,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
            },
        }
    })