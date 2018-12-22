import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune import run_experiments
from ray.tune.registry import register_env

from social_dilemmas.envs.harvest import HarvestEnv

NUM_CPUS = 1

if __name__ == "__main__":
    ray.init(num_cpus=NUM_CPUS, redirect_output=True)

    # Simple environment with `num_agents` independent cartpole entities
    def env_creator(_):
        return HarvestEnv(num_agents=3)

    register_env("harvest_env", env_creator)
    single_env = HarvestEnv()
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy(i):
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'shared': gen_policy()}

    # TODO(ev) currently all agents share the same policy
    def policy_mapping_fn(_):
        return 'shared'

    run_experiments({
        "test": {
            "run": "PPO",
            "env": "harvest_env",
            "stop": {
                "training_iteration": 100
            },
            "config": {
                "train_batch_size": 10000,
                "horizon": 1000,
                "num_workers": NUM_CPUS - 1,
                "log_level": "DEBUG",
                "num_sgd_iter": 10,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                # FIXME(ev) magic number
                "model": {"dim": 3, "conv_filters":
                          # num_outs, kernel, stride
                          # TODO(ev) pick better numbers
                          [[4, [2, 2], 1], [8, [7, 7], 1]]}
            },
        }
    })
