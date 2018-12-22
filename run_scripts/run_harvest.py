import ray
from ray import tune
from ray.rllib.agents.agent import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune import run_experiments
from ray.tune.registry import register_env

from social_dilemmas.envs.harvest import HarvestEnv

NUM_CPUS = 1

if __name__ == "__main__":
    ray.init(num_cpus=NUM_CPUS, redirect_output=False)

    # Simple environment with `num_agents` independent cartpole entities
    def env_creator(_):
        return HarvestEnv(num_agents=3)

    env_name = "harvest_env"
    register_env(env_name, env_creator)
    single_env = HarvestEnv()
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'shared': gen_policy()}

    # TODO(ev) currently all agents share the same policy
    def policy_mapping_fn(_):
        return 'shared'

    alg_run = 'PPO'
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = alg_run
    # hyperparams
    config.update({
                "train_batch_size": 30000,
                "horizon": 1000,
                "num_workers": NUM_CPUS - 1,
                "num_sgd_iter": 10,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                "model": {"dim": 3, "conv_filters":
                          # num_outs, kernel, stride
                          # TODO(ev) pick better numbers
                          [[4, [2, 2], 1], [8, [7, 7], 1]]}

    })

    run_experiments({
        "harvest_test": {
            "run": "PPO",
            "env": "harvest_env",
            "stop": {
                "training_iteration": 200
            },
            'checkpoint_freq': 20,
            "config": config,
        }
    })
