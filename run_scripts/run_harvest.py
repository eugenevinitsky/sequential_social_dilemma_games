import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env

from social_dilemmas.envs.harvest import HarvestEnv
from models.conv_to_fc_net import ConvToFCNet

NUM_CPUS = 2
NUM_AGENTS = 5


def setup():
    def env_creator(_):
        return HarvestEnv(num_agents=NUM_AGENTS)

    env_name = "harvest_env"
    register_env(env_name, env_creator)
    single_env = HarvestEnv()
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(NUM_AGENTS):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    algorithm = 'A3C'
    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()
    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm
    # hyperparams
    config.update({
                "train_batch_size": 30000,
                "horizon": 1000,
                "lr_schedule":
                [[0, 0.00136],
                    [20000000, 0.000028]],
                "num_workers": NUM_CPUS - 1,
                "entropy_coeff": -.000687,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                "model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                          "lstm_cell_size": 128}

    })
    return algorithm, env_name, config


if __name__ == "__main__":
    ray.init(num_cpus=NUM_CPUS, redirect_output=True)
    alg_run, env_name, config = setup()

    run_experiments({
        "harvest_test": {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": 200
            },
            'checkpoint_freq': 20,
            "config": config,
        }
    })
