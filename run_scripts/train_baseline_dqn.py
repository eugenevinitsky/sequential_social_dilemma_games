import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from config import config_parser
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net_no_lstm import ConvToFCNet

config_parser.set_tf_flags('baseline_dqn')
FLAGS = tf.app.flags.FLAGS
hparams = config_parser.get_env_params()


def setup(env, num_cpus, num_gpus, num_agents, use_gpus_for_workers=False,
          use_gpu_for_driver=False, num_workers_per_device=1):
    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)

    env_name = env + "_env"
    register_env(env_name, env_creator)

    single_env = env_creator(1)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (DQNPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net_no_lstm"
    ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    algorithm = 'DQN'
    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
        "train_batch_size": 128,
        "horizon": 1000,
        "num_workers": num_workers,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
        "multiagent": {
            "policy_graphs": policy_graphs,
            "policy_mapping_fn": tune.function(policy_mapping_fn),
        },
        "model": {"custom_model": "conv_to_fc_net_no_lstm",
                  "use_lstm": False, "lstm_cell_size": 128}

    })
    return algorithm, env_name, config


def main(unused_argv):
    ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=int(2e10),
             redis_max_memory=int(1e10))
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device)

    print('Commencing experiment', FLAGS.exp_name)

    run_experiments({
        FLAGS.exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": 300000
            },
            'checkpoint_freq': 1000,
            "config": config,
        }
    }, resume=FLAGS.resume)


if __name__ == '__main__':
    tf.app.run(main)
