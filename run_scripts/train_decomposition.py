import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.a3c.a3c_tf_policy_graph import A3CPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf
import logging

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net import ConvToFCNet
from models.decomposition  import A3CDecompositionPolicyGraph


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', None,
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'env', 'cleanup',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'algorithm', 'A3C',
    'Name of the rllib algorithm to use.')
tf.app.flags.DEFINE_integer(
    'num_agents', 5,
    'Number of agent policies')

# This should be the common denominator(공약수) of the length of the episode
tf.app.flags.DEFINE_integer(
    'sample_batch_size', 100,
    'Size of the total dataset over which one epoch is computed.')
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 20,
    'Number of steps before a checkpoint is saved.')
tf.app.flags.DEFINE_integer(
    'training_iterations', 10000,
    'Total number of steps to train for')
tf.app.flags.DEFINE_integer(
    'num_cpus', 2,
    'Number of available CPUs')
tf.app.flags.DEFINE_integer(
    'num_gpus', 1,
    'Number of available GPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpu_for_driver', True,
    'Set to true to run driver on GPU rather than CPU.')
tf.app.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')



tf.app.flags.DEFINE_float(
    "exponential_decay", 0.9,
    "Exponentially decay the reward")

tf.app.flags.DEFINE_integer(
    "num_altruistic_agents", 1,
    "Number of agents using publiceness")


harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': 0.000687,
    'team_coeff':0.5,
    'const_coeff': 0.01
    }

cleanup_default_params = {
    'lr_init': 0.001,
    'lr_final': 0.001,
    'entropy_coeff': 0.00176,
    # Team Coeff
    'team_coeff' : 0.5,
    # Only needs to be controlled
    'equal_const_coeff': 0.0,
    'explore_const_coeff': 0.0
    }


def setup(env, hparams, algorithm, sample_batch_size, num_cpus, num_gpus,
          num_agents, num_altruistic_agents, exponential_decay, 
          use_gpus_for_workers=False, use_gpu_for_driver=False, num_workers_per_device=1):

    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
        single_env = HarvestEnv()
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)
        single_env = CleanupEnv()

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (A3CDecompositionPolicyGraph, obs_space, act_space, {})

    # Setup A3C with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name, ConvToFCNet)

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
                "sample_batch_size": sample_batch_size,
                "horizon": 1000,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                "model": {"custom_model": "conv_to_fc_net", "use_lstm": True,
                          "lstm_cell_size": 128},
    })
    config['team_coeff']            = hparams['team_coeff']
    config['equal_const_coeff']     = hparams['equal_const_coeff']
    config['explore_const_coeff']   = hparams['explore_const_coeff']
    config['exponential_decay']     = exponential_decay
    config['num_altruistic_agents'] = num_altruistic_agents
    
    return algorithm, env_name, config


def main(unused_argv):
    ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=10737418240, log_to_driver = True) #10gb
    if FLAGS.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                      FLAGS.sample_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, 
                                      FLAGS.num_agents,
                                      FLAGS.num_altruistic_agents,
                                      FLAGS.exponential_decay,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device)
    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }
    })


if __name__ == '__main__':
    tf.app.run(main)
