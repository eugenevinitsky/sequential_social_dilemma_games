import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.a3c.a3c_policy_graph_actions import A3CPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net_actions import ConvToFCNetActions

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', 'causal_actions',
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'env', 'harvest',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_integer(
    'num_agents', 5,
    'Number of agent policies')
tf.app.flags.DEFINE_integer(
    'num_cpus', 38,
    'Number of available CPUs')
tf.app.flags.DEFINE_integer(
    'num_gpus', 0,
    'Number of available GPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpu_for_driver', False,
    'Set to true to run driver on GPU rather than CPU.')
tf.app.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')
tf.app.flags.DEFINE_boolean(
    'tune', False,
    'Set to true to do hyperparameter tuning.')
tf.app.flags.DEFINE_boolean(
    'debug', False,
    'Set to true to run in a debugging / testing mode with less memory.')
tf.app.flags.DEFINE_boolean(
    'resume', False,
    'Set to true to resume a previously stopped experiment.')

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176}


def setup(env, hparams, num_cpus, num_gpus, num_agents, use_gpus_for_workers=False,
          use_gpu_for_driver=False, num_workers_per_device=1, tune_hparams=False):
    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)

        single_env = HarvestEnv()
        default_hparams = harvest_default_params
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)

        single_env = CleanupEnv()
        default_hparams = cleanup_default_params

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy(agent_id):
        return (A3CPolicyGraph, obs_space, act_space,
                {'num_other_agents': num_agents - 1, 'agent_id': agent_id})

    # Setup A3C with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        agent_id = 'agent-' + str(i)
        policy_graphs[agent_id] = gen_policy(agent_id)

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_to_fc_net_actions"
    ModelCatalog.register_custom_model(model_name, ConvToFCNetActions)

    algorithm = 'A3C'
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
    if tune_hparams:
        config.update({
            "train_batch_size": 128,
            "horizon": 1000,
            "lr_schedule": [[0, tune.grid_search([5e-4, 5e-3])],
                            [20000000, tune.grid_search([5e-4, 5e-5, 5e-6])]],
            "num_workers": num_workers,
            "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": cpus_for_driver,
            "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
            "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
            "entropy_coeff": tune.grid_search([5e-3, 5e-4, 5e-5]),
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
            },
            "model": {"custom_model": "conv_to_fc_net_actions", "use_lstm": True,
                      "lstm_cell_size": 128, "lstm_use_prev_action_reward": True,
                      "custom_options": {"num_other_agents": num_agents - 1}}

        })
    else:
        config.update({
            "train_batch_size": 128,
            "horizon": 1000,
            "lr_schedule": [[0, default_hparams['lr_init']],
                            [20000000, default_hparams['lr_final']]],
            "num_workers": num_workers,
            "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": cpus_for_driver,
            "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
            "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
            "entropy_coeff": default_hparams['entropy_coeff'],
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
            },
            "model": {"custom_model": "conv_to_fc_net_actions", "use_lstm": True,
                      "lstm_cell_size": 128, "lstm_use_prev_action_reward": True,
                      "custom_options": {"num_other_agents": num_agents - 1}}

        })
    return algorithm, env_name, config


def main(unused_argv):
    if FLAGS.debug:
        ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=int(1e9),
                 redis_max_memory=int(1e9))
    else:
        ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=int(1e10),
                 redis_max_memory=int(2e10))
    if FLAGS.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device, FLAGS.tune)

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_A3C_actions'
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": 2000
            },
            'checkpoint_freq': 100,
            "config": config,
            'upload_dir': 's3://njaques.experiments/first_reproduction/causal_actions_harvest'
        }
    }, resume=FLAGS.resume)


if __name__ == '__main__':
    tf.app.run(main)
