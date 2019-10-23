from datetime import datetime
import pytz

import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from algorithms.ppo_causal import CausalMOATrainer
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.moa_model import MOA_LSTM

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', None,
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'env', 'harvest',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'algorithm', 'PPO',
    'Name of the rllib algorithm to use.')
tf.app.flags.DEFINE_integer(
    'num_agents', 2,
    'Number of agent policies')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 30000,
    'Size of the total dataset over which one epoch is computed.')
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 50,
    'Number of steps before a checkpoint is saved.')
tf.app.flags.DEFINE_integer(
    'training_iterations', 10000,
    'Total number of steps to train for')
tf.app.flags.DEFINE_integer(
    'num_cpus', 2,
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
tf.app.flags.DEFINE_integer(
    'num_envs_per_worker', 1,
    'Number of envs to place on a single worker'
)
tf.app.flags.DEFINE_boolean(
    'return_agent_actions', 0,
    'If true we return the previous actions of all the agents')
tf.app.flags.DEFINE_boolean(
    'multi_node', 0,
    'If true the experiments are run in multi-cluster mode'
)
tf.app.flags.DEFINE_boolean(
    'use_s3', False,
    'If true upload to s3')
tf.app.flags.DEFINE_boolean(
    'grid_search', False,
    'If true run a grid search over relevant hyperparams')

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687,
    'moa_weight': 10.911}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176,
    'moa_weight': 15.007}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, num_envs_per_worker, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1, return_agent_actions=False):

    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents, return_agent_actions=True)
        single_env = HarvestEnv(num_agents=num_agents, return_agent_actions=True)
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents, return_agent_actions=True)
        single_env = CleanupEnv(num_agents=num_agents, return_agent_actions=True)

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    model_name = "moa_lstm"
    ModelCatalog.register_custom_model(model_name, MOA_LSTM)

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": "moa_lstm"}

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

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
                "train_batch_size": train_batch_size,
                "horizon": 1000,
                "gamma": 0.99,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "num_gpus": num_gpus,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": policy_mapping_fn,
                },
                "model": {"custom_model": "moa_lstm", "use_lstm": False,
                          "custom_options": {"return_agent_actions": return_agent_actions, "cell_size": 128,
                                             "num_other_agents": num_agents - 1, "fcnet_hiddens": [32, 32],
                                             "train_moa_only_when_visible": tune.grid_search([True]),
                                             "moa_weight": 10,
                                             },
                          "conv_filters": [[6, [3, 3], 1]]},
                "num_sgd_iter": 10,
                "num_other_agents": num_agents - 1,
                "sgd_minibatch_size": 500,
                "moa_weight": hparams['moa_weight'],
                "train_moa_only_when_visible": tune.grid_search([True]),
                "influence_reward_clip": 10,
                "influence_divergence_measure": 'kl',
                "influence_reward_weight": tune.grid_search([1.0]),
                "influence_curriculum_steps": tune.grid_search([10e6]),
                "influence_scaledown_start": tune.grid_search([100e6]),
                "influence_scaledown_end": tune.grid_search([300e6]),
                "influence_scaledown_final_val": tune.grid_search([.5]),
                "influence_only_when_visible": tune.grid_search([True]),
                "vf_loss_coeff": 1e-4

    })

    if FLAGS.grid_search:
        config.update({'moa_weight': tune.grid_search([10, 100]),
                       'lr_schedule': [[0, tune.grid_search([1e-2, 1e-3, 1e-4])],
                                        [20000000, hparams['lr_final']]],
                       'vf_loss_coeff': tune.grid_search([1e-3, 1e-4, 1e-5]),
                       'entropy_coeff': tune.grid_search([0, 1e-3, 1e-4]),
                       'influence_reward_weight': tune.grid_search([1.0, 10.0])})
    return algorithm, env_name, config


def main(unused_argv):
    if FLAGS.multi_node:
        ray.init(redis_address='localhost:6379')
    else:
        ray.init()
    if FLAGS.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                      FLAGS.train_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.num_envs_per_worker,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device,
                                      FLAGS.return_agent_actions)

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name
    print('Commencing experiment', exp_name)

    config['env'] = env_name

    eastern = pytz.timezone('US/Eastern')
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://ssd-reproduce/" \
                + date + '/' + exp_name

    exp_dict = {
            'name': exp_name,
            'run_or_experiment': CausalMOATrainer,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }
    if FLAGS.use_s3:
        exp_dict['upload_dir'] = s3_string

    tune.run(**exp_dict, queue_trials=False)

if __name__ == '__main__':
    tf.app.run(main)
