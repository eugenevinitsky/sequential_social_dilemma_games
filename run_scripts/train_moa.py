import argparse
from datetime import datetime
import sys

import pytz
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from algorithms.a3c_causal import CausalA3CMOATrainer
from algorithms.ppo_causal import CausalPPOMOATrainer
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.moa_model import MOA_LSTM

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='causal_env', help='Name experiment will be stored under')
parser.add_argument('--env', type=str, default='cleanup', help='Name of the environment to rollout. Can be '
                                                               'cleanup or harvest.')
parser.add_argument('--algorithm', type=str, default='PPO', help='Name of the rllib algorithm to use.')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agent policies')
parser.add_argument('--train_batch_size', type=int, default=26000,
                    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint_frequency', type=int, default=50,
                    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training_iterations', type=int, default=50, help='Total number of steps to train for')
parser.add_argument('--num_cpus', type=int, default=2, help='Number of available CPUs')
parser.add_argument('--num_gpus', type=int, default=0, help='Number of available GPUs')
parser.add_argument('--use_gpus_for_workers', action='store_true', default=False,
                    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument('--use_gpu_for_driver', action='store_true', default=False,
                    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument('--num_workers_per_device', type=float, default=1,
                    help='Number of workers to place on a single device (CPU or GPU)')
parser.add_argument('--num_envs_per_worker', type=float, default=1,
                    help='Number of envs to place on a single worker')
parser.add_argument('--multi_node', action='store_true', default=False,
                    help='If true the experiments are run in multi-cluster mode')
parser.add_argument('--local_mode', action='store_true', default=False,
                    help='Force all the computation onto the driver. Useful for debugging.')
parser.add_argument('--use_s3', action='store_true', default=False,
                    help='If true upload to s3')
parser.add_argument('--grid_search', action='store_true', default=False,
                    help='If true run a grid search over relevant hyperparams')

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
          num_workers_per_device=1):

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
    config['env_config']['func_create'] = env_creator
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
                          "custom_options": {"return_agent_actions": True, "cell_size": 128,
                                             "num_other_agents": num_agents - 1, "fcnet_hiddens": [32, 32],
                                             "train_moa_only_when_visible": tune.grid_search([True]),
                                             "moa_weight": 10,
                                             },
                          "conv_filters": [[6, [3, 3], 1]]},
                "num_other_agents": num_agents - 1,
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
    if args.algorithm == "PPO":
        config.update({"num_sgd_iter": 10,
                       "train_batch_size": train_batch_size,
                       "sgd_minibatch_size": 500,
                       })
    elif args.algorithm == "A3C":
        config.update({"sample_batch_size": 50})
    else:
        sys.exit("The only available algorithms are A3C and PPO")

    if args.grid_search:
        config.update({'moa_weight': tune.grid_search([10, 100]),
                       'lr_schedule': [[0, tune.grid_search([1e-2, 1e-3, 1e-4])],
                                        [20000000, hparams['lr_final']]],
                       'vf_loss_coeff': tune.grid_search([1e-3, 1e-4, 1e-5]),
                       'entropy_coeff': tune.grid_search([0, 1e-3, 1e-4]),
                       'influence_reward_weight': tune.grid_search([1.0, 10.0])})
        if args.algorithm == "A3C":
            config.update({"sample_batch_size": tune.grid_search([50, 500])})
    return algorithm, env_name, config


if __name__=='__main__':
    args = parser.parse_args()
    if args.multi_node and args.local_mode:
        sys.exit("You cannot have both local mode and multi node on at the same time")
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()
    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.num_envs_per_worker,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    config['env'] = env_name

    eastern = pytz.timezone('US/Eastern')
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://ssd-reproduce/" \
                + date + '/' + exp_name

    if alg_run == "A3C":
        trainer = CausalA3CMOATrainer
    if alg_run == "PPO":
        trainer = CausalPPOMOATrainer

    exp_dict = {
            'name': exp_name,
            'run_or_experiment': trainer,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
        }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    tune.run(**exp_dict, queue_trials=False)
