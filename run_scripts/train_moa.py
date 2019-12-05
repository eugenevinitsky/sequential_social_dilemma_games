import argparse
from datetime import datetime
import sys

import pytz
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from algorithms.a3c_aux import get_a3c_trainer
from algorithms.ppo_causal import CausalPPOMOATrainer
from algorithms.impala_causal import CausalImpalaTrainer
from models.moa_model import MOA_LSTM
from config.default_args import add_default_args
from social_dilemmas.envs.env_creator import get_env_creator

parser = argparse.ArgumentParser()
add_default_args(parser)


def setup(args):
    env_creator = get_env_creator(args.env, args.num_agents, args)
    env_name = args.env + "_env"
    register_env(env_name, env_creator)

    single_env = env_creator(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    model_name = "moa_lstm"
    ModelCatalog.register_custom_model(model_name, MOA_LSTM)

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": "moa_lstm"}

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(args.num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    agent_cls = get_agent_class(args.algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = env_creator
    config['env_config']['env_name'] = env_name
    # config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(args.use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if args.use_gpus_for_workers:
        spare_gpus = (args.num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * args.num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (args.num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * args.num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
        "horizon": 1000,
        "gamma": 0.99,
        "lr_schedule": list(zip(args.lr_curriculum_steps, args.lr_curriculum_weights)),
        "sample_batch_size": args.sample_batch_size,
        "train_batch_size": args.train_batch_size,
        "num_workers": num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
        "entropy_coeff": args.entropy_coeff,
        "grad_clip": args.grad_clip,
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {"custom_model": "moa_lstm",
                  "use_lstm": False,
                  "custom_options": {
                      "aux_loss_weight": args.aux_loss_weight,
                      "aux_reward_clip": 10,
                      "aux_reward_weight": args.aux_reward_weight,
                      "aux_reward_curriculum_steps": args.aux_reward_curriculum_steps,
                      "aux_reward_curriculum_weights": args.aux_reward_curriculum_weights,
                      "cell_size": 128,
                      "num_other_agents": args.num_agents - 1,
                      "return_agent_actions": True,
                      "fcnet_hiddens": [32, 32],
                      "influence_divergence_measure": 'kl',
                      "train_moa_only_when_visible": tune.grid_search([True]),
                      "influence_only_when_visible": tune.grid_search([True]),
                  },
                  "conv_filters": [[6, [3, 3], 1]]
                  },
    })
    if args.algorithm == "PPO":
        config.update({"num_sgd_iter": 10,
                       "sgd_minibatch_size": 500,
                       "vf_loss_coeff": 1e-4
                       })
    elif args.algorithm == "A3C" or args.algorithm == "IMPALA":
        config.update({"vf_loss_coeff": 0.1})
    else:
        sys.exit("The only available algorithms are A3C, PPO and IMPALA")

    if args.grid_search:
        config["entropy_coeff"] = tune.grid_search(args.entropy_tune)
        config["model"]["custom_options"]["aux_loss_weight"] = tune.grid_search(args.aux_loss_weight_tune)
        config["model"]["custom_options"]["aux_reward_weight"] = tune.grid_search(args.aux_reward_weight_tune)

    return env_name, config


if __name__ == '__main__':
    args = parser.parse_args()
    if args.multi_node and args.local_mode:
        sys.exit("You cannot have both local mode and multi node on at the same time")
    ray.init(address=args.address,
             local_mode=args.local_mode,
             memory=args.memory,
             object_store_memory=args.object_store_memory,
             redis_max_memory=args.redis_max_memory)
    env_name, config = setup(args)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    config['env'] = env_name
    config['eager'] = args.eager_mode

    if args.algorithm == "A3C":
        trainer = get_a3c_trainer("causal", config)
    if args.algorithm == "PPO":
        trainer = CausalPPOMOATrainer
    if args.algorithm == "IMPALA":
        trainer = CausalImpalaTrainer

    exp_dict = {'name': exp_name,
                'run_or_experiment': trainer,
                "stop": {},
                'checkpoint_freq': args.checkpoint_frequency,
                'config': config,
                'num_samples': args.num_samples
                }
    if args.stop_at_episode_reward_min is not None:
        exp_dict['stop']['episode_reward_min'] = args.stop_at_episode_reward_min
    if args.stop_at_timesteps_total is not None:
        exp_dict['stop']['timesteps_total'] = args.stop_at_timesteps_total

    if args.use_s3:
        eastern = pytz.timezone('US/Eastern')
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
        s3_string = "s3://ssd-reproduce/" + date + '/' + exp_name
        exp_dict['upload_dir'] = s3_string

    tune.run(**exp_dict, queue_trials=True)
