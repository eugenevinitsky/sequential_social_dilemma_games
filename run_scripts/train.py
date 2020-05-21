import argparse
import copy
import sys
from datetime import datetime

import pytz
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import Experiment
from ray.tune.registry import register_env

from algorithms.a3c_baseline import build_a3c_baseline_trainer
from algorithms.a3c_moa import build_a3c_moa_trainer
from algorithms.impala_baseline import build_impala_baseline_trainer
from algorithms.impala_moa import build_impala_moa_trainer
from algorithms.ppo_baseline import build_ppo_baseline_trainer
from algorithms.ppo_moa import build_ppo_moa_trainer
from algorithms.ppo_scm import build_ppo_scm_trainer
from config.default_args import add_default_args
from models.baseline_model import BaselineModel
from models.moa_model import MOAModel
from models.scm_model import SocialCuriosityModule
from social_dilemmas.envs.env_creator import get_env_creator
from utility_funcs import update_nested_dict

parser = argparse.ArgumentParser()
add_default_args(parser)


def build_experiment_config_dict(args):
    env_creator = get_env_creator(args.env, args.num_agents, args)
    env_name = args.env + "_env"
    register_env(env_name, env_creator)

    single_env = env_creator(args.num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    model_name = args.model + "_lstm"
    if args.model == "scm":
        ModelCatalog.register_custom_model(model_name, SocialCuriosityModule)
    elif args.model == "moa":
        ModelCatalog.register_custom_model(model_name, MOAModel)
    elif args.model == "baseline":
        ModelCatalog.register_custom_model(model_name, BaselineModel)

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return None, obs_space, act_space, {"custom_model": model_name}

    # Create 1 distinct policy per agent
    policy_graphs = {}
    for i in range(args.num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    agent_cls = get_agent_class(args.algorithm)
    config = copy.deepcopy(agent_cls._default_config)

    config["env"] = env_name
    config["eager"] = args.eager_mode

    # information for replay
    config["env_config"]["func_create"] = env_creator
    config["env_config"]["env_name"] = env_name
    if env_name == "switch_env":
        config["env_config"]["num_switches"] = args.num_switches

    if args.small_model:
        conv_filters = [[3, [3, 3], 1]]
        fcnet_hiddens = [8, 8]
        lstm_cell_size = 16
    else:
        conv_filters = [[6, [3, 3], 1]]
        fcnet_hiddens = [32, 32]
        lstm_cell_size = 128

    train_batch_size = (
        args.train_batch_size
        if args.train_batch_size is not None
        else max(1, args.num_workers) * args.num_envs_per_worker * args.rollout_fragment_length
    )

    # hyperparams
    update_nested_dict(
        config,
        {
            "horizon": 1000,
            "gamma": 0.99,
            "lr": args.lr,
            "lr_schedule": list(zip(args.lr_schedule_steps, args.lr_schedule_weights)),
            "rollout_fragment_length": args.rollout_fragment_length,
            "train_batch_size": train_batch_size,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "num_gpus": args.gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": args.cpus_for_driver,
            "num_gpus_per_worker": args.gpus_per_worker,  # Can be a fraction
            "num_cpus_per_worker": args.cpus_per_worker,  # Can be a fraction
            "entropy_coeff": args.entropy_coeff,
            "grad_clip": args.grad_clip,
            "multiagent": {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            "callbacks": single_env.get_environment_callbacks(),
            "model": {
                "custom_model": model_name,
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_options": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": args.num_agents - 1,
                },
            },
        },
    )

    if args.model != "baseline":
        config["model"]["custom_options"].update(
            {
                "moa_loss_weight": args.moa_loss_weight,
                "influence_reward_clip": 10,
                "influence_reward_weight": args.influence_reward_weight,
                "influence_reward_schedule_steps": args.influence_reward_schedule_steps,
                "influence_reward_schedule_weights": args.influence_reward_schedule_weights,
                "return_agent_actions": True,
                "influence_divergence_measure": "kl",
                "train_moa_only_when_visible": True,
                "influence_only_when_visible": True,
            }
        )

    if args.model == "scm":
        config["model"]["custom_options"].update(
            {
                "scm_loss_weight": args.scm_loss_weight,
                "curiosity_reward_clip": 10,
                "curiosity_reward_weight": args.curiosity_reward_weight,
                "curiosity_reward_schedule_steps": args.curiosity_reward_schedule_steps,
                "curiosity_reward_schedule_weights": args.curiosity_reward_schedule_weights,
                "scm_forward_vs_inverse_loss_weight": args.scm_forward_vs_inverse_loss_weight,
            }
        )

    if args.grid_search:
        config["entropy_coeff"] = tune.grid_search(args.entropy_tune)
        config["model"]["custom_options"]["moa_loss_weight"] = tune.grid_search(
            args.moa_loss_weight_tune
        )
        config["model"]["custom_options"]["influence_reward_weight"] = tune.grid_search(
            args.influence_reward_weight_tune
        )

    if args.algorithm == "PPO":
        config.update(
            {
                "num_sgd_iter": 10,
                "sgd_minibatch_size": args.ppo_sgd_minibatch_size
                if args.ppo_sgd_minibatch_size is not None
                else train_batch_size / 2,
                "vf_loss_coeff": 1e-4,
            }
        )
    elif args.algorithm == "A3C" or args.algorithm == "IMPALA":
        config.update({"vf_loss_coeff": 0.1})
    else:
        sys.exit("The only available algorithms are A3C, PPO and IMPALA")

    return config


def get_trainer(args, config):
    if args.model == "baseline":
        if args.algorithm == "A3C":
            trainer = build_a3c_baseline_trainer(config)
        if args.algorithm == "PPO":
            trainer = build_ppo_baseline_trainer(config)
        if args.algorithm == "IMPALA":
            trainer = build_impala_baseline_trainer(config)
    elif args.model == "moa":
        if args.algorithm == "A3C":
            trainer = build_a3c_moa_trainer(config)
        if args.algorithm == "PPO":
            trainer = build_ppo_moa_trainer(config)
        if args.algorithm == "IMPALA":
            trainer = build_impala_moa_trainer(config)
    elif args.model == "scm":
        if args.algorithm == "A3C":
            # trainer = build_a3c_scm_trainer(config)
            raise NotImplementedError
        if args.algorithm == "PPO":
            trainer = build_ppo_scm_trainer(config)
        if args.algorithm == "IMPALA":
            # trainer = build_impala_scm_trainer(config)
            raise NotImplementedError
    if trainer is None:
        raise NotImplementedError("The provided combination of model and algorithm was not found.")
    return trainer


def initialize_ray(args):
    if sys.gettrace() is not None:
        print(
            "Debug mode detected through sys.gettrace(), turning on ray local mode. Saving"
            " experiment under ray_results/debug_experiment"
        )
        args.local_mode = True
    if args.multi_node and args.local_mode:
        sys.exit("You cannot have both local mode and multi node on at the same time")
    ray.init(
        address=args.address,
        local_mode=args.local_mode,
        memory=args.memory,
        object_store_memory=args.object_store_memory,
        redis_max_memory=args.redis_max_memory,
        include_webui=False,
    )


def get_experiment_name(args):
    if sys.gettrace() is not None:
        exp_name = "debug_experiment"
    elif args.exp_name is None:
        exp_name = args.env + "_" + args.model + "_" + args.algorithm
    else:
        exp_name = args.exp_name
    return exp_name


def build_experiment_dict(args, experiment_name, trainer, config):
    experiment_dict = {
        "name": experiment_name,
        "run": trainer,
        "stop": {},
        "checkpoint_freq": args.checkpoint_frequency,
        "config": config,
        "num_samples": args.num_samples,
    }
    if args.stop_at_episode_reward_min is not None:
        experiment_dict["stop"]["episode_reward_min"] = args.stop_at_episode_reward_min
    if args.stop_at_timesteps_total is not None:
        experiment_dict["stop"]["timesteps_total"] = args.stop_at_timesteps_total

    if args.use_s3:
        date = datetime.now(tz=pytz.utc)
        date = date.astimezone(pytz.timezone("US/Pacific")).strftime("%m-%d-%Y")
        s3_string = "s3://ssd-reproduce/" + date + "/" + experiment_name
        experiment_dict["upload_dir"] = s3_string

    return experiment_dict


def create_experiment(args):
    experiment_name = get_experiment_name(args)
    config = build_experiment_config_dict(args)
    trainer = get_trainer(args=args, config=config)
    experiment_dict = build_experiment_dict(args, experiment_name, trainer, config)
    return Experiment(**experiment_dict)


def run(args, experiments):
    initialize_ray(args)
    tune.run_experiments(experiments, queue_trials=args.use_s3)


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    experiment = create_experiment(parsed_args)
    run(parsed_args, experiment)
