"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import argparse
import collections
from collections import defaultdict
import json
import numpy as np
import os
import shutil
import sys

import ray
from ray.cloudpickle import cloudpickle
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# from ray.rllib.evaluation.sampler import clip_action

from models.conv_to_fc_net import ConvToFCNet
from models.conv_to_fc_net_actions import ConvToFCNetActions
from models.conv_net import ConvNet
from models.conv_to_fc_net_actions_no_lstm import (
    ConvToFCNetActions as ConvToFCNetActionsNoLSTM,
)
from models.conv_to_fc_net_no_lstm import ConvToFCNet as ConvToFCNetNoLSTM
import utility_funcs


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + "/params.json"  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + "/params.pkl"  # params.json is the config file
    with open(pklfile, "rb") as file:
        pkldata = cloudpickle.load(file)
    return pkldata


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def visualizer_rllib(args):
    result_dir = args.result_dir if args.result_dir[-1] != "/" else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get("multiagent", {}).get("policy_graphs", {}):
        multiagent = True
        config["multiagent"] = pkl["multiagent"]
    else:
        multiagent = False

    # Create and register a gym+rllib env
    env_creator = pkl["env_config"]["func_create"]
    env_name = config["env_config"]["env_name"]
    register_env(env_name, env_creator.func)

    ModelCatalog.register_custom_model("conv_to_fc_net", ConvToFCNet)
    # ModelCatalog.register_custom_model("conv_to_fc_net_no_lstm", ConvToFCNetNoLSTM)
    ModelCatalog.register_custom_model("conv_to_fc_net_actions", ConvToFCNetActions)
    # ModelCatalog.register_custom_model("conv_to_fc_net_actions_no_lstm", ConvToFCNetActionsNoLSTM)
    ModelCatalog.register_custom_model("conv_net", ConvNet)

    # Determine agent and checkpoint
    config_run = config["env_config"]["run"] if "run" in config["env_config"] else None
    if args.run and config_run:
        if args.run != config_run:
            print(
                "visualizer_rllib.py: error: run argument "
                + "'{}' passed in ".format(args.run)
                + "differs from the one stored in params.json "
                + "'{}'".format(config_run)
            )
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print(
            "visualizer_rllib.py: error: could not find parameter "
            "'run' in params.json, "
            "add argument --run to provide the algorithm or model used "
            "to train the results\n e.g. "
            "python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO"
        )
        sys.exit(1)

    # Run on only one cpu for rendering purposes if possible; A3C requires two
    if config_run == "A3C":
        config["num_workers"] = 1
        config["sample_async"] = False
    else:
        config["num_workers"] = 0

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + "/checkpoint_" + args.checkpoint_num
    checkpoint = checkpoint + "/checkpoint-" + args.checkpoint_num
    print("Loading checkpoint", checkpoint)
    agent.restore(checkpoint)

    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.local_evaluator.multiagent:
            policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {p: m.action_space.sample() for p, m in policy_map.items()}
    else:
        env = env_creator()
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if args.save_video:
        shape = env.base_map.shape
        full_obs = [
            np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            for i in range(config["horizon"])
        ]

    steps = 0
    while steps < (config["horizon"] or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.0)
        done = False
        last_actions = [0] * len(obs.keys())  # Number of agents
        reward_total = 0.0
        while not done and steps < (config["horizon"] or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            info={"all_agents_actions": last_actions},
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            info={"all_agents_actions": last_actions},
                        )
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            agent_ids = sorted(obs.keys())
            last_actions = [action_dict[a] for a in agent_ids]
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            if args.save_video:
                rgb_arr = env.map_to_colors()
                full_obs[steps] = rgb_arr.astype(np.uint8)
            steps += 1
            obs = next_obs
        print("Episode reward", reward_total)

    if args.save_video:
        path = (
            os.path.abspath(args.video_path)
            if hasattr(args, "video_path") and args.video_path is not None
            else os.path.abspath(os.path.dirname(__file__)) + "/videos"
        )
        video_name = (
            args.video_filename
            if hasattr(args, "video_filename") and args.video_filename is not None
            else "trajectory"
        )

        if not os.path.exists(path):
            os.makedirs(path)
        images_path = path + "/images/"
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        utility_funcs.make_video_from_rgb_imgs(full_obs, path, video_name=video_name)

        # Clean up images
        shutil.rmtree(images_path)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluates a reinforcement learning agent " "given a checkpoint.",
    )

    # required input parameters
    parser.add_argument("result_dir", type=str, help="Directory containing results")
    parser.add_argument("checkpoint_num", type=str, help="Checkpoint number.")

    # optional input parameters
    parser.add_argument(
        "--run",
        type=str,
        help="The algorithm or model to train. This may refer to "
        "the name of a built-on algorithm (e.g. RLLib's DQN "
        "or PPO), or a user-defined trainable function or "
        "class registered in the tune registry. "
        "Required for results trained with flow-0.2.0 and before.",
    )
    # optional input parameters
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="The number of rollouts to visualize.",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save a movie or not."
    )
    parser.add_argument(
        "--video-path", action="store", help="Path where the video should be stored."
    )
    parser.add_argument(
        "--video-filename",
        action="store",
        help="Name of the video. No file extension needed.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to watch the rollout while it happens.",
    )
    return parser


def visualize(args=None):
    parser = create_parser()
    args = parser.parse_args(args)
    ray.init(num_cpus=6)
    visualizer_rllib(args)


if __name__ == "__main__":
    visualize()
