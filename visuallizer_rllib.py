"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import argparse
import json
import numpy as np
import os
import shutil
import sys

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.cloudpickle import cloudpickle

from models.conv_to_fc_net import ConvToFCNet
import utility_funcs


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + '/params.json'  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + '/params.pkl'  # params.json is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)
    return pkldata


def visualizer_rllib(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Create and register a gym+rllib env
    env_creator = pkl['env_config']['func_create']
    env_name = config['env_config']['env_name']
    register_env(env_name, env_creator.func)

    ModelCatalog.register_custom_model("conv_to_fc_net", ConvToFCNet)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if (args.run and config_run):
        if (args.run != config_run):
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if (args.run):
        agent_cls = get_agent_class(args.run)
    elif (config_run):
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # Run on only one cpu for rendering purposes if possible; A3C requires two
    if config_run == 'A3C':
        config['num_workers'] = 1
    else:
        config['num_workers'] = 0

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env

    if args.save_video:
        shape = env.base_map.shape
        full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                    for i in range(config["horizon"])]

    rets = {}
    # map the agent id to its policy
    policy_map_fn = config['multiagent']['policy_mapping_fn'].func
    for key in config['multiagent']['policy_graphs'].keys():
        rets[key] = []

    if config['model']['use_lstm']:
        use_lstm = True
        state_init = [
            np.zeros(config['model']['lstm_cell_size'], np.float32),
            np.zeros(config['model']['lstm_cell_size'], np.float32)
        ]
    else:
        use_lstm = False

    for i in range(args.num_rollouts):
        state = env.reset()
        done = False
        if multiagent:
            ret = {key: [0] for key in rets.keys()}
        else:
            ret = 0
        for j in range(config["horizon"]):
            action = {}
            for agent_id in state.keys():
                if use_lstm:
                    action[agent_id], state_init, logits = agent.compute_action(
                        state[agent_id], state=state_init, policy_id=policy_map_fn(agent_id))
                else:
                    action[agent_id] = agent.compute_action(
                        state[agent_id], policy_id=policy_map_fn(agent_id))
            observations, reward, done, _ = env.step(action)
            if args.render:
                env.render_map()
            if args.save_video:
                rgb_arr = env.map_to_colors()
                full_obs[j] = rgb_arr.astype(np.uint8)

            for actor, rew in reward.items():
                ret[policy_map_fn(actor)][0] += rew

            if multiagent and done['__all__']:
                break
            if not multiagent and done:
                break

        for key in rets.keys():
            rets[key].append(ret[key])

        for agent_id, rew in rets.items():
            print('Round {}, Return: {} for agent {}'.format(
                i, ret, agent_id))
    for agent_id, rew in rets.items():
        print('Average, std return: {}, {} for agent {}'.format(
            np.mean(rew), np.std(rew), agent_id))

    if args.save_video:
        path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
        if not os.path.exists(path):
            os.makedirs(path)
        images_path = path + '/images/'
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        utility_funcs.make_video_from_rgb_imgs(full_obs, path)

        # Clean up images
        shutil.rmtree(images_path)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a reinforcement learning agent '
                    'given a checkpoint.')

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='whether to save a movie or not')
    parser.add_argument(
        '--render',
        action='store_true',
        help='whether to watch the rollout while it happens')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=2, redirect_output=False)
    visualizer_rllib(args)
