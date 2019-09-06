"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil
import tensorflow as tf

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'vid_path', os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
    'Path to directory where videos are saved.')
tf.app.flags.DEFINE_string(
    'env', 'switch',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'render_type', 'pretty',
    'Can be pretty or fast. Implications obvious.')
tf.app.flags.DEFINE_integer(
    'horizon', 600,
    'Number of rendered frames.')
tf.app.flags.DEFINE_integer(
    'fps', 10,
    'Number of frames per second.')


class Controller(object):

    def __init__(self, env_name='cleanup'):
        self.env_name = env_name
        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(num_agents=5, render=True)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=5, render=True)
        elif env_name == 'switch':
            print('Initializing Switch environment')
            self.env = SwitchEnv(num_agents=1, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.env.reset()

        # TODO: initialize agents here

    def rollout(self, horizon=50, save_path=None):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """
        rewards = []
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        for i in range(horizon):
            agents = list(self.env.agents.values())
            action_dim = agents[0].action_space.n
            agent_action_dict = dict()
            for agent in agents:
                rand_action = np.random.randint(action_dim)
                agent_action_dict[agent.agent_id] = rand_action
            obs, rew, dones, info, = self.env.step(agent_action_dict)

            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(i).zfill(6) + '.png')
                if i % 10 == 0:
                    print('Saved frame ' + str(i) + '/' + str(horizon))

            rgb_arr = self.env.map_to_colors()
            full_obs[i] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

            if dones["__all__"]:
                print('Environment ended early because all agents were done.')
                break

        return rewards, observations, full_obs

    def render_rollout(self, horizon=50, path=None,
                       render_type='pretty', fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            rewards, observations, full_obs = self.rollout(
                horizon=horizon, save_path=image_path)
            utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                    video_name=video_name)

            # Clean up images
            shutil.rmtree(image_path)
        else:
            rewards, observations, full_obs = self.rollout(horizon=horizon)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                                   video_name=video_name)


def main(unused_argv):
    c = Controller(env_name=FLAGS.env)
    c.render_rollout(path=FLAGS.vid_path,
                     horizon=FLAGS.horizon,
                     render_type=FLAGS.render_type,
                     fps=FLAGS.fps)


if __name__ == '__main__':
    tf.app.run(main)
