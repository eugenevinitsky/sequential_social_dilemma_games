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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'vid_path', '/home/natasha/Dropbox (MIT)/Projects/AgentEmpathy/vids',
    'Path to directory where videos are saved.')
tf.app.flags.DEFINE_string(
    'env', 'cleanup',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'render_type', 'pretty', 
    'Can be pretty or fast. Implications obvious.')
tf.app.flags.DEFINE_integer(
    'fps', 5,
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
        shape = self.env.map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        for i in range(horizon):
            # TODO: use agent policy not just random actions
            rand_action = np.random.randint(8, size=5)
            obs, rew, dones, info, = self.env.step({'agent-0': rand_action[0],
                                                    'agent-1': rand_action[1],
                                                    'agent-2': rand_action[2],
                                                    'agent-3': rand_action[3],
                                                    'agent-4': rand_action[4]})

            print("timestep", i, "action", rand_action, "reward", rew['agent-0'])
            sys.stdout.flush()

            if save_path is not None:
                self.env.render_map(filename=save_path + 'frame' + str(i).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[i] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        return rewards, observations, full_obs
            

    def render_rollout(self, horizon=50, path=None, 
                       render_type='pretty', fps=5):
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
        video_name=self.env_name + '_trajectory'

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
    c.render_rollout(path=FLAGS.vid_path, render_type=FLAGS.render_type,
                     fps=FLAGS.fps)


if __name__ == '__main__':
    tf.app.run(main)
    