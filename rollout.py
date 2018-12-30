"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

#from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv

import utility_funcs
import numpy as np
import os
import sys
import shutil


class Controller(object):

    def __init__(self, env_name='cleanup'):
        self.env_name = env_name
        if env_name == 'harvest':
            self.env = HarvestEnv(num_agents=5, render=True)
        elif env_name == 'cleanup':
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
                self.env.render_map(save_path + 'frame' + str(i) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[i] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        return rewards, observations, full_obs
            

    def render_rollout(self, horizon=50, path=None, 
                       render_type='pretty'):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out. 
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            rewards, observations, full_obs = self.rollout(
                horizon=horizon, save_path=image_path)
            utility_funcs.make_video_from_image_dir(path, image_path)
            
            # Clean up images
            shutil.rmtree(image_path)
        else:
            rewards, observations, full_obs = self.rollout(horizon=horizon)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path)


if __name__ == '__main__':
    c = Controller()
    if len(sys.argv) > 1:
        vid_path = sys.argv[1]
        c.render_rollout(path=vid_path)
    else:
        c.render_rollout()
