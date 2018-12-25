"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

from social_dilemmas.envs.harvest import HarvestEnv
# from social_dilemmas.envs.cleanup import CleanupEnv

import utility_funcs
import numpy as np
import os
import sys
import shutil


class Controller(object):

    def __init__(self):
        self.env = HarvestEnv(num_agents=5, render=True)
        self.env.reset()

        # TODO: initialize agents here

    def rollout_and_render(self, horizon=100000, render_frames=False,
                           render_full_vid=False, path=None):
        rewards = []
        observations = []

        if render_full_vid:
            if path is None:
                path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
                if not os.path.exists(path):
                    os.makedirs(path)
            images_path = path + '/images/'
            if not os.path.exists(images_path):
                os.makedirs(images_path)

            shape = self.env.map.shape
            full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

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

            if render_frames:
                self.env.render_map()

            if render_full_vid:
                rgb_arr = self.env.map_to_colors()
                full_obs[i] = rgb_arr.astype(np.uint8)

            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        if render_full_vid:
            utility_funcs.make_video_from_rgb_imgs(full_obs, path)

            # Clean up images
            shutil.rmtree(images_path)


if __name__ == '__main__':
    c = Controller()
    if len(sys.argv) > 1:
        vid_path = sys.argv[1]
        c.rollout_and_render(path=vid_path)
    else:
        c.rollout_and_render()
