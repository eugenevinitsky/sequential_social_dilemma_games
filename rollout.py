"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

from social_dilemmas.envs.harvest import HarvestEnv
import utility_funcs
import numpy as np
import os
import sys
import shutil

# TODO: Agents incorporated and controlled from here. 



class Controller(object):

    def __init__(self):
        self.env = HarvestEnv(num_agents=1, render=True)
        self.env.reset()

        # TODO: initialize agents here

    def rollout_and_render(self, horizon=50, render_frames=False,
                           render_full_vid=True):
        actions = []
        rewards = []
        observations = []

        if render_full_vid:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            images_path = path + '/images/'
            if not os.path.exists(images_path): os.makedirs(images_path)

        for i in range(horizon):
            # TODO: use agent policy not just random actions
            rand_action = np.random.randint(8)
            obs, rew, dones, info, = self.env.step({'agent-0': rand_action})

            print("timestep", i, "action", rand_action, "reward", rew['agent-0'])
            sys.stdout.flush()

            if render_frames:
                self.env.render_map()

            if render_full_vid:
                rgb_arr = self.env.map_to_colors()
                utility_funcs.save_img(rgb_arr, images_path, 'img' + str(i) + '.png')

            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        if render_full_vid:
            utility_funcs.make_video_from_image_dir(path, img_folder=images_path)

            # Clean up images
            shutil.rmtree(images_path)


if __name__ == '__main__':
    c = Controller()
    c.rollout_and_render()
