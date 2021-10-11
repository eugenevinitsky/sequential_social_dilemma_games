from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from config.default_args import add_default_args
import argparse
import numpy as np
from social_dilemmas.envs.env_creator import get_env_creator
from social_dilemmas.envs.pettingzoo_env import MAX_CYCLES, parallel_env

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

import supersuit as ss
import gym
import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, view_len=7):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        flat_out = 6 * (view_len * 2 - 1) ** 2

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "curr_obs":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=6,  # Input: 3 x 15 x 15
                        kernel_size=3,
                        stride=1,
                        padding="valid",
                    ),  # Output: 6 x 13 x 13
                    nn.ReLU(),
                    nn.Flatten(),  # Output: 1014
                    nn.Linear(in_features=flat_out, out_features=32),
                    nn.ReLU(),  # Output: 32
                    nn.Linear(in_features=32, out_features=32),
                    nn.ReLU(),  # Output: 32
                )
                total_concat_size += np.prod(subspace.shape)
            else:
                continue

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "curr_obs":
                # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
                observations[key] = torch.div(torch.FloatTensor(observations[key]), 255).permute(
                    0, 3, 1, 2
                )
                return extractor(observations[key])
            else:
                continue
                observations[key] = torch.Tensor(observations[key])
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


# Use this with lambda wrapper returning observations
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32, view_len=7):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = 24 * (view_len * 2 - 1) ** 2
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=24,  # Input: (3 * 4) x 15 x 15
                kernel_size=3,
                stride=1,
                padding="valid",
            ),  # Output: 24 x 13 x 13
            nn.ReLU(),
            nn.Flatten(),  # Output: 4056
            nn.Linear(in_features=flat_out, out_features=1024),
            nn.ReLU(),  # Output: 1024
            nn.Linear(in_features=1024, out_features=32),
            nn.ReLU(),  # Output: 32
        )

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = torch.div(torch.tensor(observations, device=device), 255)
        return self.cnn(observations)


def main(args):
    env = parallel_env(max_cycles=MAX_CYCLES, ssd_args=args)
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, 4)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=8, base_class="stable_baselines3")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32, view_len=7),
        net_arch=[128],
    )

    model = PPO("CnnPolicy", env=env, policy_kwargs=policy_kwargs, verbose=3)
    model.learn(1e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    args = parser.parse_args()
    main(args)
