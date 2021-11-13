import argparse

import gym
import supersuit as ss
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from torch import nn
from torch.nn import functional as F

from config.default_args import add_default_args
from social_dilemmas.envs.pettingzoo_env import parallel_env
from independent_ppo import IndependentPPO

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=4,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        if torch.any(torch.isnan(features)):
            breakpoint()
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features


def main(args):
    # Config
    rollout_len = 1000  # length of training rollouts AND length at which env is reset
    num_cpus = 12  # number of cpus
    num_envs = 12  # number of parallel multi-agent environments
    num_agents = 2  # number of agents
    num_frames = 6  # number of frames to stack together, use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    ent_coef = 0.001  # entropy coefficient in loss
    clip_range = 0.2
    vf_coef = 0.5
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 30
    gae_lambda = 1.0
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40

    args.num_agents = num_agents
    env = parallel_env(max_cycles=rollout_len, ssd_args=args)
    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="stable_baselines3"
    )
    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    log = "./results/sb3/cleanup_ppo_baseline"

    model = IndependentPPO(
        "CnnPolicy",
        num_agents=2,
        env=env,
        learning_rate=lr,
        n_steps=rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        verbose=3,
        tensorboard_log=log,
    )
    model.learn(total_timesteps=5e6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    args = parser.parse_args()
    main(args)
