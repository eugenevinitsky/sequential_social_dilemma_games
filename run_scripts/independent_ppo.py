import time
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch as th
import pettingzoo

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, configure_logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3 import PPO


class IndependentGymEnv(gym.Env):
    def __init__(self, idx, observation_space, action_space):
        self.idx = idx
        self.observation_space = observation_space # SB3 is wrapping this in TransposeVecEnv
        self.action_space = action_space
        self.num_envs = 1


class IndependentPPO:
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        par_env: pettingzoo.ParallelEnv,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
    ):
        self.par_env = par_env
        self.agents = par_env.possible_agents
        self.num_agents = len(self.agents)
        self.observation_space = par_env.observation_spaces[self.agents[0]]
        self.action_space = par_env.action_spaces[self.agents[0]]
        self.n_steps = n_steps
        self.policies = [
            PPO(
                policy=policy,
                env=IndependentGymEnv(index, self.observation_space, self.action_space),
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
            )
            for index in range(self.num_agents)
        ]

    def learn(
        self,
        total_timesteps: int,
        callbacks: Optional[List[MaybeCallback]] = None,
        log_interval: int = 1,
        tb_log_name: str = "IndependentPPO",
        reset_num_timesteps: bool = True,
    ):

        num_timesteps = 0
        all_total_timesteps = []
        if not callbacks:
            callbacks = [None] * self.num_agents

        # Setup for each policy
        for index, policy in enumerate(self.policies):
            policy.start_time = time.time()
            if policy.ep_info_buffer is None or reset_num_timesteps:
                policy.ep_info_buffer = deque(maxlen=100)
                policy.ep_success_buffer = deque(maxlen=100)

            if policy.action_noise is not None:
                policy.action_noise.reset()

            if reset_num_timesteps:
                policy.num_timesteps = 0
                policy._episode_num = 0
                all_total_timesteps.append(total_timesteps)
                policy._total_timesteps = total_timesteps
            else:
                # make sure training timestamps are ahead of internal counter
                all_total_timesteps.append(total_timesteps + policy.num_timesteps)
                policy._total_timesteps = total_timesteps + policy.num_timesteps

            if not policy._custom_logger:
                policy._logger = configure_logger(
                    policy.verbose,
                    policy.tensorboard_log,
                    tb_log_name + f"_{index}",
                    reset_num_timesteps,
                )

            callbacks[index] = policy._init_callback(callbacks[index])

        for callback in callbacks:
            callback.on_training_start(locals(), globals())

        all_last_obs = self.par_env.reset()

        while num_timesteps < total_timesteps:
            all_last_obs = self.collect_rollouts(all_last_obs, callbacks)
            num_timesteps += 1
            for policy in self.policies:
                policy._update_current_progress_remaining(num_timesteps, total_timesteps)
                if log_interval is not None and num_timesteps % log_interval == 0:
                    fps = int(policy.num_timesteps / (time.time() - policy.start_time))
                    policy.logger.record("time/iterations", num_timesteps, exclude="tensorboard")
                    if len(policy.ep_info_buffer) > 0 and len(policy.ep_info_buffer[0]) > 0:
                        policy.logger.record(
                            "rollout/ep_rew_mean",
                            safe_mean([ep_info["r"] for ep_info in policy.ep_info_buffer]),
                        )
                        policy.logger.record(
                            "rollout/ep_len_mean",
                            safe_mean([ep_info["l"] for ep_info in policy.ep_info_buffer]),
                        )
                    policy.logger.record("time/fps", fps)
                    policy.logger.record(
                        "time/time_elapsed",
                        int(time.time() - policy.start_time),
                        exclude="tensorboard",
                    )
                    policy.logger.record(
                        "time/total_timesteps", policy.num_timesteps, exclude="tensorboard"
                    )
                    policy.logger.dump(step=policy.num_timesteps)

                policy.train()

        for callback in callbacks:
            callback.on_training_end()

    def collect_rollouts(self, all_last_obs, callbacks):

        all_actions = {}
        all_values = {}
        all_log_probs = {}
        all_clipped_actions = {}
        all_last_episode_starts = {}
        steps = 0

        for index, policy in enumerate(self.policies):
            assert (
                all_last_obs[self.agents[index]] is not None
            ), f"No previous observation was provided for policy_{index}"
            policy.policy.set_training_mode(False)
            policy.rollout_buffer.reset()
            callbacks[index].on_rollout_start()

        while steps < self.n_steps:
            with th.no_grad():
                for index, policy in enumerate(self.policies):
                    agent = self.agents[index]
                    obs_tensor = obs_as_tensor(all_last_obs[agent], policy.device)
                    obs_tensor = obs_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                    (
                        all_actions[agent],
                        all_values[agent],
                        all_log_probs[agent],
                    ) = policy.policy.forward(obs_tensor)
                    clipped_actions = all_actions[agent].cpu().numpy()
                    if isinstance(self.action_space, Box):
                        clipped_actions = np.clip(
                            clipped_actions, self.action_space.low, self.action_space.high
                        )
                    elif isinstance(self.action_space, Discrete):
                        clipped_actions = clipped_actions.item() # get action as int from numpy array
                    all_clipped_actions[agent] = clipped_actions
                    all_last_episode_starts[agent] = policy._last_episode_starts

            all_obs, all_rewards, all_dones, all_infos = self.par_env.step(all_clipped_actions)

            for policy in self.policies:
                policy.num_timesteps += 1

            for callback in callbacks:
                callback.update_locals(locals())
            if not [callback.on_step() for callback in callbacks]:
                break

            for index, policy in enumerate(self.policies):
                agent = self.agents[index]
                policy._update_info_buffer(all_infos[agent])

            steps += 1

            if isinstance(self.action_space, Discrete):
                for agent in self.agents:
                    all_actions[agent] = all_actions[agent].reshape(-1, 1)

            # add data to the rollout buffers
            for index, policy in enumerate(self.policies):
                agent = self.agents[index]
                all_actions[agent] = all_actions[agent].cpu().numpy()
                all_last_obs[agent] = np.transpose(np.expand_dims(all_last_obs[agent], 0), axes=(0, 3, 1, 2))
                policy.rollout_buffer.add(
                    all_last_obs[agent],
                    all_actions[agent],
                    all_rewards[agent],
                    all_last_episode_starts[agent],
                    all_values[agent],
                    all_log_probs[agent],
                )
            all_last_obs = all_obs
            all_last_episode_starts = all_dones

        with th.no_grad():
            for index, policy in enumerate(self.policies):
                agent = self.agents[index]
                obs_tensor = obs_as_tensor(all_last_obs[agent], policy.device)
                obs_tensor = obs_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                _, value, _ = policy.policy.forward(obs_tensor)
                policy.rollout_buffer.compute_returns_and_advantage(
                    last_values=value, dones=all_dones[agent]
                )

        for callback in callbacks:
            callback.on_rollout_end()

        return all_last_obs
