#!/usr/bin/env bash

python train.py \
--env harvest \
--model scm \
--algorithm PPO \
--num_agents 5 \
--num_workers 2 \
--rollout_fragment_length 64 \
--num_envs_per_worker 32 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((160 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 0.333 \
--cpus_for_driver 0 \
--num_samples 5 \
--entropy_coeff 0.00223 \
--moa_loss_weight 0.091650628 \
--influence_reward_weight 2.521 \
--scm_loss_weight 1.0 \
--scm_forward_vs_inverse_loss_weight 0.5 \
--curiosity_reward_weight 0.001 \
--tune_hparams