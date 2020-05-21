#!/usr/bin/env bash

python train.py \
--env cleanup \
--model scm \
--algorithm PPO \
--num_agents 5 \
--num_workers 1 \
--rollout_fragment_length 64 \
--num_envs_per_worker 64 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((25 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 1 \
--lr_schedule_steps 0 100000000 \
--lr_schedule_weights 0.00120 0.000044 \
--entropy_coeff 0.00176 \
--moa_loss_weight 0.06663557 \
--influence_reward_weight 0.620 \
--influence_reward_schedule_steps 0 100000000 \
--influence_reward_schedule_weights 0.0 1.0 \
--scm_loss_weight 1.0 \
--scm_forward_vs_inverse_loss_weight 0.5 \
--curiosity_reward_weight 0.001