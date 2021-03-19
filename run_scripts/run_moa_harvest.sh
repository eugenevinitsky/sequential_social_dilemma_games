#!/usr/bin/env bash

python train.py \
--env harvest \
--model moa \
--algorithm PPO \
--num_agents 5 \
--num_workers 6 \
--rollout_fragment_length 1000 \
--num_envs_per_worker 32 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((160 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 4 \
--entropy_coeff 0.00223 \
--moa_loss_weight 0.091650628 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.0012 0.000044 \
--influence_reward_weight 2.521 \
--influence_reward_schedule_steps 0 10000000 100000000 300000000 \
--influence_reward_schedule_weights 0.0 0.0 1.0 0.5