#!/usr/bin/env bash

python train.py \
--exp_name harvest_moa_ppo \
--env harvest \
--model moa \
--algorithm PPO \
--num_agents 5 \
--rollout_fragment_length 1000 \
--train_batch_size 128000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((25 * 10 ** 9)) \
--num_workers 16 \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 1 \
--num_envs_per_worker 8 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.00136 0.000028 \
--entropy_coeff 0.000687 \
--aux_loss_weight 0.091650628 \
--aux_reward_weight 1.0 \
--aux_reward_schedule_steps 0 10000000 100000000 300000000 \
--aux_reward_schedule_weights 0.0 0.0 1.0 0.5 \
--ppo_sgd_minibatch_size 32000
