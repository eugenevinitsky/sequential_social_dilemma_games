#!/usr/bin/env bash

python train.py \
--exp_name switch_baseline_ppo \
--env switch \
--model baseline \
--algorithm PPO \
--num_agents 1 \
--sample_batch_size 1000 \
--train_batch_size 24000 \
--stop_at_timesteps_total $((1 * 10 ** 6)) \
--num_workers 12 \
--num_cpus_per_worker 1 \
--num_gpus_per_worker 0.25 \
--num_gpus_for_driver 1 \
--num_cpus_for_driver 1 \
--num_samples 1 \
--num_envs_per_worker 2 \
--lr_schedule_steps 0 \
--lr_schedule_weights 0.001 \
--entropy_coeff 0.00176 \
--num_workers_per_device 3 \
--small_model
