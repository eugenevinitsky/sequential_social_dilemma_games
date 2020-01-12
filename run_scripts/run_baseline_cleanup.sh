#!/usr/bin/env bash

python train.py \
--exp_name cleanup_baseline_PPO \
--env cleanup \
--model baseline \
--algorithm PPO \
--num_agents 5 \
--sample_batch_size 1000 \
--train_batch_size 12000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_workers 12 \
--num_cpus_per_worker 1 \
--num_gpus_per_worker 0.25 \
--num_gpus_for_driver 1 \
--num_cpus_for_driver 1 \
--num_samples 1 \
--num_envs_per_worker 4 \
--lr_curriculum_steps 0 20000000 \
--lr_curriculum_weights 0.00126 0.000012 \
--entropy_coeff 0.00176