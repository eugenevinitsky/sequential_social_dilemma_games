#!/usr/bin/env bash

python train.py \
--exp_name cleanup_baseline_PPO \
--env cleanup \
--model baseline \
--algorithm PPO \
--num_agents 5 \
--sample_batch_size 100 \
--train_batch_size 19200 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_cpus 24 \
--num_gpus 4 \
--use_gpu_for_driver \
--use_gpus_for_workers \
--num_workers_per_device 0.125 \
--num_samples 1 \
--lr_curriculum_steps 0 20000000 \
--lr_curriculum_weights 0.00126 0.000012 \
--entropy_coeff 0.00176