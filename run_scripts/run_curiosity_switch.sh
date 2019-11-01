#!/usr/bin/env bash

python train_curiosity.py \
--exp_name curiosity_switch \
--env switch \
--algorithm A3C \
--sample_batch_size 1000 \
--train_batch_size 30000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_cpus 24 \
--num_gpus 4 \
--use_gpu_for_driver \
--num_switches 1 \
--num_samples 5
