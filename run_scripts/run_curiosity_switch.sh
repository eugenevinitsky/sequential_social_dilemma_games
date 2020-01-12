#!/usr/bin/env bash

python train.py \
--exp_name curiosity_switch \
--env switch \
--model curiosity \
--algorithm A3C \
--sample_batch_size 1000 \
--train_batch_size 30000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_workers 12 \
--num_cpus_per_worker 1 \
--num_gpus_per_worker 0.25 \
--num_gpus_for_driver 1 \
--num_cpus_for_driver 1 \
--num_switches 1 \
--num_samples 5
