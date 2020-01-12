#!/usr/bin/env bash
# Attempt to limit the memory usage to 4gb, with 1gb for object store and 1gb for redis

python train.py \
--exp_name curiosity_switch_memory_experiment \
--env switch \
--model curiosity \
--algorithm A3C \
--sample_batch_size 8 \
--train_batch_size 64 \
--stop_at_timesteps_total $((1 * 10 ** 9)) \
--memory $((4 * 10 ** 9)) \
--object_store_memory $((1 * 10 ** 9)) \
--redis_max_memory $((1 * 10 ** 9)) \
--num_envs_per_worker 8 \
--num_workers 4 \
--num_cpus_per_worker 1 \
--num_gpus_per_worker 0.2 \
--num_gpus_for_driver 0.2 \
--num_cpus_for_driver 1 \
--num_switches 6 \
--num_samples 1
