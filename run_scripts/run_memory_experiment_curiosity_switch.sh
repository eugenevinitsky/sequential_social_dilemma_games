#!/usr/bin/env bash
# Attempt to limit the memory usage to 4gb, with 1gb for object store and 1gb for redis

python train_curiosity.py \
--exp_name curiosity_switch_memory_experiment \
--env switch \
--algorithm A3C \
--sample_batch_size 1000 \
--train_batch_size 30000 \
--stop_at_timesteps_total $((1 * 10 ** 9)) \
--memory $((4 * 10 ** 9)) \
--object_store_memory $((1 * 10 ** 9)) \
--redis_max_memory $((1 * 10 ** 9)) \
--num_cpus 8 \
--num_gpus 1 \
--use_gpu_for_driver \
--num_switches 6 \
--num_samples 10
