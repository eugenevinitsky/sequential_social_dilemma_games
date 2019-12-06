#!/usr/bin/env bash
# Attempt to limit the memory usage to 4gb, with 1gb for object store and 1gb for redis

python train_curiosity.py \
--exp_name curiosity_switch_memory_experiment \
--env switch \
--algorithm A3C \
--sample_batch_size 10 \
--train_batch_size 100 \
--stop_at_timesteps_total $((1 * 10 ** 9)) \
--memory $((4 * 10 ** 9)) \
--object_store_memory $((1 * 10 ** 9)) \
--redis_max_memory $((1 * 10 ** 9)) \
--num_cpus 5 \
--num_gpus 0 \
--num_switches 6 \
--num_samples 1
