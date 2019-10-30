#!/usr/bin/env bash

python train_curiosity.py \
--train_batch_size 1000 \
--sample_batch_size 30000 \
--num_cpus 2 \
--exp_name curiosity_switch \
--stop_at_timesteps_total 50000 \
--algorithm A3C