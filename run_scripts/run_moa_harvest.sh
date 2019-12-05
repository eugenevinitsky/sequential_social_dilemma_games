#!/usr/bin/env bash

python train_moa.py \
--exp_name harvest_moa \
--env harvest \
--algorithm A3C \
--num_agents 5 \
--sample_batch_size 1000 \
--train_batch_size 3000 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((50 * 10 ** 9)) \
--num_cpus 24 \
--num_gpus 4 \
--use_gpu_for_driver \
--num_samples 1 \
--lr_curriculum_weights 0.00136 0.000028 \
--entropy_coeff 0.000687 \
--aux_loss_weight 0.091650628
