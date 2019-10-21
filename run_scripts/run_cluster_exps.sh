#!/usr/bin/env bash

ray exec ray_autoscale.yaml "python train_moa.py --train_batch_size 26000 --training_iterations 1000 --num_cpus 14 \
--exp_name test_moa --use_s3 --num_agents 5 --exp_name reproduce_test" --cluster-name=hparam_sweep --tmux --start --stop