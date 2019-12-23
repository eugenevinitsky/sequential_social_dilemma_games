#!/usr/bin/env bash

# 10/21 experiments

# THIS IS A GPU INSTRUCTION
#ray exec ray_autoscale.yaml "source activate tensorflow_p36 && python sequential_social_dilemma_games/run_scripts/train.py --training_iterations 100000 --num_cpus 30 \
#--num_gpus 2 --exp_name moa_harvest --use_s3 --num_agents 5 --exp_name reproduce_test --env harvest --algorithm A3C --num_envs_per_worker 1" --cluster-name=a3c --start --stop

ray exec ray_autoscale.yaml "source activate tensorflow_p36 && python sequential_social_dilemma_games/run_scripts/train_moa.py --training_iterations 100000 --num_cpus 30 \
--num_gpus 2 --exp_name moa_harvest --use_s3 --num_agents 5 --exp_name reproduce_test --env harvest --algorithm IMPALA --num_envs_per_worker 1" --cluster-name=imp --start --stop

#ray exec ray_autoscale.yaml "python sequential_social_dilemma_games/run_scripts/train.py --train_batch_size 26000 --training_iterations 1000 --num_cpus 14 \
#--exp_name moa_harvest --use_s3 --num_agents 5 --exp_name reproduce_test --grid_search --env harvest --multi_node" --tmux --cluster-name=hparam_sweep --start --stop