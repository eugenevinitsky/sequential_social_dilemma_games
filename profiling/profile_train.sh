#!/usr/bin/env bash
python -m cProfile -o prof-5.out ../run_scripts/train.py \
--exp_name TEST_cleanup_scm_ppo \
--env cleanup \
--model scm \
--algorithm PPO \
--num_agents 5 \
--rollout_fragment_length 2 \
--train_batch_size 2 \
--stop_at_timesteps_total 2 \
--memory $((1 * 10 ** 9)) \
--num_workers 1 \
--cpus_per_worker 6 \
--gpus_per_worker 0 \
--gpus_for_driver 1 \
--cpus_for_driver 0 \
--num_samples 1 \
--num_envs_per_worker 2 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.00126 0.000012 \
--entropy_coeff 0.00176 \
--ppo_sgd_minibatch_size 2
pyprof2calltree -i prof-5.out -k
