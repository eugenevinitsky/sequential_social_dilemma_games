from __future__ import absolute_import, division, print_function

from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer


def build_ppo_baseline_trainer_with_config(config):
    ppo_trainer = build_trainer(
        name="PPO",
        default_config=config,
        default_policy=PPOTFPolicy,
        make_policy_optimizer=choose_policy_optimizer,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
    )
    return ppo_trainer
