from __future__ import absolute_import, division, print_function

from ray.rllib.agents.a3c.a3c import get_policy_class, make_async_optimizer, validate_config
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.trainer_template import build_trainer


def build_a3c_baseline_trainer(config):
    a3c_trainer = build_trainer(
        name="A3C",
        default_config=config,
        default_policy=A3CTFPolicy,
        get_policy_class=get_policy_class,
        validate_config=validate_config,
        make_policy_optimizer=make_async_optimizer,
    )
    return a3c_trainer
