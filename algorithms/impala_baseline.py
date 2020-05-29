from __future__ import absolute_import, division, print_function

from ray.rllib.agents.impala.impala import (
    OverrideDefaultResourceRequest,
    choose_policy,
    defer_make_workers,
    make_aggregators_and_optimizer,
    validate_config,
)
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceTFPolicy
from ray.rllib.agents.trainer_template import build_trainer


def build_impala_baseline_trainer(config):
    impala_trainer = build_trainer(
        name="IMPALA",
        default_config=config,
        default_policy=VTraceTFPolicy,
        validate_config=validate_config,
        get_policy_class=choose_policy,
        make_workers=defer_make_workers,
        make_policy_optimizer=make_aggregators_and_optimizer,
        mixins=[OverrideDefaultResourceRequest],
    )
    return impala_trainer
