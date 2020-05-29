"""Adapted from A3CTFPolicy to add V-trace.

Keep in sync with changes to A3CTFPolicy and VtraceSurrogatePolicy."""

from __future__ import absolute_import, division, print_function

import logging

import gym
import numpy as np
from ray.rllib.agents.impala import DEFAULT_CONFIG
from ray.rllib.agents.impala.impala import (
    OverrideDefaultResourceRequest,
    defer_make_workers,
    make_aggregators_and_optimizer,
    validate_config,
)
from ray.rllib.agents.impala.vtrace_tf_policy import VTraceLoss, choose_optimizer, clip_gradients
from ray.rllib.agents.impala.vtrace_tf_policy import validate_config as validate_config_policy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance

from algorithms.common_funcs_moa import (
    EXTRINSIC_REWARD,
    SOCIAL_INFLUENCE_REWARD,
    get_moa_mixins,
    moa_fetches,
    moa_postprocess_trajectory,
    setup_moa_loss,
    setup_moa_mixins,
)

MOA_CONFIG = DEFAULT_CONFIG


tf = try_import_tf()

logger = logging.getLogger(__name__)

BEHAVIOUR_LOGITS = "behaviour_logits"


def _make_time_major(policy, seq_lens, tensor, drop_last=False):
    """Swaps batch and trajectory axis.

    Arguments:
        policy: Policy reference
        seq_lens: Sequence lengths if recurrent or None
        tensor: A tensor or list of tensors to reshape.
        drop_last: A bool indicating whether to drop the last
        trajectory item.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, list):
        return [_make_time_major(policy, seq_lens, t, drop_last) for t in tensor]

    if policy.is_recurrent():
        B = tf.shape(seq_lens)[0]
        T = tf.shape(tensor)[0] // B
    else:
        # Important: chop the tensor into batches at known episode cut
        # boundaries. TODO(ekl) this is kind of a hack
        T = policy.config["rollout_fragment_length"]
        B = tf.shape(tensor)[0] // T
    rs = tf.reshape(tensor, tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))

    # swap B and T axes
    res = tf.transpose(rs, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

    if drop_last:
        return res[:-1]
    return res


def build_vtrace_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
        output_hidden_shape = [policy.action_space.n]
    elif isinstance(policy.action_space, gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
        output_hidden_shape = policy.action_space.nvec.astype(np.int32)
    else:
        is_multidiscrete = False
        output_hidden_shape = 1

    def make_time_major(*args, **kw):
        return _make_time_major(policy, train_batch.get("seq_lens"), *args, **kw)

    actions = train_batch[SampleBatch.ACTIONS]
    dones = train_batch[SampleBatch.DONES]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
    behaviour_logits = train_batch[BEHAVIOUR_LOGITS]
    unpacked_behaviour_logits = tf.split(behaviour_logits, output_hidden_shape, axis=1)
    unpacked_outputs = tf.split(logits, output_hidden_shape, axis=1)
    values = model.value_function()

    if policy.is_recurrent():
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else tf.expand_dims(actions, axis=1)

    # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
    policy.loss = VTraceLoss(
        actions=make_time_major(loss_actions, drop_last=True),
        actions_logp=make_time_major(action_dist.logp(actions), drop_last=True),
        actions_entropy=make_time_major(action_dist.multi_entropy(), drop_last=True),
        dones=make_time_major(dones, drop_last=True),
        behaviour_action_logp=make_time_major(behaviour_action_logp, drop_last=True),
        behaviour_logits=make_time_major(unpacked_behaviour_logits, drop_last=True),
        target_logits=make_time_major(unpacked_outputs, drop_last=True),
        discount=policy.config["gamma"],
        rewards=make_time_major(rewards, drop_last=True),
        values=make_time_major(values, drop_last=True),
        bootstrap_value=make_time_major(values)[-1],
        dist_class=Categorical if is_multidiscrete else dist_class,
        model=model,
        valid_mask=make_time_major(mask, drop_last=True),
        config=policy.config,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.entropy_coeff,
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"],
    )

    moa_loss = setup_moa_loss(logits, policy, train_batch)
    policy.loss.total_loss += moa_loss.total_loss

    # store this for future statistics
    policy.moa_loss = moa_loss.total_loss

    return policy.loss.total_loss


def moa_stats(policy, train_batch):
    values_batched = _make_time_major(
        policy,
        train_batch.get("seq_lens"),
        policy.model.value_function(),
        drop_last=policy.config["vtrace"],
    )

    base_stats = {
        "cur_lr": tf.cast(policy.cur_lr, tf.float32),
        "policy_loss": policy.loss.pi_loss,
        "entropy": policy.loss.entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float32),
        "var_gnorm": tf.global_norm(policy.model.trainable_variables()),
        "vf_loss": policy.loss.vf_loss,
        "vf_explained_var": explained_variance(
            tf.reshape(policy.loss.value_targets, [-1]), tf.reshape(values_batched, [-1]),
        ),
        SOCIAL_INFLUENCE_REWARD: train_batch[SOCIAL_INFLUENCE_REWARD],
        EXTRINSIC_REWARD: train_batch[EXTRINSIC_REWARD],
        "moa_loss": policy.moa_loss / policy.moa_weight,
    }
    return base_stats


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
    }


def postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    sample_batch = moa_postprocess_trajectory(policy, sample_batch)
    del sample_batch.data[SampleBatch.NEXT_OBS]
    return sample_batch


def add_behaviour_logits(policy):
    fetches = {BEHAVIOUR_LOGITS: policy.model.last_output()}
    fetches.update(moa_fetches(policy))
    return fetches


def setup_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    setup_moa_mixins(policy, obs_space, action_space, config)


def get_moa_vtrace_policy():
    moa_vtrace_policy = build_tf_policy(
        name="MOAVTracePolicy",
        get_default_config=lambda: MOA_CONFIG,
        loss_fn=build_vtrace_loss,
        stats_fn=moa_stats,
        grad_stats_fn=grad_stats,
        postprocess_fn=postprocess_trajectory,
        optimizer_fn=choose_optimizer,
        gradients_fn=clip_gradients,
        extra_action_fetches_fn=add_behaviour_logits,
        before_init=validate_config_policy,
        before_loss_init=setup_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule] + get_moa_mixins(),
        get_batch_divisibility_req=lambda p: p.config["rollout_fragment_length"],
    )
    return moa_vtrace_policy


def choose_policy(config):
    if config["vtrace"]:
        return get_moa_vtrace_policy()
    else:
        import sys

        sys.exit("Hey, set vtrace to true")


def build_impala_moa_trainer(config):
    moa_impala_trainer = build_trainer(
        name="MOAIMPALA",
        default_config=config,
        default_policy=get_moa_vtrace_policy(),
        validate_config=validate_config,
        get_policy_class=choose_policy,
        make_workers=defer_make_workers,
        make_policy_optimizer=make_aggregators_and_optimizer,
        mixins=[OverrideDefaultResourceRequest],
    )
    return moa_impala_trainer
