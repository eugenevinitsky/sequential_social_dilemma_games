"""Note: Keep in sync with changes to VTraceTFPolicy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG, \
    validate_config
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages

from algorithms.common_funcs import setup_moa_loss, causal_fetches, setup_causal_mixins, get_causal_mixins, \
    causal_postprocess_trajectory, CAUSAL_CONFIG

CAUSAL_CONFIG["use_gae"] = False
CAUSAL_CONFIG.update(DEFAULT_CONFIG)

tf = try_import_tf()


class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff -
                           self.entropy * entropy_coeff)


def postprocess_a3c_causal(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = causal_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_advantages(policy, batch)
    return batch


def actor_critic_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.loss = A3CLoss(action_dist, train_batch[SampleBatch.ACTIONS],
                          train_batch[Postprocessing.ADVANTAGES],
                          train_batch[Postprocessing.VALUE_TARGETS],
                          model.value_function(),
                          policy.config["vf_loss_coeff"],
                          policy.config["entropy_coeff"])

    moa_loss = setup_moa_loss(logits, model, policy, train_batch)
    policy.loss.total_loss += moa_loss.total_loss

    # store this for future statistics
    policy.moa_loss = moa_loss.total_loss

    return policy.loss.total_loss


def add_value_function_fetch(policy):
    fetch = {SampleBatch.VF_PREDS: policy.model.value_function()}
    fetch.update(causal_fetches(policy))
    return fetch


class ValueNetworkMixin(object):
    def __init__(self):
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model({
                SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                "is_training": tf.convert_to_tensor(False),
            }, [tf.convert_to_tensor([s]) for s in state],
                tf.convert_to_tensor([1]))
            return self.model.value_function()[0]

        self._value = value


def stats(policy, train_batch):
    base_stats = {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss.pi_loss,
        "policy_entropy": policy.loss.entropy,
        "var_gnorm": tf.global_norm(
            [x for x in policy.model.trainable_variables()]),
        "vf_loss": policy.loss.vf_loss,
    }
    base_stats["total_influence"] = train_batch["total_influence"]
    base_stats['reward_without_influence'] = train_batch['reward_without_influence']
    base_stats['moa_loss'] = policy.moa_loss / policy.moa_weight
    return base_stats


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
    }


def clip_gradients(policy, optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(
        loss, policy.model.trainable_variables())
    grads = [g for (g, v) in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
    clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    return clipped_grads


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_causal_mixins(policy, obs_space, action_space, config)

A3CTFPolicy = build_tf_policy(
    name="A3CTFPolicy",
    get_default_config=lambda: CAUSAL_CONFIG,
    loss_fn=actor_critic_loss,
    stats_fn=stats,
    grad_stats_fn=grad_stats,
    gradients_fn=clip_gradients,
    postprocess_fn=postprocess_a3c_causal,
    extra_action_fetches_fn=add_value_function_fetch,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin, LearningRateSchedule] + get_causal_mixins())

CausalA3CMOATrainer = build_trainer(
    name="CausalMOAA3C",
    default_policy=A3CTFPolicy,
    default_config=CAUSAL_CONFIG,
    validate_config=validate_config)