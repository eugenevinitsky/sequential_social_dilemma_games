"""Note: Keep in sync with changes to VTraceTFPolicy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.a3c.a3c import validate_config
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages

tf = try_import_tf()


class A3CLoss(object):
    def __init__(
        self,
        action_dist,
        actions,
        advantages,
        v_target,
        vf,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
    ):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (
            self.pi_loss + self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff
        )


def postprocess_a3c_aux(
    aux_postprocess_trajectory_fn,
    policy,
    sample_batch,
    other_agent_batches=None,
    episode=None,
):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = aux_postprocess_trajectory_fn(policy, sample_batch)
    batch = postprocess_advantages(policy, batch)
    return batch


def actor_critic_loss(setup_aux_loss_fn, policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.loss = A3CLoss(
        action_dist,
        train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.VALUE_TARGETS],
        model.value_function(),
        policy.config["vf_loss_coeff"],
        policy.config["entropy_coeff"],
    )

    aux_loss = setup_aux_loss_fn(logits, model, policy, train_batch)
    policy.loss.total_loss += aux_loss.total_loss

    # store this for future statistics
    policy.aux_loss = aux_loss.total_loss

    return policy.loss.total_loss


def add_value_function_fetch(aux_fetches_fn, policy):
    fetch = {SampleBatch.VF_PREDS: policy.model.value_function()}
    fetch.update(aux_fetches_fn(policy))
    return fetch


class ValueNetworkMixin(object):
    def __init__(self):
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model(
                {
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                    "is_training": tf.convert_to_tensor(False),
                },
                [tf.convert_to_tensor([s]) for s in state],
                tf.convert_to_tensor([1]),
            )
            return self.model.value_function()[0]

        self._value = value


def stats(policy, train_batch):
    base_stats = {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss.pi_loss,
        "policy_entropy": policy.loss.entropy,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "vf_loss": policy.loss.vf_loss,
        "cur_aux_reward_weight": tf.cast(
            policy.cur_aux_reward_weight_tensor, tf.float64
        ),
        "total_aux_reward": train_batch["total_aux_reward"],
        "reward_without_aux": train_batch["reward_without_aux"],
        "aux_loss": policy.aux_loss * policy.aux_loss_weight,
    }
    return base_stats


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function()
        ),
    }


def clip_gradients(policy, optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(
        loss, policy.model.trainable_variables()
    )
    grads = [g for (g, v) in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
    clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    return clipped_grads


def setup_mixins(setup_fn, policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_fn(policy, obs_space, action_space, config)


def get_a3c_trainer(aux_model_type, aux_config):
    if aux_model_type == "causal":
        from algorithms.common_funcs_causal import (
            setup_moa_loss as setup_aux_loss,
            causal_fetches as aux_fetches,
            setup_causal_mixins as setup_aux_mixins,
            get_causal_mixins as get_aux_mixins,
            causal_postprocess_trajectory as aux_postprocess_trajectory,
        )

        model_name = "CausalMOAA3C"
    elif aux_model_type == "curiosity":
        from algorithms.common_funcs_curiosity import (
            setup_curiosity_loss as setup_aux_loss,
            curiosity_fetches as aux_fetches,
            setup_curiosity_mixins as setup_aux_mixins,
            get_curiosity_mixins as get_aux_mixins,
            curiosity_postprocess_trajectory as aux_postprocess_trajectory,
        )

        model_name = "CuriosityA3C"

    aux_config["use_gae"] = False

    a3c_tf_policy = build_tf_policy(
        name="A3CTFPolicy",
        get_default_config=lambda: aux_config,
        loss_fn=partial(actor_critic_loss, setup_aux_loss),
        stats_fn=stats,
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=partial(postprocess_a3c_aux, aux_postprocess_trajectory),
        extra_action_fetches_fn=partial(add_value_function_fetch, aux_fetches),
        before_loss_init=partial(setup_mixins, setup_aux_mixins),
        mixins=[ValueNetworkMixin, LearningRateSchedule] + get_aux_mixins(),
    )

    trainer = build_trainer(
        name=model_name,
        default_policy=a3c_tf_policy,
        default_config=aux_config,
        validate_config=validate_config,
    )

    return trainer
