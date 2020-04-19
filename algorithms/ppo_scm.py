from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    ValueNetworkMixin,
    clip_gradients,
    postprocess_ppo_gae,
    setup_config,
    vf_preds_and_logits_fetches,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_moa import build_model
from algorithms.common_funcs_scm import (
    get_curiosity_mixins,
    scm_fetches,
    scm_postprocess_trajectory,
    setup_scm_mixins,
)
from algorithms.ppo_moa import extra_moa_stats, loss_with_moa, setup_ppo_moa_mixins

tf = try_import_tf()


def loss_with_scm(policy, model, dist_class, train_batch):
    moa_loss = loss_with_moa(policy, model, dist_class, train_batch)
    # TODO: Add scm loss
    scm_loss = 0
    return moa_loss + scm_loss


def extra_scm_stats(policy, train_batch):
    scm_stats = extra_moa_stats(policy, train_batch)
    scm_stats = {
        **scm_stats,
        "cur_curiosity_reward_weight": tf.cast(
            policy.cur_influence_reward_weight_tensor, tf.float32
        ),
        "total_scm_reward": train_batch["total_scm_reward"],
        "moa_loss": policy.scm_loss * policy.scm_loss_weight,
    }
    return scm_stats


def extra_fetches(policy):
    """Adds value function, logits, moa predictions to experience train_batches."""
    ppo_fetches = vf_preds_and_logits_fetches(policy)
    ppo_fetches.update(scm_fetches(policy))
    return ppo_fetches


def postprocess_ppo_scm(policy, sample_batch, other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = scm_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_scm_mixins(policy, obs_space, action_space, config):
    setup_ppo_moa_mixins(policy, obs_space, action_space, config)
    setup_scm_mixins(policy, obs_space, action_space, config)


def build_ppo_scm_trainer(scm_config):
    tf.keras.backend.set_floatx("float32")

    trainer_name = "SCMPPOTrainer"

    scm_ppo_policy = build_tf_policy(
        name="SCMPPOTFPolicy",
        get_default_config=lambda: scm_config,
        loss_fn=loss_with_scm,
        make_model=build_model,
        stats_fn=extra_scm_stats,
        extra_action_fetches_fn=extra_fetches,
        postprocess_fn=postprocess_ppo_scm,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_scm_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_curiosity_mixins(),
    )

    scm_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=scm_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=scm_config,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
    )

    return scm_ppo_trainer
