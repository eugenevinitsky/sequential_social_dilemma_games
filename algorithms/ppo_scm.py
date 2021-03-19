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
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_moa import build_model, get_moa_mixins, moa_postprocess_trajectory
from algorithms.common_funcs_scm import (
    SOCIAL_CURIOSITY_REWARD,
    SCMResetConfigMixin,
    get_curiosity_mixins,
    scm_fetches,
    scm_postprocess_trajectory,
    setup_scm_loss,
    setup_scm_mixins,
    validate_scm_config,
)
from algorithms.ppo_moa import (
    extra_moa_fetches,
    extra_moa_stats,
    loss_with_moa,
    setup_ppo_moa_mixins,
    validate_moa_config,
)

tf = try_import_tf()


def loss_with_scm(policy, model, dist_class, train_batch):
    """
    Calculate PPO loss with SCM and MOA loss
    :return: Combined PPO+MOA+SCM loss
    """
    _ = loss_with_moa(policy, model, dist_class, train_batch)

    scm_loss = setup_scm_loss(policy, train_batch)
    policy.scm_loss = scm_loss.total_loss

    # PPO loss_obj has already been instantiated in loss_with_moa
    policy.loss_obj.loss += scm_loss.total_loss
    return policy.loss_obj.loss


def extra_scm_fetches(policy):
    """
    Adds value function, logits, moa predictions, SCM loss/reward to experience train_batches.
    :return: Updated fetches
    """
    ppo_fetches = extra_moa_fetches(policy)
    ppo_fetches.update(scm_fetches(policy))
    return ppo_fetches


def extra_scm_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined PPO+MOA+SCM stats
    """
    scm_stats = extra_moa_stats(policy, train_batch)
    scm_stats = {
        **scm_stats,
        "cur_curiosity_reward_weight": tf.cast(
            policy.cur_curiosity_reward_weight_tensor, tf.float32
        ),
        SOCIAL_CURIOSITY_REWARD: train_batch[SOCIAL_CURIOSITY_REWARD],
        "scm_loss": policy.scm_loss,
    }
    return scm_stats


def postprocess_ppo_scm(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the influence and curiosity reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = scm_postprocess_trajectory(policy, batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_scm_mixins(policy, obs_space, action_space, config):
    """
        Calls init on all PPO+MOA+SCM mixins in the policy
        """
    setup_ppo_moa_mixins(policy, obs_space, action_space, config)
    setup_scm_mixins(policy, obs_space, action_space, config)


def validate_ppo_scm_config(config):
    """
    Validates the PPO+MOA+SCM config
    :param config: The config to validate
    """
    validate_scm_config(config)
    validate_moa_config(config)
    validate_config(config)


def build_ppo_scm_trainer(scm_config):
    """
    Creates a SCM+MOA+PPO policy class, then creates a trainer with this policy.
    :param scm_config: The configuration dictionary.
    :return: A new SCM+MOA+PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")

    trainer_name = "SCMPPOTrainer"

    scm_ppo_policy = build_tf_policy(
        name="SCMPPOTFPolicy",
        get_default_config=lambda: scm_config,
        loss_fn=loss_with_scm,
        make_model=build_model,
        stats_fn=extra_scm_stats,
        extra_action_fetches_fn=extra_scm_fetches,
        postprocess_fn=postprocess_ppo_scm,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_scm_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_moa_mixins()
        + get_curiosity_mixins(),
    )

    scm_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=scm_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=scm_config,
        validate_config=validate_ppo_scm_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[SCMResetConfigMixin],
    )
    return scm_ppo_trainer
