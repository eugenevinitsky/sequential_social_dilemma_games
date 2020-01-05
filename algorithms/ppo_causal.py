from functools import partial

from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_policy import (
    BEHAVIOUR_LOGITS,
    KLCoeffMixin,
    PPOLoss,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    setup_config,
    vf_preds_and_logits_fetches,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP, EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

POLICY_SCOPE = "func"


def loss_with_aux(setup_aux_loss_fn, policy, model, dist_class, train_batch):
    # you need to override this bit to pull out the right bits from train_batch
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    aux_loss = setup_aux_loss_fn(logits, model, policy, train_batch)
    policy.aux_loss = aux_loss.total_loss

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"],
    )

    policy.loss_obj.loss += aux_loss.total_loss
    return policy.loss_obj.loss


def extra_fetches(aux_fetches_fn, policy):
    """Adds value function, logits, aux predictions to experience train_batches."""
    ppo_fetches = vf_preds_and_logits_fetches(policy)
    ppo_fetches.update(aux_fetches_fn(policy))
    return ppo_fetches


def extra_stats(policy, train_batch):
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "cur_aux_reward_weight": tf.cast(policy.cur_aux_reward_weight_tensor, tf.float32),
        "total_aux_reward": train_batch["total_aux_reward"],
        "reward_without_aux": train_batch["reward_without_aux"],
        "aux_loss": policy.aux_loss * policy.aux_loss_weight,
    }

    return base_stats


def postprocess_ppo_aux(
    aux_postprocess_trajectory_fn, policy, sample_batch, other_agent_batches=None, episode=None
):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = aux_postprocess_trajectory_fn(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space, action_space, logit_dim, config["model"], name=POLICY_SCOPE, framework="tf",
    )

    return policy.model


def setup_mixins(setup_aux_mixins_fn, policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_aux_mixins_fn(policy, obs_space, action_space, config)


def get_ppo_trainer(aux_model_type, aux_config):
    tf.keras.backend.set_floatx("float32")

    if aux_model_type == "moa":
        from algorithms.common_funcs_causal import (
            setup_moa_loss as setup_aux_loss,
            causal_fetches as aux_fetches,
            setup_causal_mixins as setup_aux_mixins,
            get_causal_mixins as get_aux_mixins,
            causal_postprocess_trajectory as aux_postprocess_trajectory,
        )

        trainer_name = "CausalMOAPPOTrainer"
    elif aux_model_type == "curiosity":
        from algorithms.common_funcs_curiosity import (
            setup_curiosity_loss as setup_aux_loss,
            curiosity_fetches as aux_fetches,
            setup_curiosity_mixins as setup_aux_mixins,
            get_curiosity_mixins as get_aux_mixins,
            curiosity_postprocess_trajectory as aux_postprocess_trajectory,
        )

        trainer_name = "CuriosityPPOTrainer"
    else:
        raise NotImplementedError

    aux_ppo_policy = build_tf_policy(
        name="PPOAuxTFPolicy",
        get_default_config=lambda: aux_config,
        loss_fn=partial(loss_with_aux, setup_aux_loss),
        make_model=build_model,
        stats_fn=extra_stats,
        extra_action_fetches_fn=partial(extra_fetches, aux_fetches),
        postprocess_fn=partial(postprocess_ppo_aux, aux_postprocess_trajectory),
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=partial(setup_mixins, setup_aux_mixins),
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin]
        + get_aux_mixins(),
    )

    aux_ppo_trainer = build_trainer(
        name=trainer_name,
        default_policy=aux_ppo_policy,
        make_policy_optimizer=choose_policy_optimizer,
        default_config=aux_config,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
    )

    return aux_ppo_trainer
