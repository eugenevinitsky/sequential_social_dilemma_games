import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from algorithms.common_funcs_baseline import BaselineResetConfigMixin
from algorithms.common_funcs_moa import MOAResetConfigMixin

tf = try_import_tf()


PREDICTED_OBSERVATIONS = "pred_obs"
SOCIAL_CURIOSITY_REWARD = "social_curiosity_reward"
INVERSE_MODEL_LOSS = "inverse_model_loss"


class SocialCuriosityScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_curiosity_reward_weight = config["curiosity_reward_weight"]
        if any(
            config[key] is None
            for key in ["curiosity_reward_schedule_steps", "curiosity_reward_schedule_weights"]
        ):
            self.compute_curiosity_reward_weight = lambda: self.baseline_curiosity_reward_weight

        self.curiosity_reward_schedule_steps = config["curiosity_reward_schedule_steps"]
        self.curiosity_reward_schedule_weights = config["curiosity_reward_schedule_weights"]
        self.timestep = 0
        self.cur_curiosity_reward_weight = np.float32(self.compute_curiosity_reward_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_curiosity_reward_weight_tensor = tf.get_variable(
            "cur_curiosity_reward_weight",
            initializer=self.cur_curiosity_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(SocialCuriosityScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_curiosity_reward_weight = self.compute_curiosity_reward_weight()
        self.cur_curiosity_reward_weight_tensor.load(
            self.cur_curiosity_reward_weight, session=self._sess
        )

    def compute_curiosity_reward_weight(self):
        """ Computes multiplier for social_curiosity reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.curiosity_reward_schedule_steps,
            self.curiosity_reward_schedule_weights,
        )
        return weight * self.baseline_curiosity_reward_weight


class SCMLoss(object):
    def __init__(
        self,
        forward_loss,
        inverse_loss,
        scm_loss_weight=1.0,
        forward_loss_weight=0.5,
        inverse_loss_weight=0.5,
    ):
        """Surprisal with self-supervised MSE on a trajectory.

         The loss is based on the difference between the predicted encoding of the observation x
         at t+1 based on t,
         and the true encoding x at t+1.
         The loss is then -log(p(xt+1)|xt, at)
         Difference is measured as mean-squared error corresponding to a
         fixed-variance Gaussian density.

        Returns:
            A scalar loss tensor.
        """
        # Remove the first value, as this contains no sensible data.
        loss = tf.reduce_mean(
            forward_loss[1:] * forward_loss_weight + inverse_loss[1:] * inverse_loss_weight
        )

        self.total_loss = loss * scm_loss_weight


def setup_scm_loss(policy, train_batch):
    # The forward loss is equivalent to the social curiosity reward
    forward_loss = train_batch[SOCIAL_CURIOSITY_REWARD]
    inverse_loss = train_batch[INVERSE_MODEL_LOSS]

    scm_loss = SCMLoss(
        forward_loss,
        inverse_loss,
        scm_loss_weight=policy.scm_loss_weight,
        forward_loss_weight=policy.forward_loss_weight,
        inverse_loss_weight=policy.inverse_loss_weight,
    )
    return scm_loss


def scm_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Weigh social curiosity reward and add to batch.
    sample_batch = weigh_and_add_curiosity_reward(policy, sample_batch)
    return sample_batch


def weigh_and_add_curiosity_reward(policy, sample_batch):
    """Compute curiosity of this agent and add to rewards.
    """
    cur_curiosity_reward_weight = policy.compute_curiosity_reward_weight()
    # Align the reward, as the reward for timestep n is calculated at timestep n+1.
    curiosity_reward = np.concatenate((sample_batch[SOCIAL_CURIOSITY_REWARD][1:], [0]))

    # Clip curiosity reward
    reward = np.clip(curiosity_reward, -policy.curiosity_reward_clip, policy.curiosity_reward_clip)
    reward = reward * cur_curiosity_reward_weight

    # Add to trajectory
    sample_batch[SOCIAL_CURIOSITY_REWARD] = reward
    sample_batch["rewards"] = sample_batch["rewards"] + reward

    return sample_batch


def scm_fetches(policy):
    """Adds observations and causal influence to experience train_batches."""
    return {
        SOCIAL_CURIOSITY_REWARD: policy.model.social_curiosity_reward(),
        INVERSE_MODEL_LOSS: policy.model.inverse_model_loss(),
    }


class SCMConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.scm_loss_weight = tf.get_variable(
            "scm_loss_weight", initializer=config["scm_loss_weight"], trainable=False
        )
        self.curiosity_reward_clip = config["curiosity_reward_clip"]
        self.forward_loss_weight = tf.get_variable(
            "forward_loss_weight",
            initializer=config["scm_forward_vs_inverse_loss_weight"],
            trainable=False,
        )
        self.inverse_loss_weight = tf.get_variable(
            "inverse_loss_weight",
            initializer=1 - config["scm_forward_vs_inverse_loss_weight"],
            trainable=False,
        )


class SCMResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config, session):
        custom_options = new_config["model"]["custom_options"]
        for policy in policies:
            policy.scm_loss_weight.load(custom_options["scm_loss_weight"], session=session)
            policy.compute_curiosity_reward_weight = lambda: custom_options[
                "curiosity_reward_weight"
            ]
            policy.forward_loss_weight.load(
                custom_options["scm_forward_vs_inverse_loss_weight"], session=session
            )
            policy.inverse_loss_weight.load(
                1 - custom_options["scm_forward_vs_inverse_loss_weight"], session=session
            )

    def reset_config(self, new_config):
        policies = self.optimizer.policies.values()
        BaselineResetConfigMixin.reset_policies(policies, new_config)
        MOAResetConfigMixin.reset_policies(policies, new_config, self.optimizer.sess)
        self.reset_policies(policies, new_config, self.optimizer.sess)
        self.config = new_config
        return True


def setup_scm_mixins(policy, obs_space, action_space, config):
    SocialCuriosityScheduleMixIn.__init__(policy, config)
    SCMConfigInitializerMixIn.__init__(policy, config)


def get_curiosity_mixins():
    return [SCMConfigInitializerMixIn, SocialCuriosityScheduleMixIn]


def validate_scm_config(config):
    config = config["model"]["custom_options"]
    if config["curiosity_reward_weight"] < 0:
        raise ValueError("Influence reward weight must be >= 0.")
    weight = config["scm_forward_vs_inverse_loss_weight"]
    if not 0 <= weight <= 1:
        raise ValueError(
            "scm_forward_vs_inverse_loss_weight should have a value in the range"
            " [0, 1], but has the value " + str(weight)
        )
