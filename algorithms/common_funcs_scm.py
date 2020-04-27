import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from algorithms.common_funcs_moa import compute_influence_reward, get_moa_mixins, moa_fetches

tf = try_import_tf()


ENCODED_OBSERVATIONS = "enc_obs"
PREDICTED_OBSERVATIONS = "pred_obs"


class SocialCuriosityScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_social_curiosity_reward_weight = config["social_curiosity_reward_weight"]
        self.social_curiosity_reward_schedule_steps = config[
            "social_curiosity_reward_schedule_steps"
        ]
        self.social_curiosity_reward_schedule_weights = config[
            "social_curiosity_reward_schedule_weights"
        ]
        self.timestep = 0
        self.cur_social_curiosity_reward_weight = np.float32(self.compute_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_social_curiosity_reward_weight_tensor = tf.get_variable(
            "cur_social_curiosity_reward_weight",
            initializer=self.cur_social_curiosity_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(SocialCuriosityScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_social_curiosity_reward_weight = self.compute_weight()
        self.cur_social_curiosity_reward_weight_tensor.load(
            self.cur_social_curiosity_reward_weight, session=self._sess
        )

    def compute_weight(self):
        """ Computes multiplier for social_curiosity reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.social_curiosity_reward_schedule_steps,
            self.social_curiosity_reward_schedule_weights,
        )
        return weight * self.baseline_social_curiosity_reward_weight


def calculate_surprisal(pred_states, true_states):
    """Surprisal with self-supervised MSE on a trajectory.

     The loss is based on the difference between the predicted encoding of the observation x at t+1
     based on t, and the true encoding x at t+1.
     The loss is then -log(p(xt+1)|xt, at)
     Difference is measured as mean-squared error corresponding to a fixed-variance Gaussian density.

    Returns:
        A scalar loss tensor.
    """
    # Remove the prediction for the final step, since t+1 is not known for this step.
    pred_states = pred_states[:-1]  # [Batch size, Size of encoded observations]

    # Remove first true state, as we have nothing to predict this from.
    # the t+1 actions of other agents from all actions at t.
    true_states = true_states[1:]

    # Compute mean squared error of difference between prediction and truth
    mse = np.square(true_states - pred_states).mean()

    return mse


class SCMLoss(object):
    def __init__(self, pred_states, true_states, pred_influence, true_influence, loss_weight=1.0):
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
        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        pred_states = pred_states[:-1]  # [Batch size, Size of encoded observations]

        # Remove first true state, as we have nothing to predict this from.
        # the t+1 actions of other agents from all actions at t.
        true_states = true_states[1:]

        # Compute mean squared error of difference between prediction and truth
        mse = tf.losses.mean_squared_error(pred_states, true_states)

        self.total_loss = mse * loss_weight


def setup_scm_loss(model, policy, train_batch):
    # Instantiate the prediction loss
    predicted_states = model.predicted_encoded_observations()
    true_states = model.true_encoded_observations()
    predicted_influence = model.predicted_influence()
    true_influence = train_batch["total_influence_reward"]

    scm_loss = SCMLoss(
        predicted_states,
        true_states,
        predicted_influence,
        true_influence,
        loss_weight=policy.scm_loss_weight,
    )
    return scm_loss


def scm_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Compute curiosity reward and add to batch.
    sample_batch = compute_influence_reward(policy, sample_batch)
    sample_batch = compute_curiosity_reward(policy, sample_batch)
    return sample_batch


def compute_curiosity_reward(policy, trajectory):
    """Compute curiosity of this agent and add to rewards.
    """
    # Probability of the next action for all other agents.
    # Shape is [B, N, A]. This is the predicted probability
    # given the actions that we DID take.
    # extract out the probability under the actions we actually did take
    true_obs = trajectory[ENCODED_OBSERVATIONS]
    pred_obs = trajectory[PREDICTED_OBSERVATIONS]

    curiosity_reward_per_agent_step = [
        calculate_surprisal(pred, truth) for pred, truth in zip(pred_obs, true_obs)
    ]
    cur_curiosity_reward_weight = policy.compute_weight()

    # Clip curiosity reward
    reward = np.clip(
        curiosity_reward_per_agent_step, -policy.curiosity_reward_clip, policy.curiosity_reward_clip
    )
    reward = reward * cur_curiosity_reward_weight

    # Add to trajectory
    trajectory["total_curiosity_reward"] = reward
    trajectory["extrinsic_reward"] = trajectory["rewards"]
    trajectory["rewards"] = trajectory["rewards"] + reward

    return trajectory


def scm_fetches(policy):
    """Adds observations and causal influence to experience train_batches."""
    return {
        ENCODED_OBSERVATIONS: policy.model.true_encoded_observations(),
        PREDICTED_OBSERVATIONS: policy.model.predicted_encoded_observations(),
        **moa_fetches(policy),
    }


def setup_scm_mixins(policy, obs_space, action_space, config):
    SocialCuriosityScheduleMixIn.__init__(policy, config)


def get_curiosity_mixins():
    return get_moa_mixins() + [SocialCuriosityScheduleMixIn]
