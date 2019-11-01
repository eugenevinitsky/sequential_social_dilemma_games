import numpy as np

from ray.rllib.utils import try_import_tf

tf = try_import_tf()

# Frozen logits of the policy that computed the action
ACTION_LOGITS = "action_logits"
POLICY_SCOPE = "func"
ENCODED_OBSERVATIONS = "enc_obs"
PREDICTED_OBSERVATIONS = "pred_obs"


def calculate_surprisal(pred_states, true_states):
    """Surprisal with self-supervised MSE on a trajectory.

     The loss is based on the difference between the predicted encoding of the observation x at t+1 based on t,
     and the true encoding x at t+1.
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


class CuriosityLoss(object):
    def __init__(self, pred_states, true_states, loss_weight=1.0):
        """Surprisal with self-supervised MSE on a trajectory.

         The loss is based on the difference between the predicted encoding of the observation x at t+1 based on t,
         and the true encoding x at t+1.
         The loss is then -log(p(xt+1)|xt, at)
         Difference is measured as mean-squared error corresponding to a fixed-variance Gaussian density.

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


def setup_curiosity_loss(logits, model, policy, train_batch):
    # Instantiate the prediction loss
    aux_preds = model.predicted_encoded_observations()
    true_states = model.true_encoded_observations()
    curiosity_loss = CuriosityLoss(aux_preds, true_states, loss_weight=policy.aux_loss_weight)
    return curiosity_loss


def curiosity_postprocess_trajectory(policy,
                                     sample_batch,
                                     other_agent_batches=None,
                                     episode=None):
    # Compute curiosity reward and add to batch.
    sample_batch = compute_curiosity_reward(policy, sample_batch)
    return sample_batch


def compute_curiosity_reward(policy, trajectory):
    """Compute curiosity of this agent and add to rewards.
    """
    # Probability of the next action for all other agents. Shape is [B, N, A]. This is the predicted probability
    # given the actions that we DID take.
    # extract out the probability under the actions we actually did take
    true_obs = trajectory[ENCODED_OBSERVATIONS]
    pred_obs = trajectory[PREDICTED_OBSERVATIONS]

    aux_reward_per_agent_step = [calculate_surprisal(pred, truth) for pred, truth in zip(pred_obs, true_obs)]

    # Clip curiosity reward
    reward = np.clip(aux_reward_per_agent_step, -policy.aux_reward_clip, policy.aux_reward_clip)

    # Get influence curriculum weight
    # TODO(@internetcoffeephone) move this into a schedule mixin
    policy.steps_processed += len(trajectory['obs'])

    # Add to trajectory
    trajectory['total_aux_reward'] = reward
    trajectory['reward_without_aux'] = trajectory['rewards']
    trajectory['rewards'] = trajectory['rewards'] + (reward * policy.curr_aux_reward_weight)

    return trajectory


def curiosity_fetches(policy):
    """Adds value function, logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        ACTION_LOGITS: policy.model.action_logits(),
        ENCODED_OBSERVATIONS: policy.model.true_encoded_observations(),
        PREDICTED_OBSERVATIONS: policy.model.predicted_encoded_observations(),
    }


class ConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config['model']['custom_options']
        self.num_other_agents = config['num_other_agents']
        self.aux_loss_weight = config['aux_loss_weight']
        self.aux_reward_clip = config['aux_reward_clip']


class AuxScheduleMixIn(object):
    def __init__(self, config):
        config = config['model']['custom_options']
        self.aux_reward_weight = config['aux_reward_weight']
        self.aux_reward_curriculum_time = config['aux_reward_curriculum_steps']
        self.aux_reward_curriculum_weights = config['aux_reward_curriculum_weights']
        self.steps_processed = 0
        self.curr_aux_reward_weight = self.aux_reward_weight

    def current_aux_curriculum_weight(self):
        """ Computes multiplier for aux reward based on training steps
        taken and curriculum parameters.
        """
        scale = np.interp(self.steps_processed,
                          self.aux_reward_curriculum_steps,
                          self.aux_reward_curriculum_weights)
        self.curr_aux_reward_weight = scale * self.aux_reward_weight


def setup_curiosity_mixins(policy, obs_space, action_space, config):
    AuxScheduleMixIn.__init__(policy, config)
    ConfigInitializerMixIn.__init__(policy, config)


def get_curiosity_mixins():
    return [ConfigInitializerMixIn, AuxScheduleMixIn]
