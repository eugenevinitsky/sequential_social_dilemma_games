import sys

import numpy as np
import scipy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()

MOA_PREDS = "moa_preds"
OTHERS_ACTIONS = "others_actions"
ALL_ACTIONS = "all_actions"
VISIBILITY = "others_visibility"
VISIBILITY_MATRIX = "visibility_matrix"
SOCIAL_INFLUENCE_REWARD = "total_influence_reward"

# Frozen logits of the policy that computed the action
ACTION_LOGITS = "action_logits"
COUNTERFACTUAL_ACTIONS = "counterfactual_actions"
POLICY_SCOPE = "func"


class InfluenceScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_influence_reward_weight = config["influence_reward_weight"]
        self.influence_reward_schedule_steps = config["influence_reward_schedule_steps"]
        self.influence_reward_schedule_weights = config["influence_reward_schedule_weights"]
        self.timestep = 0
        self.cur_influence_reward_weight = np.float32(self.compute_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_influence_reward_weight_tensor = tf.get_variable(
            "cur_influence_reward_weight",
            initializer=self.cur_influence_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(InfluenceScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_influence_reward_weight = self.compute_weight()
        self.cur_influence_reward_weight_tensor.load(
            self.cur_influence_reward_weight, session=self._sess
        )

    def compute_weight(self):
        """ Computes multiplier for influence reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.influence_reward_schedule_steps,
            self.influence_reward_schedule_weights,
        )
        return weight * self.baseline_influence_reward_weight


def kl_div(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete probability dists

    Assumes the probability dist is over the last dimension.

    Taken from: https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7

    p, q : array-like, dtype=float
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    kl = np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=-1)

    # Don't return nans or infs
    if np.all(np.isfinite(kl)):
        return kl
    else:
        return np.zeros(kl.shape)


class MOALoss(object):
    def __init__(self, pred_logits, true_actions, loss_weight=1.0, others_visibility=None):
        """Train MOA model with supervised cross entropy loss on a trajectory.
        The model is trying to predict others' actions at timestep t+1 given all
        actions at timestep t.
        Returns:
            A scalar loss tensor (cross-entropy loss).
        """
        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        action_logits = pred_logits[:-1, :, :]  # [B, N, A]

        # # Remove first agent (self) and first action, because we want to predict
        # # the t+1 actions of other agents from all actions at t.
        true_actions = tf.cast(true_actions[1:, 1:], tf.int32)  # [B, N]

        # Compute softmax cross entropy
        self.ce_per_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_actions, logits=action_logits
        )

        # Zero out the loss if the other agent isn't visible to this one.
        if others_visibility is not None:
            # Remove first entry in ground truth visibility and flatten
            others_visibility = others_visibility[1:, :]
            self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        self.total_loss = tf.reduce_mean(self.ce_per_entry) * loss_weight
        tf.Print(self.total_loss, [self.total_loss], message="MOA CE loss")


def setup_moa_loss(logits, model, policy, train_batch):
    # Instantiate the prediction loss
    moa_preds = model.moa_preds_from_batch(train_batch)
    moa_preds = tf.reshape(moa_preds, [-1, policy.model.num_other_agents, logits.shape[-1]])
    true_actions = train_batch[ALL_ACTIONS]
    # 0/1 multiplier array representing whether each agent is visible to
    # the current agent.
    if policy.train_moa_only_when_visible:
        # if VISIBILITY in train_batch:
        others_visibility = train_batch[VISIBILITY]
    else:
        others_visibility = None
    moa_loss = MOALoss(
        moa_preds,
        true_actions,
        loss_weight=policy.moa_loss_weight,
        others_visibility=others_visibility,
    )
    return moa_loss


def moa_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Extract matrix of self and other agents' actions.
    own_actions = np.atleast_2d(np.array(sample_batch["actions"]))
    own_actions = np.reshape(own_actions, [-1, 1])
    all_actions = np.hstack((own_actions, sample_batch[OTHERS_ACTIONS]))
    sample_batch[ALL_ACTIONS] = all_actions

    # Compute social influence reward and add to batch.
    sample_batch = compute_influence_reward(policy, sample_batch)

    return sample_batch


def compute_influence_reward(policy, trajectory):
    """Compute influence of this agent on other agents and add to rewards.
    """
    # Probability of the next action for all other agents. Shape is [B, N, A].
    # This is the predicted probability given the actions that we DID take.
    # extract out the probability under the actions we actually did take
    true_probs = trajectory[COUNTERFACTUAL_ACTIONS]
    traj_index = list(range(len(trajectory["obs"])))
    true_probs = true_probs[traj_index, :, trajectory["actions"], :]
    true_probs = np.reshape(true_probs, [true_probs.shape[0], policy.num_other_agents, -1])
    true_probs = scipy.special.softmax(true_probs, axis=-1)
    true_probs = true_probs / true_probs.sum(axis=-1, keepdims=1)  # reduce numerical inaccuracies

    # Get marginal predictions where effect of self is marginalized out
    marginal_probs = marginalize_predictions_over_own_actions(
        policy, trajectory
    )  # [B, Num agents, Num actions]

    # Compute influence per agent/step ([B, N]) using different metrics
    if policy.influence_divergence_measure == "kl":
        influence_per_agent_step = kl_div(true_probs, marginal_probs)
    elif policy.influence_divergence_measure == "jsd":
        mean_probs = 0.5 * (true_probs + marginal_probs)
        influence_per_agent_step = 0.5 * kl_div(true_probs, mean_probs) + 0.5 * kl_div(
            marginal_probs, mean_probs
        )
    else:
        sys.exit("Please specify an influence divergence measure from [kl, jsd]")

    # Zero out influence for steps where the other agent isn't visible.
    if policy.influence_only_when_visible:
        # if VISIBILITY in trajectory.keys():
        visibility = trajectory[VISIBILITY]
        # else:
        #     visibility = get_agent_visibility_multiplier(trajectory, policy.num_other_agents)
        influence_per_agent_step *= visibility

    cur_influence_reward_weight = policy.compute_weight()

    # Summarize and clip influence reward
    influence = np.sum(influence_per_agent_step, axis=-1)
    influence = np.clip(influence, -policy.influence_reward_clip, policy.influence_reward_clip)
    influence = influence * cur_influence_reward_weight

    # Add to trajectory
    trajectory[SOCIAL_INFLUENCE_REWARD] = influence
    trajectory["extrinsic_reward"] = trajectory["rewards"]
    trajectory["rewards"] = trajectory["rewards"] + influence

    return trajectory


def agent_name_to_idx(agent_num, self_id):
    """split agent id around the index and return its appropriate position in terms
    of the other agents"""
    agent_num = int(agent_num)
    if agent_num > self_id:
        return agent_num - 1
    else:
        return agent_num


def get_agent_visibility_multiplier(trajectory, num_other_agents, agent_ids):
    traj_len = len(trajectory["obs"])
    visibility = np.zeros((traj_len, num_other_agents))
    for i, v in enumerate(trajectory[VISIBILITY]):
        vis_agents = [agent_name_to_idx(a, agent_ids[i]) for a in v]
        visibility[i, vis_agents] = 1
    return visibility


def marginalize_predictions_over_own_actions(policy, trajectory):
    # Probability of each action in original trajectory
    action_probs = scipy.special.softmax(trajectory[ACTION_LOGITS], axis=-1)

    # Normalize to reduce numerical inaccuracies
    action_probs = action_probs / action_probs.sum(axis=1, keepdims=1)

    # Indexing of this is [B, Num agents, Agent actions, other agent logits] before we marginalize
    counter_probs = trajectory[COUNTERFACTUAL_ACTIONS]
    counter_probs = np.reshape(
        counter_probs, [counter_probs.shape[0], policy.num_other_agents, -1, action_probs.shape[-1]],
    )
    counter_probs = scipy.special.softmax(counter_probs, axis=-1)
    marginal_probs = np.sum(counter_probs, axis=-2)

    # Multiply by probability of each action to renormalize probability
    tiled_probs = np.tile(action_probs, [1, policy.num_other_agents, 1])
    marginal_probs = np.multiply(marginal_probs, tiled_probs)

    # Normalize to reduce numerical inaccuracies
    marginal_probs = marginal_probs / marginal_probs.sum(axis=2, keepdims=1)

    return marginal_probs


def extract_last_actions_from_episodes(episodes, batch_type=False, own_actions=None):
    """Pulls every other agent's previous actions out of structured data.
    Args:
        episodes: the structured data type. Typically a dict of episode
            objects.
        batch_type: if True, the structured data is a dict of tuples,
            where the second tuple element is the relevant dict containing
            previous actions.
        own_actions: an array of the agents own actions. If provided, will
            be the first column of the created action matrix.
    Returns: a real valued array of size [batch, num_other_agents] (meaning
        each agents' actions goes down one column, each row is a timestep)
    """
    if episodes is None:
        print("Why are there no episodes?")
        import ipdb

        ipdb.set_trace()

    # Need to sort agent IDs so same agent is consistently in
    # same part of input space.
    agent_ids = sorted(episodes.keys())
    prev_actions = []

    for agent_id in agent_ids:
        if batch_type:
            prev_actions.append(episodes[agent_id][1]["actions"])
        else:
            prev_actions.append([e.prev_action for e in episodes[agent_id]])

    all_actions = np.transpose(np.array(prev_actions))

    # Attach agents own actions as column 1
    if own_actions is not None:
        all_actions = np.hstack((own_actions, all_actions))

    return all_actions


def moa_fetches(policy):
    """Adds logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        # Be aware that this is frozen here so that we don't
        # propagate agent actions through the reward
        ACTION_LOGITS: policy.model.action_logits(),
        COUNTERFACTUAL_ACTIONS: policy.model.counterfactual_actions(),
        # TODO(@evinitsky) remove this once we figure out how to split the obs
        OTHERS_ACTIONS: policy.model.other_agent_actions(),
        VISIBILITY: policy.model.visibility(),
    }


class MOAConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.num_other_agents = config["num_other_agents"]
        self.moa_loss_weight = config["moa_loss_weight"]
        self.influence_reward_clip = config["influence_reward_clip"]
        self.train_moa_only_when_visible = config["train_moa_only_when_visible"]
        self.influence_divergence_measure = config["influence_divergence_measure"]
        self.influence_only_when_visible = config["influence_only_when_visible"]


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space, action_space, logit_dim, config["model"], name=POLICY_SCOPE, framework="tf",
    )

    return policy.model


def setup_moa_mixins(policy, obs_space, action_space, config):
    InfluenceScheduleMixIn.__init__(policy, config)
    MOAConfigInitializerMixIn.__init__(policy, config)


def get_moa_mixins():
    return [MOAConfigInitializerMixIn, InfluenceScheduleMixIn]
