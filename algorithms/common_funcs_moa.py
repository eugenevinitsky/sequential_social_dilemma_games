import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from algorithms.common_funcs_baseline import BaselineResetConfigMixin

tf = try_import_tf()

MOA_PREDS = "moa_preds"
OTHERS_ACTIONS = "others_actions"
PREDICTED_ACTIONS = "predicted_actions"
VISIBILITY = "others_visibility"
VISIBILITY_MATRIX = "visibility_matrix"
SOCIAL_INFLUENCE_REWARD = "social_influence_reward"
EXTRINSIC_REWARD = "extrinsic_reward"

# Frozen logits of the policy that computed the action
ACTION_LOGITS = "action_logits"
COUNTERFACTUAL_ACTIONS = "counterfactual_actions"
POLICY_SCOPE = "func"


class InfluenceScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_influence_reward_weight = config["influence_reward_weight"]
        if any(
            config[key] is None
            for key in ["influence_reward_schedule_steps", "influence_reward_schedule_weights"]
        ):
            self.compute_influence_reward_weight = lambda: self.baseline_influence_reward_weight
        self.influence_reward_schedule_steps = config["influence_reward_schedule_steps"]
        self.influence_reward_schedule_weights = config["influence_reward_schedule_weights"]
        self.timestep = 0
        self.cur_influence_reward_weight = np.float32(self.compute_influence_reward_weight())
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
        self.cur_influence_reward_weight = self.compute_influence_reward_weight()
        self.cur_influence_reward_weight_tensor.load(
            self.cur_influence_reward_weight, session=self._sess
        )

    def compute_influence_reward_weight(self):
        """ Computes multiplier for influence reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.influence_reward_schedule_steps,
            self.influence_reward_schedule_weights,
        )
        return weight * self.baseline_influence_reward_weight


class MOALoss(object):
    def __init__(self, pred_logits, true_actions, loss_weight=1.0, others_visibility=None):
        """Train MOA model with supervised cross entropy loss on a trajectory.
        The model is trying to predict others' actions at timestep t+1 given all
        actions at timestep t.
        Returns:
            A scalar loss tensor (cross-entropy loss).
        """
        # Pred_logits[n] contains the prediction made at n-1 for actions taken at n, and a prediction
        # for t=0 cannot have been made at timestep -1, as the simulation starts at timestep 0.
        # Thus we remove the first prediction, as this value contains no sensible data.
        # NB: This means we start at n=1.
        action_logits = pred_logits[1:-1, :, :]  # [B, N, A]

        # true_actions[n] contains the actions made by other agents at n-1.
        # Therefore, true_actions[2] contains actions made at time 1,
        # which is where action_logits starts.
        true_actions = tf.cast(true_actions[2:, :], tf.int32)  # [B, N]

        # Compute softmax cross entropy
        self.ce_per_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_actions, logits=action_logits
        )

        # Zero out the loss if the other agent isn't visible to this one.
        if others_visibility is not None:
            # others_visibility[n] contains agents visible at time n. We start at n=1,
            # so the first and last values have to be removed to maintain equal array size.
            others_visibility = others_visibility[1:-1, :]
            self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        # Flatten loss to one value for the entire batch
        self.total_loss = tf.reduce_mean(self.ce_per_entry) * loss_weight
        tf.Print(self.total_loss, [self.total_loss], message="MOA CE loss")


def setup_moa_loss(logits, policy, train_batch):
    # Instantiate the prediction loss
    moa_preds = train_batch[PREDICTED_ACTIONS]
    moa_preds = tf.reshape(moa_preds, [-1, policy.model.num_other_agents, logits.shape[-1]])
    true_actions = train_batch[OTHERS_ACTIONS]
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
    # Weigh social influence reward and add to batch.
    sample_batch = weigh_and_add_influence_reward(policy, sample_batch)

    return sample_batch


def weigh_and_add_influence_reward(policy, sample_batch):
    cur_influence_reward_weight = policy.compute_influence_reward_weight()
    # Since the reward calculation is delayed by 1 step, sample_batch[SOCIAL_INFLUENCE_REWARD][0]
    # contains the reward for timestep -1, which does not exist. Hence we shift the array.
    # Then, pad with a 0-value at the end to make the influence rewards align with sample_batch.
    # This leaks some information about the episode end though.
    influence = np.concatenate((sample_batch[SOCIAL_INFLUENCE_REWARD][1:], [0]))

    # Clip and weigh influence reward
    influence = np.clip(influence, -policy.influence_reward_clip, policy.influence_reward_clip)
    influence = influence * cur_influence_reward_weight

    # Add to trajectory
    sample_batch[SOCIAL_INFLUENCE_REWARD] = influence
    sample_batch["extrinsic_reward"] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + influence

    return sample_batch


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
        # TODO(@evinitsky) remove this once we figure out how to split the obs
        OTHERS_ACTIONS: policy.model.other_agent_actions(),
        VISIBILITY: policy.model.visibility(),
        SOCIAL_INFLUENCE_REWARD: policy.model.social_influence_reward(),
        PREDICTED_ACTIONS: policy.model.predicted_actions(),
    }


class MOAConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.num_other_agents = config["num_other_agents"]
        self.moa_loss_weight = tf.get_variable(
            "moa_loss_weight", initializer=config["moa_loss_weight"], trainable=False
        )
        self.influence_reward_clip = config["influence_reward_clip"]
        self.train_moa_only_when_visible = config["train_moa_only_when_visible"]
        self.influence_divergence_measure = config["influence_divergence_measure"]
        self.influence_only_when_visible = config["influence_only_when_visible"]


class MOAResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config, session):
        custom_options = new_config["model"]["custom_options"]
        for policy in policies:
            policy.moa_loss_weight.load(custom_options["moa_loss_weight"], session=session)
            policy.compute_influence_reward_weight = lambda: custom_options[
                "influence_reward_weight"
            ]

    def reset_config(self, new_config):
        policies = self.optimizer.policies.values()
        BaselineResetConfigMixin.reset_policies(policies, new_config)
        self.reset_policies(policies, new_config, self.optimizer.sess)
        self.config = new_config
        return True


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
    return [
        MOAConfigInitializerMixIn,
        InfluenceScheduleMixIn,
    ]


def validate_moa_config(config):
    config = config["model"]["custom_options"]
    if config["influence_reward_weight"] < 0:
        raise ValueError("Influence reward weight must be >= 0.")
