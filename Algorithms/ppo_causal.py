import copy
import sys

import numpy as np

# TODO(@evinitsky) put this in alphabetical order

import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, PPOLoss, BEHAVIOUR_LOGITS, \
    KLCoeffMixin, ValueNetworkMixin, setup_mixins, setup_config, clip_gradients, \
    kl_and_loss_stats, vf_preds_and_logits_fetches
# TODO(@evinitsky) move config vals into a default config
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, choose_policy_optimizer, \
    validate_config, update_kl, warn_about_bad_reward_scales
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

CONFIG = DEFAULT_CONFIG
CONFIG.update({"num_other_agents": 1,
               "moa_weight": 10.0,
               "train_moa_only_when_visible": True,
               "influence_reward_clip": 10,
               "influence_divergence_measure": 'kl',
               "influence_reward_weight": 1.0,
               "influence_curriculum_steps": 10e6,
               "influence_scaledown_start": 100e6,
               "influence_scaledown_end": 300e6,
               "influence_scaledown_final_val": .5,
               "influence_only_when_visible": True})

tf = try_import_tf()

MOA_PREDS = "moa_preds"
OTHERS_ACTIONs = "others_actions"

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"
COUNTERFACTUAL_ACTIONS = "counterfactual_actions"
POLICY_SCOPE = "func"


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


class MOALossMixIn(object):
    def __init__(self, config):
        cust_opts = config['model']['custom_options']
        self.moa_weight = cust_opts['moa_weight']
        self.train_moa_only_when_visible = cust_opts['train_moa_only_when_visible']


class MOALoss(object):
    def __init__(self, pred_logits, true_actions, num_actions,
                 loss_weight=1.0, others_visibility=None):
        """Train MOA model with supervised cross entropy loss on a trajectory.
        The model is trying to predict others' actions at timestep t+1 given all
        actions at timestep t.
        Returns:
            A scalar loss tensor (cross-entropy loss).
        """
        # Remove the prediction for the final step, since t+1 is not known for
        # this step.
        action_logits = pred_logits[:-1, :, :]  # [B, N, A]

        # Remove first agent (self) and first action, because we want to predict
        # the t+1 actions of other agents from all actions at t.
        true_actions = true_actions[1:, 1:]  # [B, N]

        # Compute softmax cross entropy
        flat_logits = tf.reshape(action_logits, [-1, num_actions])
        flat_labels = tf.reshape(true_actions, [-1])
        self.ce_per_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_labels, logits=flat_logits)

        # Zero out the loss if the other agent isn't visible to this one.
        if others_visibility is not None:
            # Remove first entry in ground truth visibility and flatten
            others_visibility = tf.reshape(others_visibility[1:, :], [-1])
            self.ce_per_entry *= tf.cast(others_visibility, tf.float32)

        self.total_loss = tf.reduce_mean(self.ce_per_entry)
        tf.Print(self.total_loss, [self.total_loss], message="MOA CE loss")


def loss_with_moa(policy, model, dist_class, train_batch):
    # you need to override this bit to pull out the right bits from train_batch
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    # you need to overwrite this function
    # TODO(@evinitsky) get the logit shape right, we are trying to put num_actions where logits.shape[-1] is
    moa_preds = tf.reshape(  # Reshape to [B,N,A]
        train_batch[MOA_PREDS], [-1, policy.model.num_other_agents, logits.shape[-1]])
    cust_opts = policy.config['model']['custom_options']
    moa_loss = MOALoss(moa_preds, train_batch[OTHERS_ACTIONs],
                       logits.shape[-1], loss_weight=cust_opts["moa_weght"],
                       others_visibility=cust_opts["others_visibility"])

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
        policy.central_value_out,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])

    policy.loss_obj.loss += moa_loss.total_loss
    return policy.loss_obj.loss


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    # Extract matrix of self and other agents' actions.
    own_actions = np.atleast_2d(np.array(sample_batch['actions']))
    own_actions = np.reshape(own_actions, [-1, 1])
    all_actions = extract_last_actions_from_episodes(
        other_agent_batches, own_actions=own_actions, batch_type=True)
    sample_batch['others_actions'] = all_actions

    # TODO(@evinitsky) probably need to add in the MOA predictions

    if policy.train_moa_only_when_visible:
        sample_batch['others_visibility'] = \
            get_agent_visibility_multiplier(sample_batch)

    # Compute causal social influence reward and add to batch.
    sample_batch = compute_influence_reward(policy, sample_batch)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(len(policy.rl_model.state_in)):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        prev_action = sample_batch['prev_actions'][-1]
        prev_reward = sample_batch['prev_rewards'][-1]

        last_r = policy._value(sample_batch["new_obs"][-1],
                               all_actions[-1], prev_action, prev_reward,
                               *next_state)

    sample_batch = compute_advantages(sample_batch, last_r, policy.config["gamma"],
                                      policy.config["lambda"])
    return sample_batch


def compute_influence_reward(policy, trajectory):
    """Compute influence of this agent on other agents and add to rewards.
    """
    # Predict the next action for all other agents. Shape is [B, N, A]
    # TODO(@evinitsky) does this have the right indexing
    true_preds = trajectory[COUNTERFACTUAL_ACTIONS]
    # TODO(@evinitsky) extract the actual logits

    true_probs = trajectory[COUNTERFACTUAL_ACTIONS]

    # Get marginal predictions where effect of self is marginalized out
    (marginal_logits,
     marginal_probs) = marginalize_predictions_over_own_actions(
        trajectory)  # [B, N, A]

    # Compute influence per agent/step ([B, N]) using different metrics
    if policy.influence_divergence_measure == 'kl':
        influence_per_agent_step = kl_div(true_probs, marginal_probs)
    elif policy.influence_divergence_measure == 'jsd':
        mean_probs = 0.5 * (true_probs + marginal_probs)
        influence_per_agent_step = (0.5 * kl_div(true_probs, mean_probs) +
                                    0.5 * kl_div(marginal_probs, mean_probs))
    else:
        sys.exit("Please specify an influence divergence measure from [kl, jsd]")
    # TODO(natashamjaques): more policy comparison functions here.

    # Zero out influence for steps where the other agent isn't visible.
    if policy.influence_only_when_visible:
        if 'others_visibility' in trajectory.keys():
            visibility = trajectory['others_visibility']
        else:
            visibility = get_agent_visibility_multiplier(trajectory, policy.num_other_agents)
        influence_per_agent_step *= visibility

    # Logging influence metrics
    influence_per_agent = np.sum(influence_per_agent_step, axis=0)
    total_influence = np.sum(influence_per_agent_step)

    # Summarize and clip influence reward
    influence = np.sum(influence_per_agent_step, axis=-1)
    influence = np.clip(influence, -policy.influence_reward_clip,
                        policy.influence_reward_clip)

    # Get influence curriculum weight
    # TODO(@evinitsky) move this into a schedule mixin
    policy.steps_processed += len(trajectory['obs'])
    inf_weight = policy.curr_influence_weight

    # Add to trajectory
    trajectory['influence_per_agent'] = influence_per_agent
    trajectory['total_influence'] = total_influence
    trajectory['reward_without_influence'] = trajectory['rewards']
    trajectory['rewards'] = trajectory['rewards'] + (influence * inf_weight)

    return trajectory


def get_agent_visibility_multiplier(trajectory, num_other_agents):
    traj_len = len(trajectory['infos'])
    visibility = np.zeros((traj_len, num_other_agents))
    vis_lists = [info['visible_agents'] for info in trajectory['infos']]
    for i, v in enumerate(vis_lists):
        vis_agents = [agent_name_to_idx(a, self.agent_id) for a in v]
        visibility[i, vis_agents] = 1
    return visibility


def marginalize_predictions_over_own_actions(trajectory):
    # Probability of each action in original trajectory
    action_probs = trajectory[BEHAVIOUR_LOGITS]

    # Normalize to reduce numerical inaccuracies
    action_probs = action_probs / action_probs.sum(axis=1, keepdims=1)

    # Cycle through all possible actions and get predictions for what other
    # agents would do if this action was taken at each trajectory step.

    # TODO(@evinitsky) does this have the right indexing
    counter_preds = trajectory[COUNTERFACTUAL_ACTIONS]
    # TODO(@evinitsky) extract the actual logits

    counter_probs = trajectory[COUNTERFACTUAL_ACTIONS]

    marginal_preds = np.sum(counter_preds, axis=0)
    marginal_probs = np.sum(counter_probs, axis=0)

    # Multiply by probability of each action to renormalize probability
    traj_len = len(trajectory['obs'])
    tiled_probs = np.tile(action_probs, 4),
    import ipdb;
    ipdb.set_trace()
    tiled_probs = np.reshape(
        tiled_probs, [traj_len, self.num_other_agents, self.num_actions])
    marginal_preds = np.multiply(marginal_preds, tiled_probs)
    marginal_probs = np.multiply(marginal_probs, tiled_probs)

    # Normalize to reduce numerical inaccuracies
    marginal_probs = marginal_probs / marginal_probs.sum(axis=2, keepdims=1)

    return marginal_preds, marginal_probs


def extract_last_actions_from_episodes(episodes, batch_type=False,
                                       own_actions=None):
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
        import ipdb;
        ipdb.set_trace()

    # Need to sort agent IDs so same agent is consistently in
    # same part of input space.
    agent_ids = sorted(episodes.keys())
    prev_actions = []

    import ipdb;
    ipdb.set_trace()
    for agent_id in agent_ids:
        if agent_id == self.agent_id:
            continue
        if batch_type:
            prev_actions.append(episodes[agent_id][1]['actions'])
        else:
            prev_actions.append(
                [e.prev_action for e in episodes[agent_id]])

    all_actions = np.transpose(np.array(prev_actions))

    # Attach agents own actions as column 1
    if own_actions is not None:
        all_actions = np.hstack((own_actions, all_actions))

    return all_actions


def extra_fetches(policy):
    """Adds value function, logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        # TODO(@evinitsky) this will not reurn the right thing
        BEHAVIOUR_LOGITS: policy.model.last_output(),
        COUNTERFACTUAL_ACTIONS: policy.model.counterfactual_actions()
    }


class InfluenceScheduleMixIn(object):
    def __init__(self, config):
        cust_opts = config['model']['custom_options']
        self.moa_weight = cust_opts['moa_weight']
        self.train_moa_only_when_visible = cust_opts['train_moa_only_when_visible']
        self.influence_reward_clip = cust_opts['influence_reward_clip']
        self.influence_divergence_measure = cust_opts['influence_divergence_measure']
        self.influence_reward_weight = cust_opts['influence_reward_weight']
        self.influence_curriculum_steps = cust_opts['influence_curriculum_steps']
        self.influence_only_when_visible = cust_opts['influence_only_when_visible']
        self.inf_scale_start = cust_opts['influence_scaledown_start']
        self.inf_scale_end = cust_opts['influence_scaledown_end']
        self.inf_scale_final_val = cust_opts['influence_scaledown_final_val']
        self.steps_processed = 0
        self.curr_influence_weight = self.influence_reward_weight

    def current_influence_curriculum_weight(self):
        """ Computes multiplier for influence reward based on training steps
        taken and curriculum parameters.

        Returns: scalar float influence weight
        """
        if self.steps_processed < self.influence_curriculum_steps:
            percent = float(self.steps_processed) / self.influence_curriculum_steps
            self.curr_influence_weight = percent * self.influence_reward_weight
        elif self.steps_processed > self.inf_scale_start:
            percent = (self.steps_processed - self.inf_scale_start) \
                      / float(self.inf_scale_end - self.inf_scale_start)
            diff = self.influence_reward_weight - self.inf_scale_final_val
            scaled = self.influence_reward_weight - diff * percent
            self.curr_influence_weight = max(self.inf_scale_final_val, scaled)
        else:
            self.curr_influence_weight = self.influence_reward_weight


def extra_stats(policy, train_batch):
    base_stats = kl_and_loss_stats(policy, train_batch)
    import ipdb;
    ipdb.set_trace()
    base_stats["influence_reward"] = None


def build_ppo_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name=POLICY_SCOPE,
        framework="tf")

    return policy.model


CausalMOA_PPOPolicy = build_tf_policy(
    name="PPOTFPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=loss_with_moa,
    make_model=build_ppo_model,
    stats_fn=extra_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches,
    postprocess_fn=postprocess_trajectory,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin  # TODO(@evinitsky) add influence schedule to LearningRateSchedule
    ])

CausalMOATrainer = build_trainer(
    name="CausalMOA",
    default_config=DEFAULT_CONFIG,
    default_policy=CausalMOA_PPOPolicy,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales)
