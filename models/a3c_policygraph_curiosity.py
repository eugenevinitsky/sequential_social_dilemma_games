"""Note: Keep in sync with changes to VTracePolicyGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import gym

import ray
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.evaluation.policy_graph import PolicyGraph
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph, \
    LearningRateSchedule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override


def agent_name_to_idx(name):
    agent_num = int(name[6])
    return agent_num


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
        tf.print("Curiosity loss", self.total_loss, [self.total_loss])


class A3CLoss(object):
    def __init__(self,
                 action_dist,
                 actions,
                 advantages,
                 v_target,
                 vf,
                 vf_loss_coeff=0.5,
                 entropy_coeff=-0.01):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff +
                           self.entropy * entropy_coeff)


class A3CPolicyGraph(LearningRateSchedule, TFPolicyGraph):
    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG, **config)
        self.config = config
        self.sess = tf.get_default_session()

        self.num_other_agents = config['num_other_agents']
        self.agent_id = config['agent_id']

        # Read curiosity options from config
        cust_opts = config['model']['custom_options']
        self.aux_loss_weight = cust_opts['aux_loss_weight']
        self.aux_reward_clip = cust_opts['aux_reward_clip']
        self.aux_reward_weight = cust_opts['aux_reward_weight']
        self.aux_curriculum_steps = cust_opts['aux_curriculum_steps']
        self.aux_scale_start = cust_opts['aux_scaledown_start']
        self.aux_scale_end = cust_opts['aux_scaledown_end']
        self.aux_scale_final_val = cust_opts['aux_scaledown_final_val']

        # Use to compute aux curriculum weight
        self.steps_processed = 0

        # Compute output size of aux model
        self.encoded_dim_size = observation_space.shape[0] * \
                                observation_space.shape[1] * \
                                self.config['model']['conv_filters']

        # Setup the policy
        self.observations = tf.placeholder(tf.float32,
                                           [None] + list(observation_space.shape))

        # Add other agents actions placeholder for MOA preds
        # Add 1 to include own action so it can be conditioned on. Note: agent's
        # own actions will always form the first column of this tensor.
        self.others_actions = tf.placeholder(tf.int32,
                                             shape=(None, self.num_other_agents + 1),
                                             name="others_actions")

        dist_class, self.num_actions = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        prev_actions = ModelCatalog.get_action_placeholder(action_space)
        prev_rewards = tf.placeholder(tf.float32, [None], name="prev_reward")

        # We now create three models, one for the policy, one auxiliary task model, and an encoded state model
        self.rl_model, self.aux_model, self.encoder_model = ModelCatalog.get_double_fc_lstm_model({
                "obs": self.observations,
                "prev_actions": prev_actions,
                "prev_rewards": prev_rewards,
                "is_training": self._get_is_training_placeholder()},
            encoded_dim_size=self.encoded_dim_size,
            obs_space=observation_space,
            num_outputs_lstm1=self.num_actions,
            num_outputs_lstm2=self.encoded_dim_size,
            options=self.config["model"],
            lstm1_name="policy",
            lstm2_name="aux_task")

        action_dist = dist_class(self.rl_model.outputs)
        self.action_probs = tf.nn.softmax(self.rl_model.outputs)
        self.vf = self.rl_model.value_function()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)
        self.encoded_observations = self.encoder_model.outputs
        self.predicted_observations = self.aux_model.outputs

        # Setup the policy loss
        if isinstance(action_space, gym.spaces.Box):
            ac_size = action_space.shape[0]
            actions = tf.placeholder(tf.float32, [None, ac_size], name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            actions = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise UnsupportedSpaceException("Action space {} is not supported for A3C.".format(action_space))
        advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.v_target = tf.placeholder(tf.float32, [None], name="v_target")
        self.rl_loss = A3CLoss(action_dist, actions, advantages,
                               self.v_target,
                               self.vf,
                               self.config["vf_loss_coeff"],
                               self.config["entropy_coeff"])

        # Setup the aux task loss
        self.aux_loss = CuriosityLoss(pred_states=self.aux_model.outputs,
                                      true_states=self.encoder_model.outputs,
                                      loss_weight=self.aux_loss_weight)

        # Total loss
        self.total_loss = self.rl_loss.total_loss + self.aux_loss.total_loss

        # Initialize TFPolicyGraph
        loss_in = [
            ("obs", self.observations),
            ("actions", actions),
            ("prev_actions", prev_actions),
            ("prev_rewards", prev_rewards),
            ("advantages", advantages),
            ("value_targets", self.v_target),
        ]
        LearningRateSchedule.__init__(self, self.config["lr"],
                                      self.config["lr_schedule"])
        TFPolicyGraph.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=self.observations,
            action_sampler=action_dist.sample(),
            action_prob=action_dist.sampled_action_prob(),
            loss=self.total_loss,
            model=self.rl_model,
            loss_inputs=loss_in,
            state_inputs=self.rl_model.state_in + self.aux_model.state_in,
            state_outputs=self.rl_model.state_out + self.aux_model.state_out,
            prev_action_input=prev_actions,
            prev_reward_input=prev_rewards,
            seq_lens=self.rl_model.seq_lens,
            max_seq_len=self.config["model"]["max_seq_len"])

        self.total_aux_reward = tf.get_variable("total_aux_reward", initializer=tf.constant(0.0))

        self.stats = {
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "policy_loss": self.rl_loss.pi_loss,
            "policy_entropy": self.rl_loss.entropy,
            "grad_gnorm": tf.global_norm(self._grads),
            "var_gnorm": tf.global_norm(self.var_list),
            "vf_loss": self.rl_loss.vf_loss,
            "vf_explained_var": explained_variance(self.v_target, self.vf),
            "total_a3c_loss": self.rl_loss.total_loss,
            "aux_loss": self.aux_loss.total_loss,
            "total_aux_reward": self.total_aux_reward
        }

        self.sess.run(tf.global_variables_initializer())

    @override(TFPolicyGraph)
    def copy(self, existing_inputs):
        # Optional, implement to work with the multi-GPU optimizer.
        raise NotImplementedError

    @override(PolicyGraph)
    def get_initial_state(self):
        return self.rl_model.state_init + self.aux_model.state_init

    @override(TFPolicyGraph)
    def _build_compute_actions(self,
                               builder,
                               obs_batch,
                               state_batches=None,
                               prev_action_batch=None,
                               prev_reward_batch=None,
                               episodes=None):
        state_batches = state_batches or []
        if len(self._state_inputs) != len(state_batches):
            raise ValueError(
                "Must pass in RNN state batches for placeholders {}, got {}".
                format(self._state_inputs, state_batches))
        builder.add_feed_dict(self.extra_compute_action_feed_dict())

        # Extract matrix of other agents' past actions, including agent's own
        if type(episodes) == dict and 'all_agents_actions' in episodes.keys():
            # Call from visualizer_rllib, change episodes format so it complies with the default format.
            self_index = agent_name_to_idx(self.agent_id)
            # First get own action
            all_actions = [episodes['all_agents_actions'][self_index]]
            others_actions = [e for i, e in enumerate(
                episodes['all_agents_actions']) if self_index != i]
            all_actions.extend(others_actions)
            all_actions = np.reshape(np.array(all_actions), [1, -1])
        else:
            own_actions = np.atleast_2d(np.array(
                [e.prev_action for e in episodes[self.agent_id]]))
            all_actions = self.extract_last_actions_from_episodes(
                episodes, own_actions=own_actions)

        builder.add_feed_dict({self._obs_input: obs_batch,
                               self.others_actions: all_actions})

        if state_batches:
            seq_lens = np.ones(len(obs_batch))
            builder.add_feed_dict({self._seq_lens: seq_lens,
                                   self.aux_model.seq_lens: seq_lens})
        if self._prev_action_input is not None and prev_action_batch:
            builder.add_feed_dict({self._prev_action_input: prev_action_batch})
        if self._prev_reward_input is not None and prev_reward_batch:
            builder.add_feed_dict({self._prev_reward_input: prev_reward_batch})

        builder.add_feed_dict({self._is_training: False})
        builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
        fetches = builder.add_fetches([self._sampler] + self._state_outputs +
                                      [self.extra_compute_action_fetches()])

        return fetches[0], fetches[1:-1], fetches[-1]

    def _get_loss_inputs_dict(self, batch):
        # Override parent function to add seq_lens to tensor for additional LSTM
        loss_inputs = super(A3CPolicyGraph, self)._get_loss_inputs_dict(batch)
        loss_inputs[self.aux_model.seq_lens] = loss_inputs[self._seq_lens]
        return loss_inputs

    @override(TFPolicyGraph)
    def gradients(self, optimizer):
        grads = tf.gradients(self._loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        clipped_grads = list(zip(self.grads, self.var_list))
        return clipped_grads

    @override(TFPolicyGraph)
    def extra_compute_grad_fetches(self):
        """Extra values to fetch and return from compute_gradients()."""
        return {
            "stats": self.stats,
        }

    @override(TFPolicyGraph)
    def extra_compute_action_fetches(self):
        """Extra values to fetch and return from compute_actions().

        By default we only return action probability info (if present).
        """
        return dict(
            TFPolicyGraph.extra_compute_action_fetches(self),
            **{"vf_preds": self.vf,
               "encoded_observations": self.encoded_observations,
               "predicted_observations": self.predicted_observations})

    def _value(self, ob, others_actions, prev_action, prev_reward, *args):
        """Compute the value function output for a single observation
        """
        feed_dict = {self.observations: [ob],
                     self.others_actions: [others_actions],
                     self.rl_model.seq_lens: [1],
                     self._prev_action_input: [prev_action],
                     self._prev_reward_input: [prev_reward]}
        assert len(args) == len(self.rl_model.state_in), \
            (args, self.rl_model.state_in)
        for k, v in zip(self.rl_model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.vf, feed_dict)
        return vf[0]

    @override(PolicyGraph)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        # Extract matrix of self and other agents' actions.
        own_actions = np.atleast_2d(np.array(sample_batch['actions']))
        own_actions = np.reshape(own_actions, [-1, 1])
        all_actions = self.extract_last_actions_from_episodes(
            other_agent_batches, own_actions=own_actions, batch_type=True)
        sample_batch['others_actions'] = all_actions

        # Compute auxiliary reward and add to batch.
        sample_batch = self.compute_auxiliary_reward(sample_batch)

        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0
        else:
            next_state = []
            for i in range(len(self.rl_model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            prev_action = sample_batch['prev_actions'][-1]
            prev_reward = sample_batch['prev_rewards'][-1]

            last_r = self._value(sample_batch["new_obs"][-1],
                                 all_actions[-1], prev_action, prev_reward,
                                 *next_state)

        sample_batch = compute_advantages(sample_batch, last_r, self.config["gamma"],
                                          self.config["lambda"])
        return sample_batch

    def compute_auxiliary_reward(self, trajectory):
        """Compute auxiliary reward of this agent
        """
        # Logging auxiliary metrics
        aux_reward_per_agent_step = [calculate_surprisal(trajectory['predicted_observations'][i],
                                                         trajectory['encoded_observations'][i])
                                     for i in range(len(trajectory['obs']))]

        total_aux_reward = np.sum(aux_reward_per_agent_step)
        self.total_aux_reward.load(total_aux_reward, session=self.sess)

        # Clip auxiliary reward
        aux_reward_per_agent_step = np.clip(aux_reward_per_agent_step,
                                            -self.aux_reward_clip,
                                             self.aux_reward_clip)

        # Add to trajectory
        trajectory['rewards'] = trajectory['rewards'] + aux_reward_per_agent_step * self.aux_reward_weight

        return trajectory

    def extract_last_actions_from_episodes(self, episodes, batch_type=False,
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
            import pdb
            pdb.set_trace()

        # Need to sort agent IDs so same agent is consistently in
        # same part of input space.
        agent_ids = sorted(episodes.keys())
        prev_actions = []

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
            if len(all_actions) is not 0:
                all_actions = np.hstack((own_actions, all_actions))
            else:
                all_actions = own_actions

        return all_actions
