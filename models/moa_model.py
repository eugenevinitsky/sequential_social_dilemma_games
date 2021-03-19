import sys

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.moa_lstm import MoaLSTM

tf = try_import_tf()


class MOAModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        A model with convolutional layers connected to two distinct sequences of fully connected
        layers. These then each connect to their own respective LSTM, one for an actor-critic policy,
        and one for modeling the actions of other agents (MOA).
        :param obs_space: The agent's observation space.
        :param action_space: The agent's action space.
        :param num_outputs: The amount of actions available to the agent.
        :param model_config: The model config dict. Contains settings dictating layer sizes/amounts,
        amount of other agents, divergence measure used for social influence, and other experiment
        parameters.
        :param name: The model name.
        """
        super(MOAModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space

        # The inputs of the shared trunk. We will concatenate the observation space with
        # shared info about the visibility of agents.
        # Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_other_agents = model_config["custom_options"]["num_other_agents"]
        self.influence_divergence_measure = model_config["custom_options"][
            "influence_divergence_measure"
        ]

        # Declare variables that will later be used as loss fetches
        # It's
        self._model_out = None
        self._value_out = None
        self._action_pred = None
        self._counterfactuals = None
        self._other_agent_actions = None
        self._visibility = None
        self._social_influence_reward = None
        self._true_one_hot_actions = None

        self.moa_encoder_model = self.create_moa_encoder_model(obs_space, model_config)
        self.register_variables(self.moa_encoder_model.variables)
        self.moa_encoder_model.summary()

        # now output two heads, one for action selection and one for the prediction of other agents
        inner_obs_space = self.moa_encoder_model.output_shape[0][-1]

        cell_size = model_config["custom_options"].get("cell_size")
        self.actions_model = ActorCriticLSTM(
            inner_obs_space,
            action_space,
            num_outputs,
            model_config,
            "action_logits",
            cell_size=cell_size,
        )

        # predicts the actions of all the agents besides itself
        # create a new input reader per worker
        self.train_moa_only_when_visible = model_config["custom_options"][
            "train_moa_only_when_visible"
        ]
        self.influence_only_when_visible = model_config["custom_options"][
            "influence_only_when_visible"
        ]
        self.moa_weight = model_config["custom_options"]["moa_loss_weight"]

        self.moa_model = MoaLSTM(
            inner_obs_space,
            action_space,
            self.num_other_agents * num_outputs,
            model_config,
            "moa_model",
            cell_size=cell_size,
        )
        self.register_variables(self.actions_model.rnn_model.variables)
        self.register_variables(self.moa_model.rnn_model.variables)
        self.actions_model.rnn_model.summary()
        self.moa_model.rnn_model.summary()

    @staticmethod
    def create_moa_encoder_model(obs_space, model_config):
        """
        Creates the convolutional part of the MOA model.
        Also casts the input uint8 observations to float32 and normalizes them to the range [0,1].
        :param obs_space: The agent's observation space.
        :param model_config: The config dict containing parameters for the convolution type/shape.
        :return: A new Model object containing the convolution.
        """
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        inputs = tf.keras.layers.Input(original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

        # Build the CNN layer
        conv_out = build_conv_layers(model_config, last_layer)

        # Build Actor-critic FC layers
        actor_critic_fc = build_fc_layers(model_config, conv_out, "policy")

        # Build MOA layers
        moa_fc = build_fc_layers(model_config, conv_out, "moa")

        return tf.keras.Model(inputs, [actor_critic_fc, moa_fc], name="MOA_Encoder_Model")

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        First evaluate non-LSTM parts of model. Then add a time dimension to the batch before
        sending inputs to forward_rnn(), which evaluates the LSTM parts of the model.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The agent's own action logits and the new model state.
        """
        # Evaluate non-lstm layers
        actor_critic_fc_output, moa_fc_output = self.moa_encoder_model(input_dict["obs"]["curr_obs"])

        rnn_input_dict = {
            "ac_trunk": actor_critic_fc_output,
            "prev_moa_trunk": state[5],
            "other_agent_actions": input_dict["obs"]["other_agent_actions"],
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": input_dict["prev_actions"],
        }

        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        output, new_state = self.forward_rnn(rnn_input_dict, state, seq_lens)
        action_logits = tf.reshape(output, [-1, self.num_outputs])
        counterfactuals = tf.reshape(
            self._counterfactuals,
            [-1, self._counterfactuals.shape[-2], self._counterfactuals.shape[-1]],
        )
        new_state.extend([action_logits, moa_fc_output])

        self.compute_influence_reward(input_dict, state[4], counterfactuals)

        return action_logits, new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the MOA LSTMs.
        Implicitly assigns the value function output to self_value_out, and does not return this.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and new LSTM states.
        """
        # Evaluate the actor-critic model
        pass_dict = {"curr_obs": input_dict["ac_trunk"]}
        h1, c1, h2, c2, *_ = state
        (self._model_out, self._value_out, output_h1, output_c1,) = self.actions_model.forward_rnn(
            pass_dict, [h1, c1], seq_lens
        )

        # Evaluate the MOA, and generate counterfactual actions.
        # To do this: cycle through all possible actions and get predictions for what other
        # agents would do if this action was taken at each trajectory step.

        # First we have to evaluate the MOA over the true trajectory to obtain the hidden state
        # that we will use for the next timestep's counterfactual predictions.
        prev_moa_trunk = input_dict["prev_moa_trunk"]
        other_actions = input_dict["other_agent_actions"]
        agent_action = tf.expand_dims(input_dict["prev_actions"], axis=-1)
        all_actions = tf.concat([agent_action, other_actions], axis=-1, name="concat_true_actions")
        self._true_one_hot_actions = self._reshaped_one_hot_actions(all_actions, "forward_one_hot")
        true_action_pass_dict = {
            "curr_obs": prev_moa_trunk,
            "prev_total_actions": self._true_one_hot_actions,
        }

        # Compute the true action prediction, used to determine the MOA loss.
        self._action_pred, output_h2, output_c2 = self.moa_model.forward_rnn(
            true_action_pass_dict, [h2, c2], seq_lens
        )

        # Make counterfactual predictions, used for computing the influence reward.
        counterfactual_preds = []
        for i in range(self.num_outputs):
            # Shape of other_actions is (num_envs, ?, num_other_agents)
            # To add the counterfactual action to it, other_actions can be padded with the constant
            # action value.
            actions_with_counterfactual = tf.pad(
                other_actions, paddings=[[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=i
            )
            one_hot_actions = self._reshaped_one_hot_actions(
                actions_with_counterfactual, "actions_with_counterfactual_one_hot"
            )
            pass_dict = {"curr_obs": prev_moa_trunk, "prev_total_actions": one_hot_actions}
            counterfactual_pred, _, _ = self.moa_model.forward_rnn(pass_dict, [h2, c2], seq_lens)
            counterfactual_preds.append(tf.expand_dims(counterfactual_pred, axis=-2))
        self._counterfactuals = tf.concat(
            counterfactual_preds, axis=-2, name="concat_counterfactuals"
        )

        # TODO(@evinitsky) move this into ppo_moa by using restore_original_dimensions()
        self._other_agent_actions = input_dict["other_agent_actions"]
        self._visibility = input_dict["visible_agents"]

        return self._model_out, [output_h1, output_c1, output_h2, output_c2]

    def compute_influence_reward(self, input_dict, prev_action_logits, counterfactual_logits):
        """
        Compute influence of this agent on other agents.
        :param input_dict: The model input tensors.
        :param prev_action_logits: Logits for the agent's own policy/actions at t-1
        :param counterfactual_logits: The counterfactual action logits for actions made by other
        agents at t.
        """
        # Probability of the next action for all other agents. Shape is [B, N, A].
        # This is the predicted probability given the actions that we DID take.
        # extract out the probability under the actions we actually did take

        # We don't have the current action yet, so the reward for the previous step is calculated.
        # This is corrected for in the function weigh_and_add_influence_reward
        prev_agent_actions = tf.cast(tf.reshape(input_dict["prev_actions"], [-1, 1]), tf.int32)
        # Use the agent's actions as indices to select the predicted logits of other agents for
        # actions that the agent did take, discard the rest.
        predicted_logits = tf.gather_nd(
            params=counterfactual_logits, indices=prev_agent_actions, batch_dims=1
        )

        predicted_logits = tf.reshape(
            predicted_logits, [-1, self.num_other_agents, self.num_outputs]
        )
        predicted_logits = tf.nn.softmax(predicted_logits)
        predicted_logits = predicted_logits / tf.reduce_sum(
            predicted_logits, axis=-1, keepdims=True
        )  # reduce numerical inaccuracies

        # Get marginal predictions where effect of self is marginalized out
        marginal_logits = self.marginalize_predictions_over_own_actions(
            prev_action_logits, counterfactual_logits
        )  # [B, Num agents, Num actions]

        # Compute influence per agent/step ([B, N]) using different metrics
        if self.influence_divergence_measure == "kl":
            influence_reward = self.kl_div(predicted_logits, marginal_logits)
        elif self.influence_divergence_measure == "jsd":
            mean_probs = 0.5 * (predicted_logits + marginal_logits)
            influence_reward = 0.5 * self.kl_div(predicted_logits, mean_probs) + 0.5 * self.kl_div(
                marginal_logits, mean_probs
            )
        else:
            sys.exit("Please specify an influence divergence measure from [kl, jsd]")

        # Zero out influence for steps where the other agent isn't visible.
        if self.influence_only_when_visible:
            visibility = tf.cast(input_dict["obs"]["prev_visible_agents"], tf.float32)
            influence_reward *= visibility
        influence_reward = tf.reduce_sum(influence_reward, axis=-1)
        self._social_influence_reward = influence_reward

    def marginalize_predictions_over_own_actions(self, prev_action_logits, counterfactual_logits):
        """
        Calculates marginal policies for all other agents.
        :param prev_action_logits: The agent's own policy logits at time t-1 .
        :param counterfactual_logits: The counterfactual action predictions made at time t-1 for
        other agents' actions at t.
        :return: The marginal policies for all other agents.
        """
        # Probability of each action in original trajectory
        logits = tf.nn.softmax(prev_action_logits)

        # Normalize to reduce numerical inaccuracies
        logits = logits / tf.reduce_sum(logits, axis=-1, keepdims=True)

        # Indexing is currently [B, Agent actions, num_other_agents * other_agent_logits]
        # Change to [B, Agent actions, num other agents, other agent logits]
        counterfactual_logits = tf.reshape(
            counterfactual_logits, [-1, self.num_outputs, self.num_other_agents, self.num_outputs],
        )

        counterfactual_logits = tf.nn.softmax(counterfactual_logits)
        # Change shape to broadcast probability of each action over counterfactual actions
        logits = tf.reshape(logits, [-1, self.num_outputs, 1, 1])
        normalized_counterfactual_logits = logits * counterfactual_logits
        # Remove counterfactual action dimension
        marginal_probs = tf.reduce_sum(normalized_counterfactual_logits, axis=-3)

        # Normalize to reduce numerical inaccuracies
        marginal_probs = marginal_probs / tf.reduce_sum(marginal_probs, axis=-1, keepdims=True)

        return marginal_probs

    @staticmethod
    def kl_div(x, y):
        """
        Calculate KL divergence between two distributions.
        :param x: A distribution
        :param y: A distribution
        :return: The KL-divergence between x and y. Returns zeros if the KL-divergence contains NaN
        or Infinity.
        """
        dist_x = tf.distributions.Categorical(probs=x)
        dist_y = tf.distributions.Categorical(probs=y)
        result = tf.distributions.kl_divergence(dist_x, dist_y)

        # Don't return nans or infs
        is_finite = tf.reduce_all(tf.is_finite(result))

        def true_fn():
            return result

        def false_fn():
            return tf.zeros(tf.shape(result))

        result = tf.cond(is_finite, true_fn=true_fn, false_fn=false_fn)
        return result

    def _reshaped_one_hot_actions(self, actions_tensor, name):
        """
        Converts the collection of all actions from a number encoding to a one-hot encoding.
        Then, flattens the one-hot encoding so that all concatenated one-hot vectors are the same
        dimension. E.g. with a num_outputs (action_space.n) of 3:
        _reshaped_one_hot_actions([0,1,2]) returns [1,0,0,0,1,0,0,0,1]
        :param actions_tensor: The tensor containing actions.
        :return: Tensor containing one-hot reshaped action values.
        """
        one_hot_actions = tf.keras.backend.one_hot(actions_tensor, self.num_outputs)
        # Extract partially known tensor shape and combine with actions_layer known shape
        # This combination is a bit contrived for a reason: the shape cannot be determined otherwise
        batch_time_dims = [
            tf.shape(one_hot_actions)[k] for k in range(one_hot_actions.shape.rank - 2)
        ]
        reshape_dims = batch_time_dims + [actions_tensor.shape[-1] * self.num_outputs]
        reshaped = tf.reshape(one_hot_actions, shape=reshape_dims, name=name)
        return reshaped

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def counterfactual_actions(self):
        return self._counterfactuals

    def action_logits(self):
        return self._model_out

    def social_influence_reward(self):
        return self._social_influence_reward

    def predicted_actions(self):
        """ :returns Predicted actions. NB: Since the agent's own true action is not known when
         evaluating this model, the timestep is off by one (too late). Thus, for any index n > 0,
         the value at n is a prediction made at n-1, about the actions taken at n.
         predicted_actions[0] contains no sensible value, as this would have to be a prediction made
         at timestep -1, but we start time at 0."""
        return self._action_pred

    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        return self.actions_model.get_initial_state() + self.moa_model.get_initial_state()
