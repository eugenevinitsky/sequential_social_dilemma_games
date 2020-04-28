from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers
from models.moa_lstm import MoaLSTM

tf = try_import_tf()


class MOAModel(RecurrentTFModelV2):
    """An model with convolutional layers connected to two distinct sequences of fully connected
    layers. These then connect to a respective LSTM, one for an actor-critic policy, and one for
    modeling the actions of other agents (MOA)"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MOAModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space

        # The inputs of the shared trunk. We will concatenate the observation space with
        # shared info about the visibility of agents.
        # Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_other_agents = model_config["custom_options"]["num_other_agents"]

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

    def create_moa_encoder_model(self, obs_space, model_config):
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        inputs = tf.keras.layers.Input(original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)
        self.preprocessed_input = last_layer

        # Build the CNN layer
        conv_out = build_conv_layers(model_config, last_layer)

        # Build Actor-critic FC layers
        actor_critic_fc = build_fc_layers(model_config, conv_out, "policy")

        # Build MOA layers
        moa_fc = build_fc_layers(model_config, conv_out, "moa")

        return tf.keras.Model(inputs, [actor_critic_fc, moa_fc])

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """ First evaluate non-lstm parts of model. Then add a time dimension to the batch before
         sending inputs to forward_rnn()"""
        # Evaluate non-lstm layers
        actor_critic_fc_output, moa_fc_output = self.moa_encoder_model(input_dict["obs"]["curr_obs"])

        rnn_input_dict = {
            "ac_trunk": actor_critic_fc_output,
            "moa_trunk": moa_fc_output,
            "other_agent_actions": input_dict["obs"]["other_agent_actions"],
            "visible_agents": input_dict["obs"]["visible_agents"],
            "prev_actions": tf.cast(input_dict["prev_actions"], dtype=tf.uint8),
        }

        # Add time dimension to rnn inputs
        for k, v in rnn_input_dict.items():
            rnn_input_dict[k] = add_time_dimension(v, seq_lens)

        output, new_state = self.forward_rnn(rnn_input_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        # Evaluate the actor-critic model
        pass_dict = {"curr_obs": input_dict["ac_trunk"]}
        h1, c1, h2, c2 = state
        (self._model_out, self._value_out, output_h1, output_c1,) = self.actions_model.forward_rnn(
            pass_dict, [h1, c1], seq_lens
        )

        # Evaluate the MOA, and generate counterfactual actions.
        # To do this: cycle through all possible actions and get predictions for what other
        # agents would do if this action was taken at each trajectory step.

        # First we have to compute it over the trajectory to give us the hidden state
        # that we will actually use
        other_actions = input_dict["other_agent_actions"]
        agent_action = tf.expand_dims(input_dict["prev_actions"], axis=-1)
        all_actions = tf.concat([agent_action, other_actions], axis=-1, name="concat_true_actions")
        self._true_one_hot_actions = self._reshaped_one_hot_actions(all_actions, "forward_one_hot")
        true_action_pass_dict = {
            "curr_obs": input_dict["moa_trunk"],
            "prev_total_actions": self._true_one_hot_actions,
        }

        # Compute the action prediction. This is unused in the actual rollout and is only to generate
        # a series of hidden states for the counterfactuals
        action_pred, output_h2, output_c2 = self.moa_model.forward_rnn(
            true_action_pass_dict, [h2, c2], seq_lens
        )

        # Now we can use that cell state to do the counterfactual predictions
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
            pass_dict = {"curr_obs": input_dict["moa_trunk"], "prev_total_actions": one_hot_actions}
            counterfactual_pred, _, _ = self.moa_model.forward_rnn(pass_dict, [h2, c2], seq_lens)
            counterfactual_preds.append(tf.expand_dims(counterfactual_pred, axis=-2))
        self._counterfactual_preds = tf.concat(
            counterfactual_preds, axis=-2, name="concat_counterfactuals"
        )

        # TODO(@evinitsky) move this into ppo_moa by using restore_original_dimensions()
        self._other_agent_actions = input_dict["other_agent_actions"]
        self._visibility = input_dict["visible_agents"]

        return self._model_out, [output_h1, output_c1, output_h2, output_c2]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def counterfactual_actions(self):
        return self._counterfactual_preds

    def moa_preds_from_batch(self, train_batch):
        """Convenience function that calls this model with a tensor batch.

        What this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.

        This is used when setting up the loss function.
        """

        obs_dict = restore_original_dimensions(train_batch["obs"], self.obs_space)
        curr_obs = obs_dict["curr_obs"]
        seq_lens = train_batch.get("seq_lens")
        ac_trunk, moa_trunk = self.moa_encoder_model(curr_obs)

        # Concat the agent actions together
        other_agent_actions = obs_dict["other_agent_actions"]
        agent_actions = tf.expand_dims(
            tf.cast(train_batch[SampleBatch.PREV_ACTIONS], tf.uint8), axis=1
        )
        prev_total_actions = tf.concat([agent_actions, other_agent_actions], axis=-1)
        prev_total_actions = self._reshaped_one_hot_actions(prev_total_actions, "loss_one_hot")

        # Now we add the appropriate time dimension
        prev_total_actions = add_time_dimension(prev_total_actions, seq_lens)

        input_dict = {
            "curr_obs": add_time_dimension(moa_trunk, seq_lens),
            "is_training": True,
            "prev_total_actions": prev_total_actions,
        }
        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []

        # TODO(@evinitsky) remove the magic number
        i = 2
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1

        moa_preds, _, _ = self.moa_model.forward_rnn(input_dict, states, seq_lens)
        return moa_preds

    def _reshaped_one_hot_actions(self, actions_layer, name):
        """
        Converts the collection of all actions from a number encoding to a one-hot encoding.
        Then, flattens the one-hot encoding so that all one-hot vectors are in the same dimension.
        E.g. with a num_outputs (action_space.n) of 3:
        _reshaped_one_hot_actions([0,1,2]) returns [1,0,0,0,1,0,0,0,1]
        :param actions_layer: The tensor containing actions.
        :return: Tensor containing one-hot reshaped action values.
        """
        one_hot_actions = tf.keras.backend.one_hot(actions_layer, self.num_outputs)
        # Extract partially known tensor shape and combine with actions_layer known shape
        # This combination is a bit contrived for a reason: the shape cannot be determined otherwise
        batch_time_dims = [
            tf.shape(one_hot_actions)[k] for k in range(one_hot_actions.shape.rank - 2)
        ]
        reshape_dims = batch_time_dims + [actions_layer.shape[-1] * self.num_outputs]
        reshaped = tf.reshape(one_hot_actions, shape=reshape_dims, name=name)
        return reshaped

    def action_logits(self):
        return self._model_out

    # TODO(@evinitsky) pull out the time slice
    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        return self.actions_model.get_initial_state() + self.moa_model.get_initial_state()
