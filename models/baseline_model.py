from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM
from models.common_layers import build_conv_layers, build_fc_layers

tf = try_import_tf()


class BaselineModel(RecurrentTFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        The baseline model without social influence from the social influence paper.
        :param obs_space: The observation space shape.
        :param action_space: The amount of available actions to this agent.
        :param num_outputs: The amount of available actions to this agent.
        :param model_config: The model config dict. Used to determine size of conv and fc layers.
        :param name: The model name.
        """
        super(BaselineModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space
        self.num_outputs = num_outputs

        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # Determine vision network input shape: add an extra none for the time dimension
        inputs = tf.keras.layers.Input(shape=original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

        # Build the CNN layers
        last_layer = build_conv_layers(model_config, last_layer)

        # Add the fully connected layers
        last_layer = build_fc_layers(model_config, last_layer, name)

        self.encoder_model = tf.keras.Model(inputs, [last_layer], name="Baseline_Encoder_Model")
        self.register_variables(self.encoder_model.variables)
        self.encoder_model.summary()

        # Action selection/value function
        cell_size = model_config["custom_options"].get("cell_size")
        self.policy_model = ActorCriticLSTM(
            last_layer.shape[-1],
            action_space,
            num_outputs,
            model_config,
            "policy",
            cell_size=cell_size,
        )

        self.register_variables(self.policy_model.rnn_model.variables)
        self.policy_model.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Evaluate the model.
        Adds time dimension to batch before sending inputs to forward_rnn()
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and state.
        """
        trunk = self.encoder_model(input_dict["obs"]["curr_obs"])
        new_dict = {"curr_obs": add_time_dimension(trunk, seq_lens)}

        output, new_state = self.forward_rnn(new_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the LSTM.
        Implicitly assigns the value function output to self_value_out, and does not return this.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The policy logits and new state.
        """
        h1, c1 = state

        # Compute the next action
        (self._model_out, self._value_out, output_h1, output_c1,) = self.policy_model.forward_rnn(
            input_dict, [h1, c1], seq_lens
        )

        return self._model_out, [output_h1, output_c1]

    def action_logits(self):
        """
        :return: The action logits from the latest forward pass.
        """
        return self._model_out

    def value_function(self):
        """
        :return: The value function result from the latest forward pass.
        """
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        """
        :return: Initial state of this model. This model only has LSTM state from the policy_model.
        """
        return self.policy_model.get_initial_state()
