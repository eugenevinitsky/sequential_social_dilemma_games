import numpy as np
from gym.spaces import Box
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.utils import try_import_tf

from models.moa_model import MOA_LSTM

tf = try_import_tf()


class SocialCuriosityModule(MOA_LSTM):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(SocialCuriosityModule, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self._previous_encoded_state = None
        self._previous_actions = None
        self._previous_lstm_output = None

        self.scm_encoder_model = self.create_scm_encoder_model(model_config)
        encoded_space = Box(
            low=-1, high=1, shape=self.scm_encoder_model.output.shape[2:], dtype=np.float32
        )

        self.forward_model = self.create_forward_model(model_config, encoded_space)
        self.inverse_model = self.create_inverse_model(model_config, encoded_space)

        for model in [self.scm_encoder_model, self.forward_model, self.inverse_model]:
            self.register_variables(model.variables)
            model.summary()

        self.scm_loss_weight = model_config["custom_options"]["scm_loss_weight"]

    def create_scm_encoder_model(self, model_config):
        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        out_size, kernel, stride = filters[-1]
        conv_out = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            name="conv_scm_encoder",
        )(self.postprocessed_input)

        return tf.keras.Model(self.postprocessed_input, conv_out)

    # Inputs: [Encoded state at t, Actions at t, LSTM output at t, Social influence at t]
    # Output: Predicted encoded state at t+1
    def create_forward_model(self, model_config, input_space):
        output_size = int(input_space.shape[0] * input_space.shape[1] * input_space.shape[2])
        input_layer = tf.keras.layers.Input(shape=input_space, name="forward_input")
        influence_reward_input = tf.keras.layers.Input(shape=(-1, 1), name="influence_reward_input")
        input_layer_full = tf.keras.layers.concatenate(
            [
                input_layer,
                self.moa_model.actions_layer,
                self.moa_model.lstm_out,
                influence_reward_input,
            ]
        )
        activation = get_activation_fn(model_config.get("fcnet_activation"))

        fc_layer = tf.keras.layers.Dense(
            32, name="fc_forward", activation=activation, kernel_initializer=normc_initializer(1.0),
        )(input_layer_full)

        output_layer = tf.keras.layers.Dense(
            output_size, activation="relu", kernel_initializer=normc_initializer(1.0),
        )(fc_layer)

        return tf.keras.Model(input_layer, output_layer)

    # Inputs: [Encoded state at t, Encoded state at t - 1, Actions at t - 1, LSTM output at t - 1]
    # Output: Social influence at t - 1
    def create_inverse_model(self, model_config, input_space):
        input_layer = tf.keras.layers.Input(shape=input_space, name="inverse_input")
        input_with_previous_timestep = tf.keras.layers.concatenate(input_layer)

        input_layer_full = tf.keras.layers.concatenate(
            [input_with_previous_timestep, self.moa_model.actions_layer, self.moa_model.lstm_out]
        )
        activation = get_activation_fn(model_config.get("fcnet_activation"))

        fc_layer = tf.keras.layers.Dense(
            32, name="fc_forward", activation=activation, kernel_initializer=normc_initializer(1.0),
        )(input_layer_full)

        output_layer = tf.keras.layers.Dense(
            1, activation="relu", kernel_initializer=normc_initializer(1.0),
        )(fc_layer)

        return tf.keras.Model(input_layer, output_layer)

    def forward(self, input_dict, state, seq_lens):
        output, new_state = super(SocialCuriosityModule, self).forward(input_dict, state, seq_lens)

        encoder_input = {}

        # Inputs:
        forward_model_input = {
            # Encoded state at t
            "encoded_state": self.scm_encoder_model(input_dict["obs"]["curr_obs"]),
            # Actions at t
            "actions": self._true_one_hot_actions,
            # LSTM output at t
            "lstm_output": self.moa_model.lstm_out,
            # Social influence at t
            "social_influence": input_dict["total_influence_reward"],
        }

        inverse_model_input = {
            # Encoded state at t - 1
            "previous_encoded_state": self._previous_encoded_state,
            # Encoded state at t
            "encoded_state": self.scm_encoder_model(input_dict["obs"]["curr_obs"]),
            # Actions at t - 1
            "actions": self._previous_actions,
            # LSTM output at t - 1
            "lstm_output": self._previous_lstm_output,
        }

        # Outputs:
        self._encoder_output = self.scm_encoder_model(encoder_input)

        # Outputs predicted encoded state at t+1
        self._forward_model_output = self.forward_model(forward_model_input)

        # Outputs predicted social influence reward
        self._inverse_model_output = self.inverse_model(inverse_model_input)

        return output, new_state

    def true_encoded_observations(self):
        return self._encoder_output

    def predicted_encoded_observations(self):
        return self._forward_model_output

    def predicted_influence(self):
        return self._inverse_model_output
