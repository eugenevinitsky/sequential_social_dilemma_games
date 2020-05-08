import numpy as np
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.utils import override, try_import_tf

from models.moa_model import MOAModel

tf = try_import_tf()


class SocialCuriosityModule(MOAModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(SocialCuriosityModule, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self._encoded_state = None
        self._current_lstm_hidden_state = None
        self._forward_model_output = None
        self._inverse_model_output = None
        self._social_curiosity_reward = None
        self._inverse_model_loss = None

        self.scm_encoder_model = self.create_scm_encoder_model(obs_space, model_config)
        self.forward_model = self.create_forward_model(model_config, self.scm_encoder_model)
        self.inverse_model = self.create_inverse_model(model_config, self.scm_encoder_model)

        for model in [self.scm_encoder_model, self.forward_model, self.inverse_model]:
            self.register_variables(model.variables)
            model.summary()

        self.scm_loss_weight = model_config["custom_options"]["scm_loss_weight"]

    @staticmethod
    def create_scm_encoder_model(obs_space, model_config):
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        input_layer = tf.keras.layers.Input(original_obs_dims, name="observations", dtype=tf.uint8)

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(input_layer, tf.float32)
        last_layer = tf.math.divide(last_layer, 255.0)

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
        )(last_layer)
        flattened_conv_out = tf.keras.layers.Flatten()(conv_out)

        return tf.keras.Model(input_layer, flattened_conv_out, name="SCM_Encoder_Model")

    # Inputs: [Encoded state at t,
    #          Actions at t,
    #          LSTM output at t,
    #          Social influence at t]
    # Output: Predicted encoded state at t+1
    def create_forward_model(self, model_config, encoder):
        encoder_output_size = encoder.output_shape[-1]
        inputs = [
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_now"),
            self.create_action_input_layer(),
            self.create_lstm_input_layer(model_config),
            tf.keras.layers.Input(shape=1, name="influence_reward_input"),
        ]
        inputs_concatenated = tf.keras.layers.concatenate(inputs)
        activation = get_activation_fn(model_config.get("fcnet_activation"))

        fc_layer = tf.keras.layers.Dense(
            32, name="fc_forward", activation=activation, kernel_initializer=normc_initializer(1.0),
        )(inputs_concatenated)

        output_layer = tf.keras.layers.Dense(
            encoder_output_size, activation="relu", kernel_initializer=normc_initializer(1.0),
        )(fc_layer)

        return tf.keras.Model(inputs, output_layer, name="SCM_Forward_Model")

    # Inputs:[Encoded state at t + 1,
    #         Encoded state at t,
    #         Actions at t,
    #         MOA LSTM output at t]
    # Output: Predicted social influence at t
    # Note that this is different from the paper: we can only work with historical values, so the
    # results of this model are "behind" by 1 timestep, which is corrected for in the loss function.
    def create_inverse_model(self, model_config, encoder):
        encoder_output_size = encoder.output_shape[-1]
        inputs = [
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_now"),
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_next"),
            self.create_action_input_layer(),
            self.create_lstm_input_layer(model_config),
        ]
        inputs_concatenated = tf.keras.layers.concatenate(inputs)
        activation = get_activation_fn(model_config.get("fcnet_activation"))

        fc_layer = tf.keras.layers.Dense(
            32, name="fc_forward", activation=activation, kernel_initializer=normc_initializer(1.0),
        )(inputs_concatenated)

        output_layer = tf.keras.layers.Dense(
            1, activation="relu", kernel_initializer=normc_initializer(1.0),
        )(fc_layer)

        return tf.keras.Model(inputs, output_layer, name="SCM_Inverse_Model")

    @staticmethod
    def create_encoded_input_layer(encoded_input_shape, name):
        return tf.keras.layers.Input(shape=encoded_input_shape, name=name)

    @staticmethod
    def create_lstm_input_layer(model_config):
        cell_size = model_config["custom_options"].get("cell_size")
        return tf.keras.layers.Input(shape=cell_size, name="lstm_input")

    def create_action_input_layer(self):
        return tf.keras.layers.Input(
            shape=(self.action_space.n * (self.num_other_agents + 1)), name="action_input"
        )

    def forward(self, input_dict, state, seq_lens):
        output, new_state = super(SocialCuriosityModule, self).forward(input_dict, state, seq_lens)

        self._encoded_state = self.scm_encoder_model(input_dict["obs"]["curr_obs"])
        new_state.append(self._encoded_state)

        one_hot_actions = tf.reshape(
            self._true_one_hot_actions, shape=[-1, self._true_one_hot_actions.shape[-1]]
        )

        influence_reward = tf.expand_dims(self._social_influence_reward, axis=-1)

        forward_model_input = {
            # Encoded state at t
            "encoded_input_now": state[6],
            # Social influence at t
            "influence_reward_input": influence_reward,
            # Actions at t
            "action_input": one_hot_actions,
            # MOA LSTM output at t
            "lstm_input": state[2],
        }

        inverse_model_input = {
            # Encoded state at t
            "encoded_input_now": state[6],
            # Encoded state at t + 1
            "encoded_input_next": self._encoded_state,
            # Actions at t
            "action_input": one_hot_actions,
            # MOA LSTM output at t
            "lstm_input": state[2],
        }

        self._forward_model_output = self.forward_model(forward_model_input)
        self._inverse_model_output = self.inverse_model(inverse_model_input)

        curiosity_reward = self.compute_curiosity_reward(
            self._encoded_state, self._forward_model_output
        )
        curiosity_reward = tf.reduce_sum(curiosity_reward, axis=-1)
        self._social_curiosity_reward = curiosity_reward

        self._inverse_model_loss = self.compute_inverse_model_loss(
            influence_reward, self._inverse_model_output
        )

        return output, new_state

    def compute_curiosity_reward(self, true_encoded_state, predicted_encoded_state):
        mse = self.batched_mse(true_encoded_state, predicted_encoded_state)
        div_mse = tf.multiply(mse, 0.5, name="mult_mse")
        return div_mse

    def compute_inverse_model_loss(self, true_influence_reward, predicted_influence_reward):
        return self.batched_mse(true_influence_reward, predicted_influence_reward)

    @staticmethod
    def batched_mse(true_tensor, pred_tensor):
        squared_difference = tf.squared_difference(true_tensor, pred_tensor)
        mse = tf.reduce_mean(squared_difference, axis=-1, keepdims=True)
        return mse

    def true_encoded_observations(self):
        return self._encoded_state

    def forward_model_output(self):
        return self._forward_model_output

    def inverse_model_output(self):
        return self._inverse_model_output

    def social_curiosity_reward(self):
        return self._social_curiosity_reward

    def inverse_model_loss(self):
        return self._inverse_model_loss

    @override(MOAModel)
    def get_initial_state(self):
        moa_initial_state = super(SocialCuriosityModule, self).get_initial_state()
        return moa_initial_state + [np.zeros(self.scm_encoder_model.output_shape[-1], np.float32)]
