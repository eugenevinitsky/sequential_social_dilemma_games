import numpy as np
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import override, try_import_tf
from ray.rllib.utils.framework import get_activation_fn

from models.moa_model import MOAModel

tf = try_import_tf()


class SocialCuriosityModule(MOAModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        An extension of the MOA, including a forward and inverse model that together create a
        "social curiosity reward".
        :param obs_space: The agent's observation space.
        :param action_space: The agent's action space.
        :param num_outputs: The amount of actions available to the agent.
        :param model_config: The model config dict.
        :param name: The model name.
        """
        super(SocialCuriosityModule, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
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
        """
        Create the encoder submodel, which is part of the SCM.
        :param obs_space: A single agent's observation space.
        :param model_config: The model config dict.
        :return: A new encoder model.
        """
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

    def create_forward_model(self, model_config, encoder):
        """
        Create the forward submodel of the SCM.
        Inputs: [Encoded state at t - 1,
                 Actions at t - 1,
                 LSTM output at t - 1,
                 Social influence at t - 1]
        Output: Predicted encoded state at t
        :param model_config: The model config dict.
        :param encoder: The SCM encoder submodel.
        :return: A new forward model.
        """
        encoder_output_size = encoder.output_shape[-1]
        inputs = [
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_now"),
            self.create_action_input_layer(self.action_space.n, self.num_other_agents + 1),
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

    def create_inverse_model(self, model_config, encoder):
        """
        Create the inverse submodel of the SCM.
        Inputs:[Encoded state at t,
                Encoded state at t - 1,
                Actions at t - 1,
                MOA LSTM output at t - 1]
        Output: Predicted social influence reward at t - 1
        :param model_config: The model config dict.
        :param encoder: The SCM encoder submodel.
        :return: A new inverse model.
        """
        encoder_output_size = encoder.output_shape[-1]
        inputs = [
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_now"),
            self.create_encoded_input_layer(encoder_output_size, "encoded_input_next"),
            self.create_action_input_layer(self.action_space.n, self.num_other_agents + 1),
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

    @staticmethod
    def create_action_input_layer(action_space_size, num_agents):
        return tf.keras.layers.Input(shape=(action_space_size * num_agents), name="action_input")

    def forward(self, input_dict, state, seq_lens):
        """
        The forward pass through the SCM network.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: The LSTM sequence lengths.
        :return: The SCM output and new model state.
        """
        output, new_state = super(SocialCuriosityModule, self).forward(input_dict, state, seq_lens)

        encoded_state = self.scm_encoder_model(input_dict["obs"]["curr_obs"])
        new_state.append(encoded_state)

        influence_reward = tf.expand_dims(self._social_influence_reward, axis=-1)
        one_hot_actions = tf.reshape(
            self._true_one_hot_actions, shape=[-1, self._true_one_hot_actions.shape[-1]]
        )
        # Stop backpropagation through the LSTM.
        # This is done because the SCM should not influence what the MOA is modeling
        lstm_input = tf.stop_gradient(state[2])

        # TODO(@internetcoffeephone): Change state[6] magic number to something that does not depend
        #  on the order
        # Note that the inputs are different from the paper: we can only work with historical actions
        # values, so the inputs of the forward and inverse models are "behind" by 1 timestep, which
        # is corrected for in the reward function.
        forward_model_input = {
            # Encoded state at t-1
            "encoded_input_now": state[6],
            # Social influence at t-1
            "influence_reward_input": influence_reward,
            # Actions at t-1
            "action_input": one_hot_actions,
            # MOA LSTM output at t-1
            "lstm_input": lstm_input,
        }

        inverse_model_input = {
            # Encoded state at t-1
            "encoded_input_now": state[6],
            # Encoded state at t
            "encoded_input_next": encoded_state,
            # Actions at t-1
            "action_input": one_hot_actions,
            # MOA LSTM output at t-1
            "lstm_input": lstm_input,
        }

        forward_model_output = self.forward_model(forward_model_input)
        inverse_model_output = self.inverse_model(inverse_model_input)

        curiosity_reward = self.compute_curiosity_reward(encoded_state, forward_model_output)
        curiosity_reward = tf.reshape(curiosity_reward, [-1])
        self._social_curiosity_reward = curiosity_reward

        inverse_model_loss = self.compute_inverse_model_loss(influence_reward, inverse_model_output)
        self._inverse_model_loss = tf.reshape(inverse_model_loss, [-1])

        return output, new_state

    def compute_curiosity_reward(self, true_encoded_state, predicted_encoded_state):
        mse = self.batched_mse(true_encoded_state, predicted_encoded_state)
        div_mse = tf.multiply(mse, 0.5, name="mult_mse")
        return div_mse

    def compute_inverse_model_loss(self, true_influence_reward, predicted_influence_reward):
        return self.batched_mse(true_influence_reward, predicted_influence_reward)

    @staticmethod
    def batched_mse(true_tensor, pred_tensor):
        """
        Calculate the mean square error on a batched tensor.
        The output has the same amount of dimensions as the input,
        but sets the last dimension size to 1, which contains the mean.
        :param true_tensor: The true values
        :param pred_tensor: The predicted values
        :return: The mean square error between the true and predicted tensors.
        """
        squared_difference = tf.squared_difference(true_tensor, pred_tensor)
        mse = tf.reduce_mean(squared_difference, axis=-1, keepdims=True)
        return mse

    def social_curiosity_reward(self):
        return self._social_curiosity_reward

    def inverse_model_loss(self):
        return self._inverse_model_loss

    @override(MOAModel)
    def get_initial_state(self):
        """
        :return: This model's initial state. Consists of the MOA initial state, plus the output of
        the encoder at time t.
        """
        moa_initial_state = super(SocialCuriosityModule, self).get_initial_state()
        return moa_initial_state + [np.zeros(self.scm_encoder_model.output_shape[-1], np.float32)]
