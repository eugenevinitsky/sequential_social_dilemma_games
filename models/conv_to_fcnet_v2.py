from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn, flatten
from ray.rllib.utils import try_import_tf
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override


tf = try_import_tf()


class ConvToFCNetv2(RecurrentTFModelV2):
    """Implementation of the model in the causal influence paper."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(ConvToFCNetv2, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        self.model_config = model_config
        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        if not filters:
            filters = _get_filter_config(obs_space.shape)
        vf_share_layers = model_config.get("vf_share_layers")

        inputs = tf.keras.layers.Input(
            shape=obs_space, name="observations")
        last_layer = inputs

        # Build the action layers
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                name="conv{}".format(i))(last_layer)
        out_size, kernel, stride = filters[-1]

        if len(filters) == 1:
            i = 0
        last_layer = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            name="conv{}".format(i + 1))(last_layer)
        last_layer = flatten(last_layer)
        # Now apply the Dense layers
        hiddens = [32, 32]
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        i += 1

        # Now lets construct the LSTM
        # Preprocess observation with a hidden layer and send to LSTM cell
        cell_size = model_config["custom_options"]["cell_size"]
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        import ipdb;ipdb.set_trace()
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
            inputs=last_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(lstm_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[inputs, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        inputs = input_dict["obs"]["curr_obs"]
        model_out, h, c = self.rnn_model([inputs, seq_lens] +
                                         state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]