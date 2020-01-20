import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()


class KerasRNN(RecurrentTFModelV2):
    """Maps the input direct to an LSTM cell"""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        cell_size=64,
        use_value_fn=False,
    ):
        super(KerasRNN, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = cell_size
        self.use_value_fn = use_value_fn

        # TODO(@internetcoffeephone) make this time distributed only at the last moment
        input_layer = tf.keras.layers.Input(shape=(None,) + obs_space.shape, name="inputs")
        flat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input_layer)

        # Add the fully connected layers
        hiddens = model_config.get("fcnet_hiddens")
        last_layer = flat_layer
        i = 1
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}_{}".format(i, name),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
            i += 1

        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(inputs=last_layer, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c],)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name=name
        )(lstm_out)

        inputs = [input_layer, seq_in, state_in_h, state_in_c]
        if use_value_fn:
            value_out = tf.keras.layers.Dense(
                1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01),
            )(lstm_out)
            outputs = [logits, value_out, state_h, state_c]
        else:
            outputs = [logits, state_h, state_c]

        self.rnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens):
        input = [input_dict["curr_obs"], seq_lens] + state

        if self.use_value_fn:
            model_out, self._value_out, h, c = self.rnn_model(input)
            return model_out, self._value_out, h, c
        else:
            model_out, h, c = self.rnn_model(input)
            return model_out, h, c

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
