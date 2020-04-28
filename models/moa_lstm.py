import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()


class MoaLSTM(RecurrentTFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, cell_size=64,
    ):
        super(MoaLSTM, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = cell_size

        # Define input layers
        # TODO(@evinitsky) add in an option for prev_action_reward
        input_layer = tf.keras.layers.Input(shape=(None, obs_space), name="inputs")

        self.actions_layer = tf.keras.layers.Input(
            shape=(None, self.num_outputs + self.action_space.n), name="action_input"
        )
        last_layer = tf.keras.layers.concatenate([input_layer, self.actions_layer])

        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        self.lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(inputs=last_layer, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c],)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name=name
        )(self.lstm_out)

        inputs = [input_layer, seq_in, state_in_h, state_in_c]
        inputs.insert(1, self.actions_layer)
        outputs = [logits, state_h, state_c]
        self.rnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens):
        rnn_input = [input_dict["curr_obs"], seq_lens] + state
        rnn_input.insert(1, input_dict["prev_total_actions"])
        model_out, h, c = self.rnn_model(rnn_input)
        return model_out, h, c

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
