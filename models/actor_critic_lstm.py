import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

tf = try_import_tf()


class ActorCriticLSTM(RecurrentTFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, cell_size=64,
    ):
        """
        Create a LSTM with an actor-critic output: an output head with size num_outputs for the
        policy, and an output head of size 1 for the value function.
        :param obs_space: The size of the previous layer.
        :param action_space: The amount of actions available to the agent.
        :param num_outputs: The amount of actions available to the agent.
        :param model_config: The config dict for the model, unused.
        :param name: The name of the model.
        :param cell_size: The amount of LSTM units.
        """
        super(ActorCriticLSTM, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.cell_size = cell_size

        input_layer = tf.keras.layers.Input(shape=(None, obs_space), name="inputs")

        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm"
        )(inputs=input_layer, mask=tf.sequence_mask(seq_in), initial_state=[state_in_h, state_in_c],)

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name=name
        )(lstm_out)

        inputs = [input_layer, seq_in, state_in_h, state_in_c]
        value_out = tf.keras.layers.Dense(
            1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01),
        )(lstm_out)
        outputs = [logits, value_out, state_h, state_c]

        self.rnn_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Actor_Critic_Model")

    @override(RecurrentTFModelV2)
    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Forward pass through the LSTM.
        :param input_dict: The input tensors.
        :param state: The model state.
        :param seq_lens: LSTM sequence lengths.
        :return: The model output.
        """
        input = [input_dict["curr_obs"], seq_lens] + state

        model_out, self._value_out, h, c = self.rnn_model(input)
        return model_out, self._value_out, h, c

    @override(ModelV2)
    def get_initial_state(self):
        """
        :return: Initial state of this model: all-zeros for the LSTM layer's hidden and cell states.
        """
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
