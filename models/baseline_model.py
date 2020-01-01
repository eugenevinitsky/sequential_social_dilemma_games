import numpy as np
from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import get_activation_fn
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.curiosity_model import KerasRNN

tf = try_import_tf()


class Baseline_LSTM(RecurrentTFModelV2):
    """The baseline model from the causal influence paper"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(Baseline_LSTM, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space
        self.num_outputs = num_outputs

        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # Determine vision network input shape: add an extra none for the time dimension
        inputs = tf.keras.layers.Input(shape=(None,) + original_obs_dims, name="observations")

        # Build the CNN layers
        last_layer = inputs
        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    out_size,
                    kernel,
                    strides=(stride, stride),
                    activation=activation,
                    padding="same",
                    channels_last=True,
                    name="conv{}".format(i),
                )
            )(last_layer)
        out_size, kernel, stride = filters[-1]
        if len(filters) == 1:
            i = -1

        # should be batch x time x height x width x channel
        conv_out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="valid",
                name="conv{}".format(i + 1),
            )
        )(last_layer)

        self.base_model = tf.keras.Model(inputs, [conv_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

        inner_obs_space = Box(low=-1, high=1, shape=conv_out.shape[2:], dtype=np.float32)

        # Action selection/value function
        cell_size = model_config["custom_options"].get("cell_size")
        self.policy_model = KerasRNN(
            inner_obs_space,
            action_space,
            num_outputs,
            model_config,
            "policy",
            cell_size=cell_size,
            use_value_fn=True,
        )

        self.register_variables(self.policy_model.rnn_model.variables)
        self.policy_model.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""
        # first we add the time dimension for each object
        new_dict = {
            "obs": {k: add_time_dimension(v, seq_lens) for k, v in input_dict["obs"].items()}
        }
        new_dict.update({"prev_action": add_time_dimension(input_dict["prev_actions"], seq_lens)})

        output, new_state = self.forward_rnn(new_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        trunk = self.base_model(input_dict["obs"]["curr_obs"])
        pass_dict = {"curr_obs": trunk, "test_var": 0}
        h1, c1 = state

        # Compute the next action
        (self._model_out, self._value_out, output_h1, output_c1,) = self.policy_model.forward_rnn(
            pass_dict, [h1, c1], seq_lens
        )

        return self._model_out, [output_h1, output_c1]

    def action_logits(self):
        return self._model_out

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return self.policy_model.get_initial_state()
