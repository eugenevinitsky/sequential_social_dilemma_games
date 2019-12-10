from gym.spaces import Box
import numpy as np
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

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
        super(KerasRNN, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.cell_size = cell_size
        self.use_value_fn = use_value_fn

        # TODO(@internetcoffeephone) make this time distributed only at the last moment
        input_layer = tf.keras.layers.Input(
            shape=(None,) + obs_space.shape, name="inputs"
        )
        flat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(
            input_layer
        )

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
        )(
            inputs=last_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name=name
        )(lstm_out)

        inputs = [input_layer, seq_in, state_in_h, state_in_c]
        if use_value_fn:
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01),
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


class CuriosityLSTM(RecurrentTFModelV2):
    """An LSTM with two heads, one for taking actions and one for predicting future state."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CuriosityLSTM, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.obs_space = obs_space
        self.num_outputs = num_outputs
        self.num_other_agents = model_config["custom_options"]["num_other_agents"]

        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # Determine vision network input shape: add an extra none for the time dimension
        inputs = tf.keras.layers.Input(
            shape=(None,) + original_obs_dims, name="observations"
        )

        # A temp config with custom_model false so that we can get a basic vision model
        # with the desired filters
        # Build the CNN layer
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

        # now output two heads, one for action selection and one for the prediction the next state
        inner_obs_space = Box(
            low=-1, high=1, shape=conv_out.shape[2:], dtype=np.float32
        )
        inner_obs_shape = inner_obs_space.shape
        inner_obs_size = int(
            inner_obs_shape[0] * inner_obs_shape[1] * inner_obs_shape[2]
        )

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

        # predicts the actions of all the agents besides itself
        # create a new input reader per worker
        self.aux_loss_weight = model_config["custom_options"]["aux_loss_weight"]

        self.curiosity_model = KerasRNN(
            inner_obs_space,
            action_space,
            inner_obs_size,
            model_config,
            "curiosity_model",
            cell_size=cell_size,
            use_value_fn=False,
        )
        self.register_variables(self.policy_model.rnn_model.variables)
        self.register_variables(self.curiosity_model.rnn_model.variables)
        self.policy_model.rnn_model.summary()
        self.curiosity_model.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""
        # first we add the time dimension for each object
        new_dict = {
            "obs": {
                k: add_time_dimension(v, seq_lens) for k, v in input_dict["obs"].items()
            }
        }
        new_dict.update(
            {"prev_action": add_time_dimension(input_dict["prev_actions"], seq_lens)}
        )

        output, new_state = self.forward_rnn(new_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        trunk = self.base_model(input_dict["obs"]["curr_obs"])
        pass_dict = {"curr_obs": trunk, "test_var": 0}
        h1, c1, h2, c2 = state

        # Save true encoded environment
        encoded_size = self.curiosity_model.num_outputs
        self._true_encoded_obs = tf.reshape(trunk, (-1, encoded_size))

        # Compute the next action
        (
            self._model_out,
            self._value_out,
            output_h1,
            output_c1,
        ) = self.policy_model.forward_rnn(pass_dict, [h1, c1], seq_lens)
        # Compute the next state prediction
        self._pred_encoded_obs, output_h2, output_c2 = self.curiosity_model.forward_rnn(
            pass_dict, [h2, c2], seq_lens
        )
        self._pred_encoded_obs = tf.reshape(self._pred_encoded_obs, (-1, encoded_size))

        return self._model_out, [output_h1, output_c1, output_h2, output_c2]

    def action_logits(self):
        return self._model_out

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def true_encoded_observations(self):
        return self._true_encoded_obs

    def predicted_encoded_observations(self):
        return self._pred_encoded_obs

    @override(ModelV2)
    def get_initial_state(self):
        return (
            self.policy_model.get_initial_state()
            + self.curiosity_model.get_initial_state()
        )
