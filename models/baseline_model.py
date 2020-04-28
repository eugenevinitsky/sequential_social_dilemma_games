from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from models.actor_critic_lstm import ActorCriticLSTM

tf = try_import_tf()


class BaselineModel(RecurrentTFModelV2):
    """The baseline model from the causal influence paper"""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
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
        last_layer = self.build_conv_layers(model_config, last_layer)

        # Add the fully connected layers
        last_layer, last_size = self.build_fc_layers(model_config, last_layer, name)

        self.encoder_model = tf.keras.Model(inputs, [last_layer])
        self.register_variables(self.encoder_model.variables)
        self.encoder_model.summary()

        # Action selection/value function
        cell_size = model_config["custom_options"].get("cell_size")
        self.policy_model = ActorCriticLSTM(
            last_size, action_space, num_outputs, model_config, "policy", cell_size=cell_size,
        )

        self.register_variables(self.policy_model.rnn_model.variables)
        self.policy_model.rnn_model.summary()

    @staticmethod
    def build_conv_layers(model_config, last_layer):
        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                channels_last=True,
                name="conv{}".format(i),
            )(last_layer)
        out_size, kernel, stride = filters[-1]
        if len(filters) == 1:
            i = -1

        # should be batch x time x height x width x channel
        conv_out = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            name="conv{}".format(i + 1),
        )(last_layer)

        flattened_conv_out = tf.keras.layers.Flatten()(conv_out)

        return flattened_conv_out

    @staticmethod
    def build_fc_layers(model_config, last_layer, name):
        hiddens = model_config.get("fcnet_hiddens")
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        for i, size in enumerate(hiddens):
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}_{}".format(i + 1, name),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)
        return last_layer, size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""
        trunk = self.encoder_model(input_dict["obs"]["curr_obs"])
        new_dict = {"curr_obs": add_time_dimension(trunk, seq_lens)}

        output, new_state = self.forward_rnn(new_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        h1, c1 = state

        # Compute the next action
        (self._model_out, self._value_out, output_h1, output_c1,) = self.policy_model.forward_rnn(
            input_dict, [h1, c1], seq_lens
        )

        return self._model_out, [output_h1, output_c1]

    def action_logits(self):
        return self._model_out

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def get_initial_state(self):
        return self.policy_model.get_initial_state()
