from copy import copy

from gym.spaces import Box
import numpy as np
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

class KerasRNN(RecurrentTFModelV2):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64,
                 use_value_fn=False):
        super(KerasRNN, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size
        self.use_value_fn = use_value_fn

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(cell_size, ), name="h")
        state_in_c = tf.keras.layers.Input(shape=(cell_size, ), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        # Preprocess observation with a hidden layer and send to LSTM cell
        dense1 = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")(input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")(
                inputs=dense1,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h, state_in_c])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(lstm_out)
        if use_value_fn:
            value_out = tf.keras.layers.Dense(
                1,
                name="value_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(lstm_out)
            self.rnn_model = tf.keras.Model(
                inputs=[input_layer, seq_in, state_in_h, state_in_c],
                outputs=[logits, value_out, state_h, state_c])
        else:
            self.rnn_model = tf.keras.Model(
                inputs=[input_layer, seq_in, state_in_h, state_in_c],
                outputs=[logits, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        if self.use_value_fn:
            model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                             state)
        else:
            model_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                              state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]


class MOA_LSTM(RecurrentTFModelV2):
    """An LSTM with two heads, one for taking actions and one for predicting actions. Currently only works for starcraft"""
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=256,
                 cell_size=64):
        temp_config = copy(model_config)
        temp_config["custom_model"] = False

        # The inputs of the shared trunk. We will concatenate the observation space with shared info about the
        # visibility of agents. Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_other_agents = model_config['num_other_agents']
        # TODO(@evinitsky) this is going to break for discrete so watch out
        num_actions = self.num_other_agents * action_space.shape[0]
        num_prev_action = action_space.shape[0]
        num_prev_rewards = 1
        input_size = obs_space.shape[0] + num_actions + num_prev_action + num_prev_rewards
        input_obs_space = Box(low=-1, high=1, shape=(input_size,))

        # The number of features that will be output by the shared trunk
        inner_feature_dim = model_config.get("inner_feature_dim")
        self.trunk = ModelCatalog.get_model_v2(input_obs_space, action_space, inner_feature_dim, temp_config)

        # now output two heads, one for action selection and one for the prediction of other agents
        inner_obs_space = Box(low=-1, high=1, shape=(inner_feature_dim))
        self.actions_model = KerasRNN(inner_obs_space, action_space, num_outputs,
                                                model_config, "actions", use_value_fn=True)
        self.moa_model = KerasRNN(inner_obs_space, action_space, num_actions, model_config, "moa_model")

    def forward(self, input_dict, state, seq_lens):
        # we operate on our obs, others previous actions, our previous actions, our previous rewards
        # TODO(@evinitsky) are we passing seq_lens correctly?
        h1, c1, h2, c2 = state
        model_out, self._value_out, h1, c1 = self.actions_model.forward([np.concatenate((input_dict["obs"]["curr_obs"],
                                                                      input_dict["obs"]["others_actions"],
                                                         input_dict["prev_actions"], input_dict["prev_rewards"])), seq_lens] + [h1, c1])

        # TODO(@evinitsky) make sure the other_actions only contain the actions of other agents

        possible_actions = np.arange(self.num_outputs)[np.newaxis, :]
        other_actions = input_dict["obs"]["others_actions"]
        other_actions_tile = np.repeat(other_actions, self.num_outputs, axis=0)
        stacked_actions = np.hstack(possible_actions, other_actions_tile)
        self._conterfactual_preds, h2, c2 = self.moa_model.forward([stacked_actions, seq_lens] + [h2, c2])

        return model_out, [h1, c1, h2, c2]

    def get_batch_outputs(self, train_batch, is_training=True):
        """Operate over a batch and return the MOA predictions as well as the logits
        """

        # TODO(@evinitsky) make this output theright thing

        input_dict = {
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": is_training,
        }
        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []
        i = 0
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1
        # TODO(@evinitsky) this is probably wrong
        return self.__call__(input_dict, states, train_batch.get("seq_lens"))

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def counterfactual_actions(self):
        return self._conterfactual_preds