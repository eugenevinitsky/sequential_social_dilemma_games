import numpy as np
from gym.spaces import Box
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import get_activation_fn, normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
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
        append_actions=False,
    ):
        super(KerasRNN, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = cell_size
        self.use_value_fn = use_value_fn
        self.append_actions = append_actions

        # Define input layers
        # TODO(@evinitsky) add in an option for prev_action_reward

        # TODO(@evinitsky) make this time distributed only at the last moment
        input_layer = tf.keras.layers.Input(shape=(None,) + obs_space.shape, name="inputs")
        flat_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(input_layer)

        if self.append_actions:
            name = "pred_logits"
        else:
            name = "action_logits"

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

        if self.append_actions:
            num_other_agents = model_config["custom_options"]["num_other_agents"]
            actions_layer = tf.keras.layers.Input(
                shape=(None, num_other_agents + 1), name="other_actions"
            )

            last_layer = tf.keras.layers.concatenate([last_layer, actions_layer])

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
        if self.append_actions:
            inputs.insert(1, actions_layer)
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
        if self.append_actions:
            input.insert(1, input_dict["prev_total_actions"])

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


class MOA_LSTM(RecurrentTFModelV2):
    """An LSTM with two heads, one for taking actions and one for predicting actions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MOA_LSTM, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.obs_space = obs_space

        # The inputs of the shared trunk. We will concatenate the observation space with
        # shared info about the visibility of agents.
        # Currently we assume all the agents have equally sized action spaces.
        self.num_outputs = num_outputs
        self.num_other_agents = model_config["custom_options"]["num_other_agents"]

        # Build the vision network here
        original_obs_dims = obs_space.original_space.spaces["curr_obs"].shape
        # an extra none for the time dimension
        inputs = tf.keras.layers.Input(
            shape=(None,) + original_obs_dims, name="observations", dtype="uint8"
        )

        # Divide by 255 to transform [0,255] uint8 rgb pixel values to [0,1] float32.
        last_layer = tf.keras.backend.cast(inputs, "float32")
        last_layer = tf.math.divide(last_layer, 255.0)

        # A temp config with custom_model false so that we can get a basic vision model
        # with the desired filters
        # Build the CNN layer
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

        # now output two heads, one for action selection and one for the prediction of other agents
        inner_obs_space = Box(low=-1, high=1, shape=conv_out.shape[2:], dtype=np.float32)

        cell_size = model_config["custom_options"].get("cell_size")
        self.actions_model = KerasRNN(
            inner_obs_space,
            action_space,
            num_outputs,
            model_config,
            "actions",
            cell_size=cell_size,
            use_value_fn=True,
            append_actions=False,
        )

        # predicts the actions of all the agents besides itself
        # create a new input reader per worker
        self.train_moa_only_when_visible = model_config["custom_options"][
            "train_moa_only_when_visible"
        ]
        self.moa_weight = model_config["custom_options"]["aux_loss_weight"]

        self.moa_model = KerasRNN(
            inner_obs_space,
            action_space,
            self.num_other_agents * num_outputs,
            model_config,
            "moa_model",
            cell_size=cell_size,
            use_value_fn=False,
            append_actions=True,
        )
        self.register_variables(self.actions_model.rnn_model.variables)
        self.register_variables(self.moa_model.rnn_model.variables)
        self.actions_model.rnn_model.summary()
        self.moa_model.rnn_model.summary()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn()"""
        # first we add the time dimension for each object
        new_dict = {
            "obs": {k: add_time_dimension(v, seq_lens) for k, v in input_dict["obs"].items()}
        }
        new_dict.update(
            {
                "prev_action": add_time_dimension(
                    tf.cast(input_dict["prev_actions"], tf.uint8), seq_lens
                )
            }
        )
        # new_dict.update({k: add_time_dimension(v, seq_lens)
        # for k, v in input_dict.items() if k != "obs"})

        output, new_state = self.forward_rnn(new_dict, state, seq_lens)
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        # we operate on our obs, others previous actions, our previous actions, our previous rewards
        # TODO(@evinitsky) are we passing seq_lens correctly?
        #  should we pass prev_actions, prev_rewards etc?

        trunk = self.base_model(input_dict["obs"]["curr_obs"])

        pass_dict = {"curr_obs": trunk}

        h1, c1, h2, c2 = state
        # TODO(@evinitsky) what's the right way to pass in the prev actions and such?
        (self._model_out, self._value_out, output_h1, output_c1,) = self.actions_model.forward_rnn(
            pass_dict, [h1, c1], seq_lens
        )

        # Cycle through all possible actions and get predictions for what other
        # agents would do if this action was taken at each trajectory step.

        # First we have to compute it over the trajectory to give us the hidden state
        # that we will actually use
        other_actions = input_dict["obs"]["other_agent_actions"]
        agent_action = tf.expand_dims(input_dict["prev_action"], axis=-1)
        stacked_actions = tf.concat([agent_action, other_actions], axis=-1, name="concat_lemon")
        cast_actions = tf.cast(stacked_actions, tf.float32)
        pass_dict = {"curr_obs": trunk, "prev_total_actions": cast_actions}

        # Compute the action prediction. This is unused in the actual rollout and is only to generate
        # a series of hidden states for the counterfactuals
        action_pred, output_h2, output_c2 = self.moa_model.forward_rnn(pass_dict, [h2, c2], seq_lens)

        # Now we can use that cell state to do the counterfactual predictions
        counterfactual_preds = []
        for i in range(self.num_outputs):
            possible_actions = np.array([i])[None, None, :]
            stacked_actions = tf.concat(
                [possible_actions, other_actions], axis=-1, name="concat_counterfactual"
            )
            cast_actions = tf.cast(stacked_actions, tf.float32)
            pass_dict = {"curr_obs": trunk, "prev_total_actions": cast_actions}
            counterfactual_pred, _, _ = self.moa_model.forward_rnn(pass_dict, [h2, c2], seq_lens)
            counterfactual_preds.append(tf.expand_dims(counterfactual_pred, axis=-2))
        self._counterfactual_preds = tf.concat(counterfactual_preds, axis=-2)

        # TODO(@evinitsky) move this into ppo_causal by using restore_original_dimensions()
        self._other_agent_actions = input_dict["obs"]["other_agent_actions"]
        self._visibility = input_dict["obs"]["visible_agents"]

        return self._model_out, [output_h1, output_c1, output_h2, output_c2]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def counterfactual_actions(self):
        return self._counterfactual_preds

    def moa_preds_from_batch(self, train_batch):
        """Convenience function that calls this model with a tensor batch.

        What this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.

        This is used when setting up the loss function.
        """

        obs_dict = restore_original_dimensions(train_batch["obs"], self.obs_space)
        curr_obs = obs_dict["curr_obs"]

        # stack the agent actions together
        other_agent_actions = tf.cast(obs_dict["other_agent_actions"], tf.uint8)
        agent_actions = tf.cast(
            tf.expand_dims(train_batch[SampleBatch.PREV_ACTIONS], axis=1), tf.uint8
        )
        prev_total_actions = tf.concat([agent_actions, other_agent_actions], axis=-1)
        prev_total_actions = tf.cast(prev_total_actions, tf.float32)

        # Now we add the appropriate time dimension
        curr_obs = add_time_dimension(curr_obs, train_batch.get("seq_lens"))
        prev_total_actions = add_time_dimension(prev_total_actions, train_batch.get("seq_lens"))

        trunk = self.base_model(curr_obs)
        input_dict = {
            "curr_obs": trunk,
            "is_training": True,
            "prev_total_actions": prev_total_actions,
        }
        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []

        # TODO(@evinitsky) remove the magic number
        i = 2
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1

        moa_preds, _, _ = self.moa_model.forward_rnn(input_dict, states, train_batch.get("seq_lens"))
        return moa_preds

    def action_logits(self):
        return self._model_out

    # TODO(@evinitsky) pull out the time slice
    def visibility(self):
        return tf.reshape(self._visibility, [-1, self.num_other_agents])

    def other_agent_actions(self):
        return tf.reshape(self._other_agent_actions, [-1, self.num_other_agents])

    @override(ModelV2)
    def get_initial_state(self):
        return self.actions_model.get_initial_state() + self.moa_model.get_initial_state()
