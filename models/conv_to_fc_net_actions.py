# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import tensorflow as tf

from ray.rllib.models.misc import normc_initializer, flatten
from ray.rllib.models.model import Model
import tensorflow.contrib.slim as slim


class ConvToFCNetActions(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        others_actions = input_dict["other_actions"]
        num_other_agents = options["custom_options"]["num_other_agents"]
        # Reshape others actions into size:
        # [batch_size, num_outputs * num_other_agents]
        others_actions = tf.reshape(others_actions,
                                    [-1, num_outputs * num_other_agents])
        others_actions = tf.cast(others_actions, tf.float32)

        inputs = input_dict["obs"]

        hiddens = [32, 32]
        with tf.name_scope("custom_net"):
            inputs = slim.conv2d(
                inputs,
                6,
                [3, 3],
                1,
                activation_fn=tf.nn.relu,
                scope="conv")
            last_layer = flatten(inputs)
            i = 1
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.relu,
                    scope=label)
                i += 1

            # Add the others_actions in as input directly to the LSTM
            last_layer = tf.concat([last_layer, others_actions], 1)

            # Add an output layer just in case
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.relu,
                scope="output_layer",
            )

            return output, last_layer
