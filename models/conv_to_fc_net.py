# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.models.model import Model
from ray.rllib.models.tf.misc import flatten, normc_initializer


class ConvToFCNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        # The last row is other agent actions
        if options["custom_options"]["return_agent_actions"]:
            inputs = input_dict["obs"][:-1, :]
        else:
            inputs = input_dict["obs"]

        hiddens = [32, 32]
        with tf.name_scope("custom_net"):
            inputs = slim.conv2d(
                inputs, 6, [3, 3], 1, activation_fn=tf.nn.relu, scope="conv"
            )
            last_layer = flatten(inputs)
            i = 1
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.relu,
                    scope=label,
                )
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out",
            )
            return output, last_layer
