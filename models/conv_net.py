# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1,
# and options['conv_filters'] output channels.

import tensorflow as tf
import tensorflow.contrib.slim as slim
from ray.rllib.models.misc import flatten
from ray.rllib.models.model import Model


class ConvNet(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]

        with tf.name_scope("custom_net"):
            last_layer = slim.conv2d(
                inputs,
                options["conv_filters"],
                [3, 3],
                1,
                activation_fn=tf.nn.relu,
                scope="conv",
            )
            output = flatten(last_layer)
            return output, last_layer
