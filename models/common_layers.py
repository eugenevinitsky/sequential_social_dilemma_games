from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.framework import get_activation_fn

tf = try_import_tf()


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
    return last_layer


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
