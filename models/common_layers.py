from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.framework import get_activation_fn

tf = try_import_tf()


def build_fc_layers(model_config, last_layer, name):
    """
    Create a sequence of fully-connected (dense) layers.
    :param model_config: The config dict containing information on what fully-connected layers to
    create.
    :param last_layer: The layer that feeds into the fully connected layer(s) constructed here.
    :param name: The FC layer name.
    :return: The last constructed FC layer.
    """
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
    """
    Create a sequence of convoluational layers.
    :param model_config: The config dict containing information on what convolutional layers to
    create.
    :param last_layer: The layer that feeds into the convolutional layer(s) constructed here.
    :return: The last constructed convolutional layer.
    """
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
