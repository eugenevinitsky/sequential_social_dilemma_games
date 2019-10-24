import ast
from os.path import abspath, join, dirname, expanduser

from localconfig import config
import tensorflow as tf

config.read(abspath(join(dirname(__file__) + '/config.ini')))


def set_tf_flags(experiment_name=None):
    '''Sets tensorflow flags
    :param str experiment_name: Name of the ray_results experiment directory where results are stored.
     The environment is automatically appended, e.g. for the cleanup game, 'dir_name' becomes 'dir_name_cleanup'.
    '''

    flags = tf.app.flags

    for flag, value in config.items('flags'):
        if flag.endswith('_docstring'):
            continue
        else:
            docstring = config.get('flags', flag + '_docstring')

        if type(value) is str:
            flags.DEFINE_string(flag, value, docstring)
        elif type(value) is int:
            flags.DEFINE_integer(flag, value, docstring)
        elif type(value) is float:
            flags.DEFINE_float(flag, value, docstring)
        elif type(value) is bool:
            flags.DEFINE_boolean(flag, value, docstring)

    if experiment_name is None:
        experiment_name = config.get('flags', 'experiment')

    # Experiment name requires a combination of strings and is processed separately.
    flags.DEFINE_string(
        'exp_name', experiment_name + '_' + config.get('flags', 'env'),
        'Name of the ray_results experiment directory where results are stored.')


def get_env_params(model_type=None):
    if model_type is None:
        model_type = tf.app.flags.FLAGS.experiment
    env = tf.app.flags.FLAGS.env
    # Default parameters for given environment
    config_section = 'parameters_' + env

    # Set experiment-specific parameters
    if model_type is not None:
        config_section = config_section + "_" + model_type

    params = dict(config.items(config_section))

    for key, value in params.items():
        key_split = key.split('_')
        if key_split[-1] == 'tune':
            params[key] = ast.literal_eval(value)

    return params


def sanitize_int_flag(flag_string):
    return int(flag_string) if flag_string else None


def get_redis_address():
    address = config.get('network', 'redis_address')
    return address if address else None


def get_upload_dir():
    upload_dir = config.get('network', 'upload_dir')
    return upload_dir if upload_dir else None


def get_ray_results_path():
    ray_results_dir = expanduser(config.get('paths', 'ray_results_path'))
    return ray_results_dir


def get_plot_path():
    plot_path = expanduser(config.get('paths', 'plot_path'))
    return plot_path


def get_video_path():
    video_path = expanduser(config.get('paths', 'video_path'))
    return video_path
