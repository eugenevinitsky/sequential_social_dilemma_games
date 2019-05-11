from os.path import abspath, join, dirname

from localconfig import config
import tensorflow as tf

config.read(abspath(join(dirname(__file__) + '/config.ini')))


def set_tf_flags(experiment_name):
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

    # Experiment name requires a combination of strings and is processed separately.
    flags.DEFINE_string(
        'exp_name', experiment_name + '_' + config.get('flags', 'env'),
        'Name of the ray_results experiment directory where results are stored.')


def get_env_params():
    env = tf.app.flags.FLAGS.env
    params = dict(config.items('parameters_' + env))

    tune = [value for key, value in params.items() if key.startswith('entropy_tune')]
    params = {key: params[key] for key in params if not key.startswith('entropy_tune')}
    params['entropy_tune'] = tune
    return params


def get_redis_address():
    address = config.get('network', 'redis_address')
    address = address if address != "" else None
    return address


def get_upload_dir():
    upload_dir = config.get('network', 'upload_dir')
    upload_dir = upload_dir if upload_dir != "" else None
    return upload_dir
