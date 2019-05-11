from localconfig import config
import tensorflow as tf

config.read('config.ini')


def set_tf_flags(experiment_name):
    '''Sets tensorflow flags
    :param str experiment_name: Name of the ray_results experiment directory where results are stored.
     The environment is automatically appended, e.g. for the cleanup game, 'dir_name' becomes 'dir_name_cleanup'.
    '''

    flags = tf.app.flags

    for flag, value in config.items('flags'):
        if flag.endswith('_docstring'):
            pass
        else:
            docstring = config['flags'][flag + '_docstring']

        if isinstance(value, str):
            flags.DEFINE_string(flag, value, docstring)
        elif isinstance(value, int):
            flags.DEFINE_integer(flag, value, docstring)
        elif isinstance(value, float):
            flags.DEFINE_float(flag, value, docstring)
        elif isinstance(value, bool):
            flags.DEFINE_boolean(flag, value, docstring)

    # Experiment name requires a combination of strings and is processed separately.
    flags.DEFINE_string(
        'exp_name', experiment_name + '_' + config['experiment']['env'],
        'Name of the ray_results experiment directory where results are stored.')


def get_default_params():
    def convert_dict_keys_to_float(param_dict):
        for param, value in param_dict.items():
            param_dict[param] = float(value)
            return param_dict

    cleanup_default_params = convert_dict_keys_to_float(config['default_parameters_cleanup'])
    harvest_default_params = convert_dict_keys_to_float(config['default_parameters_harvest'])

    return cleanup_default_params, harvest_default_params


def get_redis_address():
    address = config['network']['redis_address']
    address = address if address != "" else None
    return address


def get_upload_dir():
    upload_dir = config['network']['upload_dir']
    upload_dir = upload_dir if upload_dir != "" else None
    return upload_dir
