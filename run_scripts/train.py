import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.a3c.a3c_policy_graph_curiosity import A3CPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from config import config_parser
from social_dilemmas.envs.env_creator import get_env_creator
from models.conv_net import ConvNet

config_parser.set_tf_flags()
FLAGS = tf.app.flags.FLAGS
hparams = config_parser.get_env_params()


def setup(env, num_cpus, num_gpus, num_agents, use_gpus_for_workers=False,
          use_gpu_for_driver=False, num_workers_per_device=1, tune_hparams=False):
    env_creator = get_env_creator(env, num_agents)
    env_name = env + "_env"
    register_env(env_name, env_creator)

    single_env = env_creator(1)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy(agent_id):
        return (A3CPolicyGraph, obs_space, act_space,
                {'num_other_agents': num_agents - 1, 'agent_id': agent_id})

    # Setup A3C with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    for i in range(num_agents):
        agent_id = 'agent-' + str(i)
        policy_graphs[agent_id] = gen_policy(agent_id)

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    model_name = "conv_net"
    ModelCatalog.register_custom_model(model_name, ConvNet)

    algorithm = 'A3C'
    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config_dict = {
        "sample_batch_size": 100,
        "train_batch_size": 200,
        "horizon": 1000,
        "lr_schedule": [[0, hparams['lr_init']],
                        [20000000, hparams['lr_final']]],
        "num_workers": num_workers,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
        "entropy_coeff": hparams['entropy_coeff'],
        "grad_clip": FLAGS.grad_clip,
        "multiagent": {
            "policy_graphs": policy_graphs,
            "policy_mapping_fn": tune.function(policy_mapping_fn),
                      },
        "model": {"custom_model": "conv_net",
                  "use_lstm": True,
                  "lstm_cell_size": 128,
                  "conv_filters": 6,
                  "fcnet_hiddens": [32, 32],
                  "custom_options": {
                      "aux_loss_weight": hparams["aux_loss_weight"],
                      "aux_reward_clip": 10,
                      "aux_reward_weight": hparams["aux_reward_weight"],
                      "aux_curriculum_steps": 1e7,
                      "aux_scaledown_start": 1e8,
                      "aux_scaledown_end": 3e8,
                      "aux_scaledown_final_val": 0.5}
                  },
        "callbacks": single_env.get_environment_callbacks(),
    }

    if tune_hparams:
        config_dict["entropy_coeff"] = tune.grid_search(hparams['entropy_tune'])
        config_dict["model"]["custom_options"]["aux_loss_weight"] = tune.grid_search(hparams['aux_loss_weight_tune'])
        config_dict["model"]["custom_options"]["aux_reward_weight_tune"] = tune.grid_search(hparams['aux_reward_weight_tune'])

    config.update(config_dict)
    return algorithm, env_name, config


def main(unused_argv):
    ray.init(object_store_memory=config_parser.sanitize_int_flag(FLAGS.object_store_memory),
             redis_max_memory=config_parser.sanitize_int_flag(FLAGS.redis_max_memory),
             redis_address=config_parser.get_redis_address(),
             local_mode=FLAGS.local_mode)
    alg_run, env_name, config = setup(FLAGS.env, FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device, FLAGS.tune)

    print('Commencing experiment', FLAGS.exp_name)

    run_experiments({
        FLAGS.exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "timesteps_total": FLAGS.stop_at_timesteps_total,
                "episode_reward_min": FLAGS.stop_at_episode_reward_min
            },
            'checkpoint_freq': 100,
            "config": config,
            'upload_dir': config_parser.get_upload_dir()
        }
    }, resume=FLAGS.resume)


if __name__ == '__main__':
    tf.app.run(main)
