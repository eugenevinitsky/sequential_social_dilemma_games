import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.a3c.a3c_policy_graph_causal import A3CPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow as tf

from config import config_parser
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv
from models.conv_to_fc_net_actions import ConvToFCNetActions

config_parser.set_tf_flags('influence_a3c')
FLAGS = tf.app.flags.FLAGS
hparams = config_parser.get_env_params()


def setup(env, num_cpus, num_gpus, num_agents, use_gpus_for_workers=False,
          use_gpu_for_driver=False, num_workers_per_device=1, tune_hparams=False):
    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)

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
    model_name = "conv_to_fc_net_actions"
    ModelCatalog.register_custom_model(model_name, ConvToFCNetActions)

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
    if tune_hparams:
        config.update({
            "train_batch_size": 2000,
            "horizon": 1000,
            # "lr_schedule": [[0, tune.grid_search([5e-4, 5e-3])],
            #                 [20000000, tune.grid_search([5e-4, 5e-5, 5e-6])]],
            "num_workers": num_workers,
            "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": cpus_for_driver,
            "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
            "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
            "entropy_coeff": tune.grid_search(hparams['entropy_tune']),
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
            },
            "model": {"custom_model": "conv_to_fc_net_actions", "use_lstm": True,
                      "lstm_cell_size": 128, "lstm_use_prev_action_reward": True,
                      "custom_options": {
                          "num_other_agents": num_agents,
                          "moa_weight": tune.grid_search([10.0]),
                          "train_moa_only_when_visible": tune.grid_search([True, False]),
                          "influence_reward_clip": 10,
                          "influence_divergence_measure": 'kl',
                          "influence_reward_weight": tune.grid_search([1.0]),
                          "influence_curriculum_steps": tune.grid_search([10e6]),
                          "influence_scaledown_start": tune.grid_search([100e6]),
                          "influence_scaledown_end": tune.grid_search([300e6]),
                          "influence_scaledown_final_val": tune.grid_search([.5]),
                          "influence_only_when_visible": tune.grid_search([True, False])}}

        })
    else:
        config.update({
            # "train_batch_size": 2000,
            "horizon": 1000,
            # "lr_schedule": [[0, default_hparams['lr_init']],
            #                 [20000000, default_hparams['lr_final']]],
            "num_workers": num_workers,
            "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
            "num_cpus_for_driver": cpus_for_driver,
            "num_gpus_per_worker": num_gpus_per_worker,  # Can be a fraction
            "num_cpus_per_worker": num_cpus_per_worker,  # Can be a fraction
            # "entropy_coeff": default_hparams['entropy_coeff'],
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(policy_mapping_fn),
            },
            "model": {"custom_model": "conv_to_fc_net_actions", "use_lstm": True,
                      "lstm_cell_size": 128, "lstm_use_prev_action_reward": True,
                      "custom_options": {"num_other_agents": num_agents,
                                         "train_moa_only_when_visible": True,
                                         "moa_weight": 12.0,
                                         "influence_reward_clip": 10,
                                         "influence_divergence_measure": 'kl',
                                         "influence_reward_weight": 2.5,
                                         "influence_curriculum_steps": 10e6,
                                         "influence_scaledown_start": 100e6,
                                         "influence_scaledown_end": 300e6,
                                         "influence_scaledown_final_val": 0.5,
                                         "influence_only_when_visible": True}}

        })
    return algorithm, env_name, config


def main(unused_argv):
    # if FLAGS.debug:
    #     ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=FLAGS.object_store_memory,
    #              redis_max_memory=FLAGS.redis_max_memory)
    # else:
    #     ray.init(num_cpus=FLAGS.num_cpus, object_store_memory=FLAGS.object_store_memory,
    #              redis_max_memory=FLAGS.redis_max_memory)
    ray.init()  # redis_address=config_parser.get_redis_address()
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
                "training_iteration": 20000
            },
            'checkpoint_freq': 100,
            "config": config,
            'upload_dir': config_parser.get_upload_dir()
        }
    }, resume=FLAGS.resume)


if __name__ == '__main__':
    tf.app.run(main)
