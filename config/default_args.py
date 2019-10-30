def add_default_args(parser):
    parser.add_argument('--exp_name', type=str, default='causal_env', help='Name experiment will be stored under')
    parser.add_argument('--env', type=str, default='cleanup', help='Name of the environment to rollout. Can be '
                                                                   'cleanup or harvest.')
    parser.add_argument('--algorithm', type=str, default='PPO', help='Name of the rllib algorithm to use.')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agent policies')
    parser.add_argument('--sample_batch_size', type=int, default=1000,
                        help='Size of samples taken from single workers, concatenated to size train_batch_size.')
    parser.add_argument('--train_batch_size', type=int, default=30000,
                        help='Size of the total dataset over which one epoch is computed.')
    parser.add_argument('--checkpoint_frequency', type=int, default=50,
                        help='Number of steps before a checkpoint is saved.')
    parser.add_argument('--training_iterations', type=int, default=50, help='Total number of steps to train for')
    parser.add_argument('--num_cpus', type=int, default=2, help='Number of available CPUs')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of available GPUs')
    parser.add_argument('--use_gpus_for_workers', action='store_true', default=False,
                        help='Set to true to run workers on GPUs rather than CPUs')
    parser.add_argument('--use_gpu_for_driver', action='store_true', default=False,
                        help='Set to true to run driver on GPU rather than CPU.')
    parser.add_argument('--num_workers_per_device', type=float, default=1,
                        help='Number of workers to place on a single device (CPU or GPU)')
    parser.add_argument('--num_envs_per_worker', type=float, default=1,
                        help='Number of envs to place on a single worker')
    parser.add_argument('--multi_node', action='store_true', default=False,
                        help='If true the experiments are run in multi-cluster mode')
    parser.add_argument('--local_mode', action='store_true', default=False,
                        help='Force all the computation onto the driver. Useful for debugging.')
    parser.add_argument('--eager_mode', action='store_true', default=False,
                        help='Perform eager execution. Useful for debugging.')
    parser.add_argument('--use_s3', action='store_true', default=False,
                        help='If true upload to s3')
    parser.add_argument('--grid_search', action='store_true', default=False,
                        help='If true run a grid search over relevant hyperparameters')
    parser.add_argument('--num_switches', type=int, default=6,
                        help='Amount of switches in a switch map environment')
