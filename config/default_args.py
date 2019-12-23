def add_default_args(parser):
    parser.add_argument(
        "--exp_name",
        type=str,
        default="nameless_experiment",
        help="Name experiment will be stored under",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="switch",
        help="Name of the environment to use. Can be\
                                                                   cleanup or harvest.",
    )
    parser.add_argument(
        "--algorithm", type=str, default="A3C", help="Name of the rllib algorithm to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="curiosity",
        help="Name of the model to use. Can be curiosity," "moa, moa_curiosity",
    )
    parser.add_argument(
        "--small_model",
        action="store_true",
        default=False,
        help="Set to true to use a neural network with smaller layers.",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agent policies")
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=2,
        help="Size of samples taken from single workers, concatenated to size train_batch_size.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Size of the total dataset over which one epoch is computed.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=50,
        help="Number of steps before a checkpoint is saved.",
    )
    parser.add_argument(
        "--stop_at_timesteps_total",
        type=int,
        default=int(5e6),
        help="Experiment stops when this total amount of timesteps has been reached",
    )
    parser.add_argument(
        "--stop_at_episode_reward_min",
        type=float,
        default=1000.0,
        help="Experiment stops when this is the minimum episode reward within 1 iteration",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Amount of times to repeat all experiments",
    )
    parser.add_argument("--memory", type=int, default=int(2e9), help="Amount of total usable memory")
    parser.add_argument(
        "--object_store_memory",
        type=int,
        default=None,
        help="Amount of memory for the object store",
    )
    parser.add_argument(
        "--redis_max_memory", type=int, default=None, help="Amount of memory for redis"
    )

    parser.add_argument("--num_cpus", type=int, default=2, help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of available GPUs")
    parser.add_argument(
        "--use_gpus_for_workers",
        action="store_true",
        default=False,
        help="Set to true to run workers on GPUs rather than CPUs",
    )
    parser.add_argument(
        "--use_gpu_for_driver",
        action="store_true",
        default=False,
        help="Set to true to run driver on GPU rather than CPU.",
    )
    parser.add_argument(
        "--num_workers_per_device",
        type=float,
        default=1,
        help="Number of workers to place on a single device (CPU or GPU)",
    )
    parser.add_argument(
        "--num_envs_per_worker",
        type=float,
        default=16,
        help="Number of envs to place on a single worker",
    )
    parser.add_argument(
        "--multi_node",
        action="store_true",
        default=False,
        help="If true the experiments are run in multi-cluster mode",
    )
    parser.add_argument(
        "--local_mode",
        action="store_true",
        default=False,
        help="Force all the computation onto the driver. Useful for debugging.",
    )
    parser.add_argument(
        "--eager_mode",
        action="store_true",
        default=False,
        help="Perform eager execution. Useful for debugging.",
    )
    parser.add_argument(
        "--address", type=str, default=None, help="The address of the Ray cluster to connect to.",
    )
    parser.add_argument("--use_s3", action="store_true", default=False, help="If true upload to s3")
    parser.add_argument(
        "--grid_search",
        action="store_true",
        default=False,
        help="If true run a grid search over relevant hyperparameters",
    )
    parser.add_argument(
        "--num_switches", type=int, default=6, help="Amount of switches in a switch map environment",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=40,
        help="Gradients are clipped by this amount per update.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Default learning rate. Not used due to lr_curriculum, only exists for debugging.",
    )
    parser.add_argument(
        "--lr_curriculum_steps",
        nargs="+",
        type=int,
        default=[0, int(2e7)],
        help="Amounts of environment steps at which the learning rate has a value specified in"
        "--lr_curriculum_weights",
    )
    parser.add_argument(
        "--lr_curriculum_weights",
        nargs="+",
        type=float,
        default=[0.001, 0.0001],
        help="Values for the learning rate curriculum. Linearly interpolates using "
        "--lr_curriculum_steps",
    )

    parser.add_argument("--entropy_coeff", type=float, default=0.001, help="Entropy reward weight.")
    parser.add_argument(
        "--aux_loss_weight", type=float, default=1.0, help="Loss weight of the auxiliary network",
    )

    parser.add_argument(
        "--aux_reward_weight", type=float, default=0.001, help="The auxiliary reward weight.",
    )
    parser.add_argument(
        "--aux_reward_curriculum_steps",
        nargs="+",
        type=int,
        default=[0, int(1e7), int(1e8)],
        help="Amounts of environment steps at which the aux reward has a value specified in"
        "--aux_reward_curriculum_weights",
    )
    parser.add_argument(
        "--aux_reward_curriculum_weights",
        nargs="+",
        type=float,
        default=[0, 1.0, 0.5],
        help="Values for the aux reward curriculum. Linearly interpolates using "
        "--aux_reward_curriculum_steps. The final value is"
        " --aux_reward_weight * interpolated_value",
    )

    parser.add_argument(
        "--entropy_tune",
        nargs="+",
        type=float,
        default=[0.001],
        help="When --grid_search is provided, perform a grid search over these entropy_coeff\
                                parameters. Replaces --entropy_coeff when used.",
    )
    parser.add_argument(
        "--aux_loss_weight_tune",
        nargs="+",
        type=float,
        default=[1.0],
        help="When --grid_search is provided, perform a grid search over these aux_loss_weight\
                                parameters. Replaces --aux_reward_weight when used.",
    )
    parser.add_argument(
        "--aux_reward_weight_tune",
        nargs="+",
        type=float,
        default=[0.001],
        help="When --grid_search is provided, perform a grid search over these aux_reward_weight\
                                parameters. Replaces --entropy_coeff.",
    )
