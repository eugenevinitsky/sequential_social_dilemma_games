def add_default_args(parser):
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name experiment will be stored under. When left empty, the name is formatted as:"
        "env_model_algorithm",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="cleanup",
        help="Name of the environment to use. Can be switch, cleanup or harvest.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="Name of the rllib algorithm to use. Can be A3C or PPO.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        help="Name of the model to use. Can be baseline, moa, or scm",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume previous experiment.",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="path to checkpoint",
    )
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agent policies")
    parser.add_argument(
        "--rollout_fragment_length",
        type=int,
        default=1000,
        help="Size of samples taken from single workers. These are concatenated with samples of"
        "other workers to size train_batch_size.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Size of the total dataset over which one epoch is computed. If not specified,"
        "defaults to num_workers * num_envs_per_worker * rollout_fragment_length",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=100,
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
        default=None,
        help="Experiment stops when this is the minimum episode reward within 1 iteration",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Amount of times to repeat all experiments",
    )
    parser.add_argument("--memory", type=int, default=None, help="Amount of total usable memory")
    parser.add_argument(
        "--object_store_memory",
        type=int,
        default=None,
        help="Amount of memory for the object store",
    )
    parser.add_argument(
        "--redis_max_memory", type=int, default=None, help="Amount of memory for redis"
    )

    parser.add_argument("--num_workers", type=int, default=4, help="Total number of workers")
    parser.add_argument(
        "--cpus_for_driver", type=int, default=0, help="Number of CPUs used by the driver"
    )
    parser.add_argument(
        "--gpus_for_driver", type=float, default=1, help="Number of GPUs used by the driver"
    )
    parser.add_argument(
        "--cpus_per_worker", type=int, default=1, help="Number of CPUs used by one worker"
    )
    parser.add_argument(
        "--gpus_per_worker", type=float, default=0, help="Number of GPUs used by one worker"
    )

    parser.add_argument(
        "--num_envs_per_worker",
        type=int,
        default=8,
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
        "--address",
        type=str,
        default=None,
        help="The address of the Ray cluster to connect to.",
    )
    parser.add_argument("--use_s3", action="store_true", default=False, help="If true upload to s3")
    parser.add_argument(
        "--tune_hparams",
        action="store_true",
        default=False,
        help="When provided, run population-based training over hyperparameters",
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
        help="Default learning rate. Used when lr_schedule_steps/weights are not provided.",
    )
    parser.add_argument(
        "--lr_schedule_steps",
        nargs="+",
        type=int,
        default=None,
        help="Amounts of environment steps at which the learning rate has a value specified in"
        "--lr_schedule_weights",
    )
    parser.add_argument(
        "--lr_schedule_weights",
        nargs="+",
        type=float,
        default=None,
        help="Values for the learning rate schedule. Linearly interpolates using "
        "--lr_schedule_steps",
    )

    parser.add_argument("--entropy_coeff", type=float, default=0.001, help="Entropy reward weight.")

    parser.add_argument(
        "--use_collective_reward",
        action="store_true",
        default=False,
        help="Train using collective reward instead of individual reward.",
    )

    # MOA Parameters
    parser.add_argument(
        "--moa_loss_weight",
        type=float,
        default=1.0,
        help="Loss weight of the moa network",
    )

    parser.add_argument(
        "--influence_reward_weight",
        type=float,
        default=0.001,
        help="The moa reward weight.",
    )
    parser.add_argument(
        "--influence_reward_schedule_steps",
        nargs="+",
        type=int,
        default=None,
        help="Amounts of environment steps at which the moa reward has a value specified in"
        "--influence_reward_schedule_weights",
    )
    parser.add_argument(
        "--influence_reward_schedule_weights",
        nargs="+",
        type=float,
        default=None,
        help="Values for the moa reward schedule. Linearly interpolates using "
        "--influence_reward_schedule_steps. The final value is"
        " --influence_reward_weight * interpolated_value",
    )

    # SCM parameters
    parser.add_argument(
        "--scm_loss_weight",
        type=float,
        default=1.0,
        help="Loss weight of the scm network",
    )

    parser.add_argument(
        "--curiosity_reward_weight",
        type=float,
        default=0.001,
        help="The scm reward weight.",
    )
    parser.add_argument(
        "--curiosity_reward_schedule_steps",
        nargs="+",
        type=int,
        default=None,
        help="Amounts of environment steps at which the scm reward has a value specified in"
        "--curiosity_reward_schedule_weights",
    )
    parser.add_argument(
        "--curiosity_reward_schedule_weights",
        nargs="+",
        type=float,
        default=None,
        help="Values for the scm reward schedule. Linearly interpolates using "
        "--curiosity_reward_schedule_steps. The final value is"
        " --curiosity_reward_weight * interpolated_value",
    )

    parser.add_argument(
        "--scm_forward_vs_inverse_loss_weight",
        type=float,
        default=0.2,
        help="This weight balances forward and inverse loss weights in the following way:"
        "weight * forward_loss + (1 - weight) * inverse_loss"
        "Must be in the range [0, 1].",
    )

    # PPO parameters
    parser.add_argument(
        "--ppo_sgd_minibatch_size",
        type=int,
        default=None,
        help="Minibatch size for the stochastic gradient descent step in the PPO algorithm. If not"
        "specified, defaults to --train_batch_size / 2",
    )

    # Env-specific parameters
    parser.add_argument(
        "--num_switches",
        type=int,
        default=6,
        help="Amount of switches in a switch map environment",
    )
