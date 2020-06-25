import argparse

from config.default_args import add_default_args
from run_scripts.train import create_experiment, run


def evaluate_bash_expression(line):
    """
    !UNSAFE! - uses eval.
    Evaluate a bash expression from a string.
    Can only handle a single bash expression per line.
    :param line: The bash line to evaluate
    :return: The evaluated run script line.
    """
    if "$((" not in line:
        return line
    else:
        arg_name = line.split(" ")[0]
        expression = line.split("$((")[1][:-3]
        number = str(eval(expression))
        return arg_name + " " + number + " "


def parse_run_script(filename):
    """
    Parses the args from a single run script.
    :param filename: The run script filename.
    :return: The args belonging to the run script.
    """
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    run_script = open(filename, "r").read()
    args = run_script.split("\\\n")[1:]
    replace_bash_args = [evaluate_bash_expression(line).replace("\n", "") for line in args]
    args = "".join(replace_bash_args).split(" ")
    parsed_args = parser.parse_args(args)
    return parsed_args


def parse_filenames():
    """
    Parse filenames that were provided as args.
    :return: A list of run script filenames.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_script_filenames",
        nargs="+",
        type=str,
        default=[
            "run_moa_cleanup.sh",
            "run_moa_harvest.sh",
            "run_scm_cleanup.sh",
            "run_scm_harvest.sh",
            "run_baseline_cleanup.sh",
            "run_baseline_harvest.sh",
        ],
        help="Names of scripts to run concurrently, as resources allow.",
    )
    args = parser.parse_args()
    return args.run_script_filenames


if __name__ == "__main__":
    run_scripts = parse_filenames()
    args_set = [parse_run_script(filename) for filename in run_scripts]
    experiments = [create_experiment(args) for args in args_set]
    run(args_set[0], experiments)
