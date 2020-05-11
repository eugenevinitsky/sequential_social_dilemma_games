import os

import pandas as pd

from utility_funcs import get_all_files

print_params = [
    "lr_schedule_steps",
    "lr_schedule_weights",
    "entropy_coeff",
    "moa_loss_weight",
    "influence_reward_weight",
    "influence_reward_schedule_steps",
    "influence_reward_schedule_weights",
    "scm_loss_weight",
    "scm_forward_vs_inverse_loss_weight",
    "curiosity_reward_weight",
]


def extract_script_params(script):
    param_lines = {}
    for line in script:
        # Starts with argument
        if line.count("--") == 1:
            split = line.split(" ")
            hparam_name = split[0][2:]
            if hparam_name in print_params:
                hparam_value = split[1:]
                if "\\\n" in hparam_value:
                    hparam_value.remove("\\\n")
                if len(hparam_value) == 1:
                    hparam_value = hparam_value[0]
                param_lines[hparam_name] = hparam_value
    return param_lines


def is_normal_experiment(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    return len(base_name.split("_")) == 3 and filename[-3:] == ".sh"


def get_model_and_env(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    split = base_name.split("_")
    return split[1], split[2]


def create_table_per_model(table_contents):
    for model in sorted(table_contents.keys()):
        df = pd.DataFrame(table_contents[model])
        latex = df.to_latex()
        latex = latex.replace("{lll}", "{|l|l|l|}")
        latex = latex.replace("{llll}", "{|l|l|l|l|}")
        for line in ["toprule", "midrule", "bottomrule"]:
            latex = latex.replace(line, "hline")
        if model == "moa" or model == "scm":
            model_name_in_table_title = model.upper()
        else:
            model_name_in_table_title = "Baseline"

        print("% Generated with run_script_to_latex_tables.py.")
        print(
            "\\begin{{figure}}\
                    \\centering\
                    \\caption{{{0} hyperparameters}}\
                    \\label{{fig:appendix_{1}_hparams}}".format(
                model_name_in_table_title, model
            )
        )
        print(latex)
        print("\\end{figure}")


def run():
    script_path = "../run_scripts"

    all_files = get_all_files(script_path)
    all_run_scripts = [file for file in all_files if is_normal_experiment(file)]

    table_contents = {}

    for run_script_filename in all_run_scripts:
        script_contents = open(run_script_filename).readlines()
        model, env = get_model_and_env(run_script_filename)
        if model not in table_contents.keys():
            table_contents[model] = {}
        table_contents[model][env] = extract_script_params(script_contents)

    create_table_per_model(table_contents)


run()
