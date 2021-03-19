import os.path
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from utility_funcs import get_all_subdirs

ray_results_path = os.path.expanduser("~/ray_results")
plot_path = os.path.expanduser("~/ray_results_plot")


class PlotGraphics(object):
    def __init__(self, column_name, legend_name, color):
        self.column_name = column_name
        self.legend_name = legend_name
        self.color = color


class PlotData(object):
    def __init__(self, x_data, y_data, column_name, legend_name, color):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_graphics = PlotGraphics(column_name, legend_name, color)


def plot_and_save(fn, path, file_name_addition):
    # Clear plot to prevent slowdown when drawing multiple figures
    plt.clf()
    fn()
    # Sort legend by label name
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].lower()))
    plt.legend(handles, labels)

    # Strip path of all but last folder
    path_split = os.path.dirname(path).split("/")
    pngpath = plot_path + "/png/" + path_split[-2] + "/"
    svgpath = plot_path + "/svg/" + path_split[-2] + "/"
    pngfile = pngpath + file_name_addition + ".png"
    svgfile = svgpath + file_name_addition + ".svg"
    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    if not os.path.exists(svgpath):
        os.makedirs(svgpath)
    plt.savefig(pngfile)
    plt.savefig(svgfile)


def plot_multiple_category_result(plotdata_list):
    for plotdata in plotdata_list:
        plot_single_category_result(
            plotdata.x_data,
            plotdata.y_data,
            plotdata.plot_graphics.color,
            plotdata.plot_graphics.legend_name,
            plotdata.plot_graphics.column_name,
            with_individual_experiments=False,
            with_label=True,
        )


def plot_single_category_result(
    x_lists,
    y_lists,
    color,
    legend_name,
    y_label_name,
    with_individual_experiments=True,
    with_label=False,
):
    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.nanmin(list(map(np.nanmin, x_lists)))
    x_max = np.nanmax(list(map(np.nanmax, x_lists)))
    y_max = np.nanmax(list(map(np.nanmax, y_lists)))
    interpolated_time = np.linspace(x_min, x_max, most_timesteps)
    interpolated_scores = []
    individual_experiment_label_added = False
    for x, y in zip(x_lists, y_lists):
        interpolated_score = np.interp(interpolated_time, x, y, left=np.nan, right=np.nan)
        interpolated_scores.append(interpolated_score)
        light_color = change_color_luminosity(color, 0.5)
        if with_label:
            label_name = legend_name
        elif not individual_experiment_label_added:
            label_name = legend_name + ": Individual experiment"
            individual_experiment_label_added = True
        else:
            label_name = None
        if with_individual_experiments:
            plt.plot(
                interpolated_time, interpolated_score, color=light_color, label=label_name, alpha=0.7
            )

    # Plot the mean and confidence intervals
    # Calculate t-value for p<0.05 CI
    interpolated_scores = np.array(interpolated_scores)
    num_experiments = interpolated_scores.shape[0]
    significance_level = 0.05
    t_value = t.ppf(1 - significance_level / 2, num_experiments - 1)
    sqrt_n = sqrt(num_experiments)
    means = []
    confidence_limits = []

    for std_dev_index in range(interpolated_scores.shape[-1]):
        std_dev = np.std(interpolated_scores[:, std_dev_index], ddof=1)
        mean_confidence_limit = std_dev * t_value / sqrt_n
        confidence_limits.append(mean_confidence_limit)
        mean = np.mean(interpolated_scores[:, std_dev_index])
        means.append(mean)

    lower_confidence_bound = means - np.array(confidence_limits)
    upper_confidence_bound = means + np.array(confidence_limits)

    plt.plot(interpolated_time, means, color=color, label=legend_name)
    fill_color = change_color_luminosity(color, 0.2)
    plt.fill_between(
        interpolated_time,
        lower_confidence_bound,
        upper_confidence_bound,
        color=fill_color,
        alpha=0.5,
    )

    plt.xlabel("Environment steps (1e8)")
    plt.ylabel(y_label_name)
    bottom = 0 if "reward" in y_label_name.lower() else None
    old_bot, old_top = plt.ylim()
    y_max = max(y_max, old_top)
    plt.ylim(bottom=bottom, top=y_max)
    plt.ticklabel_format(useOffset=False)


def extract_stats(dfs, requested_keys):
    column_names = [df.columns.values.tolist() for df in dfs]
    column_names = [item for sublist in column_names for item in sublist]
    available_keys = [name.split("/")[-1] for name in column_names]
    unique_keys = set()
    [unique_keys.add(key) for key in available_keys]
    available_keys = list(unique_keys)
    keys = [key for key in requested_keys if key in available_keys]

    all_df_lists = {}
    for key in keys:
        all_df_lists[key] = []

    # Per file, extract the mean trajectory for each key.
    # The mean is taken over all distinct agents, per metric,
    # to create a mean value per metric.
    for df in dfs:
        df_list = {}
        for key in keys:
            key_column_names = [name for name in column_names if key == name.split("/")[-1]]
            key_columns = df[key_column_names]
            mean_trajectory = list(key_columns.mean(axis=1))
            df_list[key] = mean_trajectory

        for key, value in df_list.items():
            all_df_lists[key].append(value)
    return all_df_lists


# Plot the results for a given generated progress.csv file, found in your ray_results folder.
def plot_csvs_results(paths):
    path = paths[0]
    env, model_name = get_env_and_model_name_from_path(path)

    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    plots = []

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    reward_color = get_color_from_model_name(model_name)
    reward_means = [df.episode_reward_mean for df in dfs]
    plots.append(
        PlotData(
            timesteps_totals, reward_means, "Reward", "Mean collective episode reward", reward_color
        )
    )

    episode_len_means = [df.episode_len_mean for df in dfs]
    plots.append(
        PlotData(
            timesteps_totals, episode_len_means, "episode_length", "Mean episode length", "pink",
        )
    )

    metric_details = [
        PlotGraphics("cur_lr", "Learning rate", "purple"),
        PlotGraphics("policy_entropy", "Policy Entropy", "b"),
        PlotGraphics("policy_loss", "Policy loss", "r"),
        PlotGraphics("vf_loss", "Value function loss", "orange"),
        PlotGraphics("total_a3c_loss", "Total A3C loss", "yellow"),
        PlotGraphics("total_loss", "Total loss", "yellow"),
        PlotGraphics("moa_loss", "MOA loss", "black"),
        PlotGraphics("scm_loss", "SCM loss", "black"),
        PlotGraphics("social_influence_reward", "MOA reward", "black"),
        PlotGraphics("social_curiosity_reward", "Curiosity reward", "black"),
        PlotGraphics("cur_influence_reward_weight", "Influence reward weight", "orange"),
        PlotGraphics("cur_curiosity_reward_weight", "Curiosity reward weight", "orange"),
        PlotGraphics("extrinsic_reward", "Extrinsic reward", "g"),
        PlotGraphics("total_successes_mean", "Total successes", "black"),
        # Switch environment metrics
        PlotGraphics("switches_on_at_termination_mean", "Switches on at termination", "black"),
        PlotGraphics("total_pulled_on_mean", "Total switched on", "black"),
        PlotGraphics("total_pulled_off_mean", "Total switched off", "black"),
        PlotGraphics("timestep_first_switch_pull_mean", "Time at first switch pull", "black"),
        PlotGraphics("timestep_last_switch_pull_mean", "Time at last switch pull", "black"),
    ]

    extracted_data = extract_stats(dfs, [detail.column_name for detail in metric_details])
    for metric in metric_details:
        if metric.column_name in extracted_data:
            plots.append(
                PlotData(
                    timesteps_totals,
                    extracted_data[metric.column_name],
                    metric.column_name,
                    metric.legend_name,
                    metric.color,
                )
            )

    for plot in plots:

        def plot_fn():
            plot_single_category_result(
                plot.x_data,
                plot.y_data,
                plot.plot_graphics.color,
                plot.plot_graphics.legend_name,
                plot.plot_graphics.column_name,
            )

        try:
            plot_and_save(
                plot_fn, path, plot.plot_graphics.column_name + "_" + env + "_" + model_name
            )
        except ZeroDivisionError:
            pass


def get_color_from_model_name(model_name):
    name_to_color = {
        "baseline": "blue",
        "moa": "red",
        "scm": "orange",
        "scm no influence reward": "green",
    }
    name_lower = model_name.lower()
    if name_lower in name_to_color.keys():
        return name_to_color[name_lower]
    else:
        default_color = "darkgreen"
        print(
            "Warning: model name "
            + model_name
            + " has no default plotting color. Defaulting to "
            + default_color
        )
        return default_color


def get_env_and_model_name_from_path(path):
    category_path = path.split("/")[-3]
    if "baseline" in category_path:
        model_name = "baseline"
    elif "moa" in category_path:
        model_name = "MOA"
    elif "scm" in category_path:
        if "no_influence" in category_path:
            model_name = "SCM no influence reward"
        else:
            model_name = "SCM"
    else:
        raise NotImplementedError
    env = category_path.split("_")[0]
    return env, model_name


def get_experiment_rewards(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    env, model_name = get_env_and_model_name_from_path(paths[0])
    color = get_color_from_model_name(model_name)

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [
        [timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals
    ]

    most_timesteps = np.max(list(map(len, timesteps_totals)))
    x_min = np.nanmin(list(map(np.nanmin, timesteps_totals)))
    x_max = np.nanmax(list(map(np.nanmax, timesteps_totals)))
    # Cut off plotting at 5e8 steps
    x_max = min(x_max, 5.0)
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    for x, y, in zip(timesteps_totals, [df.episode_reward_mean for df in dfs]):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)
    reward_plotdata = PlotData(
        [interp_x] * 5, interpolated, "Mean collective reward", model_name, color,
    )
    return reward_plotdata, env


def change_color_luminosity(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Taken from https://stackoverflow.com/a/49601444
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    c = mc.cnames[color] if color in mc.cnames else color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_separate_results():
    # Plot separate experiment results
    for category_folder in get_all_subdirs(ray_results_path):
        print("Plotting category folder: " + category_folder.split("/")[-1])
        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)
        plot_csvs_results(csvs)


def plot_combined_results():
    # Plot combined experiment rewards per environment, for means per model
    env_rewards = {}
    for category_folder in get_all_subdirs(ray_results_path):
        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        experiment_rewards, env = get_experiment_rewards(csvs)
        if env not in env_rewards:
            env_rewards[env] = []
        env_rewards[env].append(experiment_rewards)

    for env, experiment_rewards in env_rewards.items():
        print("Plotting collective plot for environment: " + env)

        def plot_fn():
            plot_multiple_category_result(experiment_rewards)

        # Add filler to path which will be removed
        collective_env_path = "collective/filler/"
        plot_and_save(plot_fn, collective_env_path, env + "_collective_reward")


if __name__ == "__main__":
    print("Plotting separate results..")
    plot_separate_results()
    print("Plotting combined results..")
    plot_combined_results()
