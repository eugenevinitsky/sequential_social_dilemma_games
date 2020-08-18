import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    epspath = plot_path + "/eps/" + path_split[-2] + "/"
    pngfile = pngpath + file_name_addition + ".png"
    epsfile = epspath + file_name_addition + ".eps"
    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    if not os.path.exists(epspath):
        os.makedirs(epspath)
    plt.savefig(pngfile)
    plt.savefig(epsfile)


def plot_multiple_category_result(plotdata_list):
    for plotdata in plotdata_list:
        plot_single_category_result(
            plotdata.x_data,
            plotdata.y_data,
            plotdata.plot_graphics.color,
            plotdata.plot_graphics.legend_name,
            plotdata.plot_graphics.column_name,
            with_mean=False,
            with_label=True,
        )


def plot_single_category_result(
    x_lists, y_lists, color, legend_name, y_label_name, with_mean=True, with_label=False
):
    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.nanmin(list(map(np.nanmin, x_lists)))
    x_max = np.nanmax(list(map(np.nanmax, x_lists)))
    y_max = np.nanmax(list(map(np.nanmax, y_lists)))
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    for x, y in zip(x_lists, y_lists):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)
        light_color = change_color_luminosity(color, 0.5) if with_mean else color
        label_name = legend_name if with_label else None
        plt.plot(interp_x, interp_y, color=light_color, label=label_name)
    if with_mean:
        means = np.nanmean(interpolated, axis=0)
        plt.plot(interp_x, means, color=color, label=legend_name)

    plt.xlabel("Environment steps (1e8)")
    plt.ylabel(y_label_name)
    bottom = 0 if "reward" in y_label_name else None
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
    env = path.split("/")[-3].split("_")[0]
    model = path.split("/")[-3].split("_")[1]

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

    reward_color = get_color_from_model_name(model)
    reward_means = [df.episode_reward_mean for df in dfs]
    plots.append(
        PlotData(timesteps_totals, reward_means, "reward", "Mean episode reward", reward_color)
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
            plot_and_save(plot_fn, path, plot.plot_graphics.column_name + "_" + env + "_" + model)
        except ZeroDivisionError:
            pass


def get_color_from_model_name(model_name):
    name_to_color = {
        "baseline": "blue",
        "moa": "red",
        "scm": "orange",
    }
    name_lower = model_name.lower()
    if name_lower in name_to_color.keys():
        return name_to_color[name_lower]
    else:
        raise NotImplementedError


def get_experiment_reward_means(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        dfs.append(df)

    category_path = paths[0].split("/")[-3]
    if "baseline" in category_path:
        model_name = "baseline"
    elif "moa" in category_path:
        model_name = "MOA"
    elif "scm" in category_path:
        model_name = "SCM"
    else:
        raise NotImplementedError
    color = get_color_from_model_name(model_name)

    env = category_path.split("_")[0]

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
    means = np.nanmean(interpolated, axis=0)
    mean_plotdata = PlotData([interp_x], [means], "Collective reward", model_name, color)
    return mean_plotdata, env


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
        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)
        plot_csvs_results(csvs)


def plot_combined_results():
    # Plot combined experiment rewards per environment, with means per model
    env_means = {}
    for category_folder in get_all_subdirs(ray_results_path):
        csvs = []
        experiment_folders = get_all_subdirs(category_folder)
        for experiment_folder in experiment_folders:
            csv_path = experiment_folder + "/progress.csv"
            if os.path.getsize(csv_path) > 0:
                csvs.append(csv_path)

        reward_means, env = get_experiment_reward_means(csvs)
        if env not in env_means:
            env_means[env] = []
        env_means[env].append(reward_means)

    for env, reward_means in env_means.items():

        def plot_fn():
            plot_multiple_category_result(reward_means)

        collective_env_path = "collective/filler/"
        plot_and_save(plot_fn, collective_env_path, env + "_collective_reward")


if __name__ == "__main__":
    plot_separate_results()
    plot_combined_results()
