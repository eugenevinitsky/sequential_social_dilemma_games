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
        self.plot_details = PlotGraphics(column_name, legend_name, color)


def plot_and_save(fn, path, file_name_addition):
    # Clear plot to prevent slowdown when drawing multiple figures
    plt.clf()
    fn()
    plt.legend()
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


def plot_with_mean(x_lists, y_lists, color, y_label):
    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.nanmin(list(map(np.nanmin, x_lists)))
    x_max = np.nanmax(list(map(np.nanmax, x_lists)))
    y_min = np.nanmin(list(map(np.nanmin, y_lists)))
    y_max = np.nanmax(list(map(np.nanmax, y_lists)))
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    for x, y in zip(x_lists, y_lists):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)
        plt.plot(interp_x, interp_y, color=color, alpha=0.2)
    means = np.nanmean(interpolated, axis=0)
    plt.plot(interp_x, means, color=color, label=y_label, alpha=1)

    plt.xlabel("Environment steps (1e8)")
    plt.ylabel(y_label)
    bottom = 0 if "reward" in y_label else None
    plt.ylim(bottom=bottom, top=y_max + (y_max - y_min) / 100)
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
    # Remove curly braces and their contents, as they are nested and contain commas.
    # Commas are delimiters, and replacing them with quotechars does not help as they are nested.
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

    reward_means = [df.episode_reward_mean for df in dfs]
    plots.append(PlotData(timesteps_totals, reward_means, "reward", "Mean episode reward", "g"))

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
        PlotGraphics("moa_loss", "MOA loss", "black"),
        PlotGraphics("total_influence_reward", "MOA reward", "black"),
        PlotGraphics("extrinsic_reward", "Extrinsic reward", "g"),
        PlotGraphics("total_successes_mean", "Total successes", "black"),
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
    path = paths[0]

    for plot in plots:

        def plot_fn():
            plot_with_mean(
                plot.x_data, plot.y_data, plot.plot_details.color, plot.plot_details.legend_name,
            )

        try:
            plot_and_save(plot_fn, path, plot.plot_details.column_name)
        except ZeroDivisionError:
            pass

    def plot_losses():
        for plot in plots:
            if "loss" in plot.plot_details.column_name or "reward" == plot.plot_details.column_name:
                if len(plot.y_data) > 0:
                    plot_with_mean(
                        plot.x_data,
                        plot.y_data,
                        plot.plot_details.color,
                        plot.plot_details.legend_name,
                    )


for category_folder in get_all_subdirs(ray_results_path):
    csvs = []
    experiment_folders = get_all_subdirs(category_folder)
    for experiment_folder in experiment_folders:
        csv_path = experiment_folder + "/progress.csv"
        if os.path.getsize(csv_path) > 0:
            csvs.append(csv_path)
    plot_csvs_results(csvs)
