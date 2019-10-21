import ast
import io
import re
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from utility_funcs import get_all_subdirs, get_all_files

from config.config_parser import get_ray_results_path, get_plot_path


class PlotDetails(object):
    def __init__(self, column_name, legend_name, color):
        self.column_name = column_name
        self.legend_name = legend_name
        self.color = color


class PlotData(object):
    def __init__(self, x_data, y_data, column_name, legend_name, color):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_details = PlotDetails(column_name, legend_name, color)


def plot_and_save(fn, path, file_name_addition):
    # Clear plot to prevent slowdown when drawing multiple figures
    plt.clf()
    fn()
    plt.legend()
    # Strip path of all but last folder
    path_split = os.path.dirname(path).split('/')
    filename = path_split[-2] + "-" + path_split[-1]
    plt.savefig(plot_path + "/png/" + filename + "-" + file_name_addition + ".png")
    plt.savefig(plot_path + "/eps/" + filename + "-" + file_name_addition + ".eps")


def plot_with_mean(x_lists, y_lists, color, y_label):
    most_timesteps = np.max(list(map(len, x_lists)))
    x_min = np.min(list(map(np.min, x_lists)))
    x_max = np.max(list(map(np.max, x_lists)))
    y_min = np.min(list(map(np.min, y_lists)))
    y_max = np.max(list(map(np.max, y_lists)))
    interp_x = np.linspace(x_min, x_max, most_timesteps)
    interpolated = []
    for x, y in zip(x_lists, y_lists):
        interp_y = np.interp(interp_x, x, y, left=np.nan, right=np.nan)
        interpolated.append(interp_y)
        plt.plot(interp_x, interp_y, color=color, alpha=.2)
    means = np.nanmean(interpolated, axis=0)
    plt.plot(interp_x, means, color=color, label=y_label, alpha=1)

    plt.xlabel('Environment steps (1e8)')
    plt.ylabel(y_label)
    plt.ylim(top=y_max + 5)


def extract_mean_agent_stats(dfs, keys):
    all_df_lists = {}
    for key in keys:
        all_df_lists[key] = []

    for df in dfs:
        info = list(df['info'])
        learner_dicts = [ast.literal_eval(info_line)['learner'] for info_line in info]
        df_list = {}
        for key in keys:
            df_list[key] = []
        for learner_dict in learner_dicts:
            summed_values = {}
            for key in keys:
                summed_values[key] = 0
            for agent, agent_stats in learner_dict.items():
                for key in keys:
                    value = agent_stats[key]
                    summed_values[key] += value
            for key, value in summed_values.items():
                mean = value / len(learner_dict)
                df_list[key].append(mean)
        for key, value in df_list.items():
            all_df_lists[key].append(value)
    return all_df_lists


# Plot the results for a given generated progress.csv file, found in your ray_results folder.
def plot_csvs_results(paths):
    # Remove curly braces and their contents, as they are nested and contain commas.
    # Commas are delimiters, and replacing them with quotechars does not help as they are nested.
    dfs = []
    for path in paths:
        with open(path, 'r') as f:
            fo = io.StringIO()
            data = f.readlines()
            fo.writelines(re.sub("/{([^}]*)}/", "", line) for line in data)
            fo.seek(0)
            df = pd.read_csv(fo, sep=",")
            # Set NaN values to 0, common at start of training due to ray behavior
            df = df.fillna(0)
            dfs.append(df)

    plots = []

    # Convert environment steps to 1e8 representation
    timesteps_totals = [df.timesteps_total for df in dfs]
    timesteps_totals = [[timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals]

    reward_means = [df.episode_reward_mean for df in dfs]
    reward_means = [gaussian_filter1d(reward_mean, 1, mode='nearest') for reward_mean in reward_means]
    plots.append(PlotData(timesteps_totals, reward_means, 'reward', 'Mean episode reward', 'g'))

    episode_len_means = [df.episode_len_mean for df in dfs]
    plots.append(PlotData(timesteps_totals, episode_len_means, 'episode_length', 'Mean episode length', 'pink'))

    agent_metric_details = [PlotDetails('policy_entropy', 'Mean Entropy', 'b'),
                            PlotDetails('policy_loss', 'Policy loss', 'r'),
                            PlotDetails('vf_loss', 'Value function loss', 'r'),
                            PlotDetails('total_a3c_loss', 'Total A3C loss', 'r'),
                            PlotDetails('aux_loss', 'Auxiliary task loss', 'black'),
                            PlotDetails('total_aux_reward', 'Policy loss', 'black'),
                            PlotDetails('total_successes', 'Total successes', 'black'),
                            PlotDetails('switches_on_at_termination', 'Switches on at termination', 'black'),
                            PlotDetails('total_pulled_on', 'Total switched on by agent', 'black'),
                            PlotDetails('total_pulled_off', 'Total switched off by agent', 'black'),
                            PlotDetails('timestep_first_switch_pull', 'Time at first switch pull', 'black'),
                            PlotDetails('timestep_last_switch_pull', 'Time at last switch pull', 'black')
                            ]

    agent_stats = extract_mean_agent_stats(dfs, [detail.column_name for detail in agent_metric_details])
    for metric in agent_metric_details:
        plots.append(PlotData(timesteps_totals,
                              agent_stats[metric.column_name],
                              metric.column_name,
                              metric.legend_name,
                              metric.color))

    path = paths[0]

    for plot in plots:
        plot_fn = lambda: plot_with_mean(plot.x_data, plot.y_data,
                                         plot.plot_details.color,
                                         plot.plot_details.legend_name)
        plot_and_save(plot_fn, path, plot.plot_details.column_name)


ray_results_path = get_ray_results_path()
plot_path = get_plot_path()

if not os.path.exists(plot_path):
    os.mkdir(plot_path)
    os.mkdir(plot_path + "/png")
    os.mkdir(plot_path + "/eps")

category_folders = get_all_subdirs(ray_results_path)
experiment_folders = [get_all_subdirs(category_folder) for category_folder in category_folders]
# Sort by device
for experiment_folder in experiment_folders:
    csvs = []
    for subdir in experiment_folder:
        csvs.append(subdir + "/progress.csv")
    plot_csvs_results(csvs)
