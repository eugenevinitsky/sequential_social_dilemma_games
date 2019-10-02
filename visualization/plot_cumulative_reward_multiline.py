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


def plot_using_lambda(fn, path, file_name_addition):
    # Clear plot to prevent slowdown when drawing multiple figures
    plt.clf()
    fn()
    plt.legend()
    # Strip path of all but last folder
    path_split = os.path.dirname(path).split('/')
    filename = path_split[-2] + "-" + path_split[-1]
    plt.savefig(plot_path + "/png/" + filename + "-" + file_name_addition + ".png")
    plt.savefig(plot_path + "/eps/" + filename + "-" + file_name_addition + ".eps")


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

    reward_means = [df.episode_reward_mean for df in dfs]
    timesteps_totals = [df.timesteps_total for df in dfs]

    episode_len_means = [df.episode_len_mean for df in dfs]

    policy_loss_lists = []
    entropy_lists = []
    for df in dfs:
        info = list(df['info'])
        learner_dicts = [ast.literal_eval(info_line)['learner'] for info_line in info]
        policy_loss_list = []
        entropy_list = []
        for learner_dict in learner_dicts:
            average_loss = 0
            average_entropy = 0
            for agent, agent_stats in learner_dict.items():
                policy_loss = agent_stats['policy_loss']
                entropy = agent_stats['policy_entropy']
                average_loss += policy_loss
                average_entropy += entropy
            average_loss = average_loss / len(learner_dict)
            average_entropy = average_entropy / len(learner_dict)
            policy_loss_list.append(average_loss)
            entropy_list.append(average_entropy)
        policy_loss_lists.append(policy_loss_list)
        entropy_lists.append(entropy_list)

    #reward_min = gaussian_filter1d(reward_min, 1, mode='nearest')
    #reward_max = gaussian_filter1d(reward_max, 1, mode='nearest')
    reward_means = [gaussian_filter1d(reward_mean, 1, mode='nearest') for reward_mean in reward_means]

    # Assert that timesteps_totals are equal
    lengths = list(map(len, timesteps_totals))
    transposed = np.transpose(timesteps_totals)
    a = list(column.tolist() == transposed[0].tolist() for column in transposed)
    if not np.all(length == lengths[0] for length in lengths) \
            and not np.all(column == transposed[0] for column in transposed):
        print("Could not plot " + paths[0])
        return

    # Convert environment steps to 1e8 representation
    timesteps_totals = [[timestep / 1e8 for timestep in timesteps_total] for timesteps_total in timesteps_totals]

    def plot_with_mean(input_data, color, y_label):
        means = np.mean(input_data, axis=1)
        plt.plot(timesteps_totals[0], means, color=color, label=y_label, alpha=1)
        for l in input_data:
            plt.plot(timesteps_totals[0], l, color=color, label=y_label, alpha=.2)

        plt.xlabel('Environment steps (1e8)')
        plt.ylabel(y_label)
        plt.ylim(top=input_data.max() + 5, bottom=input_data.min())

    plot_reward = lambda: plot_with_mean(reward_means, 'g', 'Mean episode reward')
    plot_entropy = lambda: plot_with_mean(entropy_lists, 'b', 'Mean entropy')
    plot_policy_loss = lambda: plot_with_mean(policy_loss_lists, 'r', 'Policy loss')
    plot_mean_episode_length = lambda: plot_with_mean(episode_len_means, 'pink', 'Mean episode length')

    path = paths[0]

    plot_using_lambda(plot_reward, path, 'reward')
    plot_using_lambda(plot_entropy, path, 'entropy')
    plot_using_lambda(plot_policy_loss, path, 'policy_loss')
    plot_using_lambda(plot_mean_episode_length, path, 'episode_length')


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
    meteor = []
    pablo = []
    for subdir in experiment_folder:
        subdir_files = get_all_files(subdir)
        for subdir_file in subdir_files:
            if 'meteor' in subdir_file:
                meteor.append(subdir + "/progress.csv")
                break
            elif 'pablo' in subdir_file:
                pablo.append(subdir + "/progress.csv")
                break
    print("Plotting meteor")
    plot_csvs_results(meteor)

    print("Plotting pablo")
    plot_csvs_results(pablo)
