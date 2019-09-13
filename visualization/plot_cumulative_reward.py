import ast
import io
import re
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from utility_funcs import get_all_subdirs

from config.config_parser import get_ray_results_path, get_plot_path


def smooth_list(score_list):
    # Smooth using gaussian filter
    score_list = gaussian_filter1d(score_list, np.std(score_list), mode='nearest')
    return score_list


# Plot the results for a given generated progress.csv file, found in your ray_results folder.
def plot_csv_results(path):
    # Remove curly braces and their contents, as they are nested and contain commas.
    # Commas are delimiters, and replacing them with quotechars does not help as they are nested.
    with open(path, 'r') as f:
        fo = io.StringIO() 
        data = f.readlines()
        fo.writelines(re.sub("/{([^}]*)}/", "", line) for line in data)
        fo.seek(0)

    try:
        print("Plotting " + path)
        df = pd.read_csv(fo, sep=",")
        # Set NaN values to 0, common at start of training due to ray behavior
        df = df.fillna(0)
        reward_min = df.episode_reward_min
        reward_max = df.episode_reward_max
        reward_mean = df.episode_reward_mean
        timesteps_total = df.timesteps_total

        episode_len_mean = df.episode_len_mean

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

        reward_min = smooth_list(reward_min)
        reward_max = smooth_list(reward_max)
        reward_mean = smooth_list(reward_mean)

        # Convert agent steps to 1e8 representation
        timesteps_total = [timestep / 1e8 for timestep in timesteps_total]

        # Clear plot to prevent slowdown when drawing multiple figures
        plt.clf()

        # Draw score
        plt.plot(timesteps_total, reward_mean, color='g', label='Mean reward')
        plt.plot(timesteps_total, smooth_list(entropy_list), color='b', label='Policy entropy', alpha=.2)
        plt.plot(timesteps_total, smooth_list(policy_loss_list), color='r', label='Policy loss', alpha=.2)
        plt.plot(timesteps_total, smooth_list(episode_len_mean), color='pink', label='Policy loss', alpha=.2)

        # Fill area between score
        plt.fill_between(timesteps_total, reward_min, reward_mean, color='g', alpha=.2, label='Min/max reward')
        plt.fill_between(timesteps_total, reward_max, reward_mean, color='g', alpha=.2)
        plt.legend()

        plt.xlabel('Agent steps (1e8)')
        plt.ylabel('cumulative reward')
        plt.ylim(top=reward_max.max() + 5, bottom=reward_min.min())

        # Strip path of all but last folder
        path_split = os.path.dirname(path).split('/')
        filename = path_split[-2] + "-" + path_split[-1]
        plt.savefig(plot_path + "/png/" + filename + ".png")
        plt.savefig(plot_path + "/eps/" + filename + ".eps")
    except:
        print("Could not plot file " + path)


ray_results_path = get_ray_results_path()
plot_path = get_plot_path()

if not os.path.exists(plot_path):
    os.mkdir(plot_path)
    os.mkdir(plot_path + "/png")
    os.mkdir(plot_path + "/eps")


category_folders = get_all_subdirs(ray_results_path)
experiment_folders = [get_all_subdirs(category_folder) for category_folder in category_folders]
# Flatten list
experiment_folders = [item for sublist in experiment_folders for item in sublist]
for i, subdir in enumerate(experiment_folders):
    print("Plotting " + str(i + 1) + "/" + str(len(experiment_folders)))
    plot_csv_results(subdir + "/progress.csv")
