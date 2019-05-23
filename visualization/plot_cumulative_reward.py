import io
import re
import os.path
import pandas as pd
import matplotlib.pyplot as plt

from config.config_parser import get_ray_results_path, get_plot_path

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
        df = pd.read_csv(fo, sep=",")
        reward_min = df.episode_reward_min
        reward_max = df.episode_reward_max
        reward_mean = df.episode_reward_mean
        timesteps_total = df.timesteps_total

        # Convert to 1e8 representation
        # Multiply by the number of agents (always 5) to get amount of agent steps
        timesteps_total = [timestep / 1e8 * 5 for timestep in timesteps_total]

        plt.clf()
        plt.plot(timesteps_total, reward_min, color='r')
        plt.plot(timesteps_total, reward_mean, color='b')
        plt.plot(timesteps_total, reward_max, color='g')
        plt.xlabel('Agent steps (1e8)')
        plt.ylabel('cumulative reward')
        plt.ylim(bottom=-10)

        # Strip path of all but last folder
        path_split = os.path.dirname(path).split('/')
        filename = path_split[-2] + '-' + path_split[-1] + '.png'
        plt.savefig(plot_path + "/" + filename)
    except:
        print("Could not plot file " + path)


def get_all_subdirs(path):
    return [path + '/' + d for d in os.listdir(path) if os.path.isdir(path + '/' + d)]


def get_all_files(path):
    return [path + '/' + d for d in os.listdir(path) if not os.path.isdir(path + '/' + d)]


ray_results_path = get_ray_results_path()
plot_path = get_plot_path()

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

for folder in get_all_subdirs(ray_results_path):
    for subdir in get_all_subdirs(folder):
        plot_csv_results(subdir + "/progress.csv")
