import ray

from config.config_parser import get_ray_results_path, get_video_path
from utility_funcs import get_all_subdirs
from visualization.visualizer_rllib import visualize


def create_args(result_dir, checkpoint_num, video_path, video_filename):
    args = list()
    args.append(result_dir)
    args.append(checkpoint_num)
    args.append("--video-path")
    args.append(video_path)
    args.append("--video-filename")
    args.append(video_filename)
    args.append("--save-video")
    return args


def render():
    ray_results_path = get_ray_results_path()
    video_base_path = get_video_path() + "/"
    category_folders = get_all_subdirs(ray_results_path)
    for category_folder in category_folders:
        experiment_folders = get_all_subdirs(category_folder)
        for i, experiment_folder in enumerate(experiment_folders):
            print("Rendering experiment" + str(i + 1) + "/" + str(len(experiment_folders)))
            checkpoint_folders = get_all_subdirs(experiment_folder)
            video_path = video_base_path + experiment_folder.split("/")[-1]
            for j, checkpoint_folder in enumerate(checkpoint_folders):
                print("Rendering checkpoint" + str(j + 1) + "/" + str(len(checkpoint_folders)))
                checkpoint_number = checkpoint_folder.split("_")[-1]
                video_name = checkpoint_number
                args = create_args(experiment_folder, checkpoint_number, video_path, video_name)
                visualize(args)
                ray.shutdown()


render()
