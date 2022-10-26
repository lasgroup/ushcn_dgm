from dgm.utils.USHCN_handling import extract_time_and_value_arrays_from_GRU_data, plot_reformatted_data, load_raw_data
from matplotlib import pyplot as plt
import os
import numpy as np

PATH_TO_DATA = "/home/philippe/data/small_chunked_sporadic_with_daylength.csv"
PATH_TO_PLOTS = "/home/philippe/ultimate_thesis_plots/USHCN"


def main():
    plot_without_augmentation()
    plot_with_augmentation()


def plot_without_augmentation():
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    original_data = load_raw_data(PATH_TO_DATA)
    times, values = extract_time_and_value_arrays_from_GRU_data(original_data, augmented=False)

    fig, axes = plt.subplots(5, 3, figsize=(10, 13))

    titles = ["Station 1", "Station 2", "Station 3"]
    ylabels = ["precipitation", 'snowfall', 'snow depth', 'max temperature', 'min temperature']

    for traj_id in range(3):
        for dim_id in range(5):
            axes[dim_id][traj_id].plot(times[traj_id][dim_id], values[traj_id][dim_id], 'r.', markersize=10)
            axes[dim_id][0].set_ylabel(ylabels[dim_id])
            axes[dim_id][traj_id].set_xlabel("time")
            axes[0][traj_id].set_title(titles[traj_id])

    fig.tight_layout()
    fig.savefig(os.path.join(PATH_TO_PLOTS, "no_augmentation.png"), dpi=300)
    fig.show()


def plot_with_augmentation():
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    original_data = load_raw_data(PATH_TO_DATA)
    times, values = extract_time_and_value_arrays_from_GRU_data(original_data, augmented=True)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    titles = ["Station 1", "Station 2", "Station 3"]
    ylabels = ['max temperature', 'min temperature', 'day length']

    time_cutoff = 50

    for traj_id in range(3):
        for dim_id in range(3):
            curr_times = np.array(times[traj_id][dim_id+3])
            indices = np.where(curr_times <= time_cutoff)
            curr_times = curr_times[indices]
            curr_values = np.array(values[traj_id][dim_id+3])
            curr_values = curr_values[indices]
            axes[dim_id][traj_id].plot(curr_times, curr_values, 'r.', markersize=10)
            axes[dim_id][0].set_ylabel(ylabels[dim_id])
            axes[dim_id][traj_id].set_xlabel("time")
            axes[0][traj_id].set_title(titles[traj_id])

    fig.tight_layout()
    fig.savefig(os.path.join(PATH_TO_PLOTS, "with_augmentation.png"), dpi=300)
    fig.show()


if __name__ == "__main__":
    main()
