from dgm.utils.USHCN_handling import extract_time_and_value_arrays_from_GRU_data, plot_reformatted_data, load_raw_data
from matplotlib import pyplot as plt
import os
import numpy as np

# PATH_TO_DATA = "~/data/small_chunked_sporadic.csv"
PATH_TO_DATA = "~/data/small_chunked_sporadic_with_daylength.csv"
PATH_TO_PLOTS = "/home/philippe/GRU_Playground/length_analysis"
INTERESTING_DIMS = [3, 4]  # {'PRCP': 0, 'SNOW': 1, 'SNWD': 2, 'TMAX': 3, 'TMIN': 4}


def main():
    original_data = load_raw_data(PATH_TO_DATA)
    times, values = extract_time_and_value_arrays_from_GRU_data(original_data)
    # data_length_analysis(times, values)
    # gap_analysis(times, values)
    initial_condition_gap_analysis(times)


def data_length_analysis(times, values, min_length=30):
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)
    # plot min length per trajectory
    min_lengths_per_trajectory = []
    for traj_id in range(len(times)):
        traj_lengths_per_dim_for_this_trajectory = []
        for dim_id in range(len(times[0])):
            if dim_id in INTERESTING_DIMS:
                traj_lengths_per_dim_for_this_trajectory.append(times[traj_id][dim_id].size)
            else:
                print("Do not care about dim {}".format(dim_id))
        min_lengths_per_trajectory.append(min(traj_lengths_per_dim_for_this_trajectory))
    plt.figure()
    plt.hist(min_lengths_per_trajectory, bins=100)
    plt.title("minimal length per trajectory")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "min_length.png"))
    plt.show()
    plt.close()

    small_min_lengths = []
    for curr_len in min_lengths_per_trajectory:
        if curr_len < min_length:
            small_min_lengths.append(curr_len)
    plt.figure()
    plt.hist(small_min_lengths, bins=100)
    plt.title("small minimal lengths per trajectory")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "small_min_length.png"))
    plt.show()
    plt.close()

    # plot small trajectories
    interesting_trajectory_lengths = [3, 5, 7, 8, 20, 22, 23, 26, 27, 28, 29]
    filtered_times = []
    filtered_values = []
    for int_len in interesting_trajectory_lengths:
        trajectory_id = min_lengths_per_trajectory.index(int_len)
        filtered_times.append(times[trajectory_id])
        filtered_values.append(values[trajectory_id])
    plot_reformatted_data(filtered_times, filtered_values,
                          os.path.join(PATH_TO_PLOTS, "min_length_traj"))


def gap_analysis(times, values, n_largest_gaps=20):
    # calculate max gap
    max_gaps_per_trajectory = []
    for traj_id in range(len(times)):
        max_gaps_for_this_traj = []
        for dim_id in range(len(times[0])):
            if len(times[traj_id][dim_id]) < 2:
                gaps_for_this_dim = [0]
            else:
                curr_times = times[traj_id][dim_id].to_numpy()
                curr_times.sort()
                gaps_for_this_dim = curr_times[1:] - curr_times[:-1]
            if dim_id in INTERESTING_DIMS:
                max_gaps_for_this_traj.append(max(gaps_for_this_dim))
        max_gaps_per_trajectory.append(max(max_gaps_for_this_traj))

    # plot max gap histogram
    plt.figure()
    plt.hist(max_gaps_per_trajectory, bins=100)
    plt.title("max time step per trajectory")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "all_max_gaps.png"))
    plt.show()
    plt.close()

    # plot interesting max gap histos
    sorted_max_gaps = max_gaps_per_trajectory.copy()
    sorted_max_gaps.sort()
    large_gaps = sorted_max_gaps[-n_largest_gaps:]
    plt.figure()
    plt.hist(large_gaps, bins=100)
    plt.title("large max time steps")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "large_max_gaps.png"))
    plt.show()
    plt.close()

    # plot trajectories with large gaps
    filtered_times = []
    filtered_values = []
    for large_gap in large_gaps:
        traj_id = max_gaps_per_trajectory.index(large_gap)
        filtered_times.append(times[traj_id])
        filtered_values.append(values[traj_id])
    plot_reformatted_data(filtered_times, filtered_values,
                          os.path.join(PATH_TO_PLOTS, "gap_trajectories"))


def initial_condition_gap_analysis(times, n_smallest_times=800):
    # calculate max gap
    count = 0
    min_times_per_trajectory = []
    for traj_id in range(len(times)):
        min_times_for_this_traj = []
        for dim_id in range(len(times[0])):
            if dim_id in INTERESTING_DIMS:
                if len(times[traj_id][dim_id]) < 1:
                    min_times_for_this_traj.append(200)
                    count += 1
                else:
                    curr_times = times[traj_id][dim_id].to_numpy()
                    min_times_for_this_traj.append(min(curr_times))
        min_times_per_trajectory.append(max(min_times_for_this_traj))

    min_times_array = np.asarray(min_times_per_trajectory)

    # plot max gap histogram
    plt.figure()
    plt.hist(min_times_per_trajectory, bins=100)
    plt.title("initial condition times")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "all_ic_times.png"))
    plt.show()
    plt.close()

    # plot interesting max gap histos
    sorted_min_times = min_times_per_trajectory.copy()
    sorted_min_times.sort()
    small_times = sorted_min_times[:n_smallest_times]
    plt.figure()
    plt.hist(small_times, bins=100)
    plt.title("ic times small")
    plt.savefig(os.path.join(PATH_TO_PLOTS, "small_ic_times.png"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
