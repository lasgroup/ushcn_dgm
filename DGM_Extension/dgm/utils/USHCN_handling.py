from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from jax import numpy as jnp


def format_data_for_dgm(times, values, augmented_states=False):
    """
    Takes return values of extract_time_and_value_arrays_from_GRU_data and reformats it so that it is
    consistent with the DGM data format.
    """
    dgm_times = []
    dgm_values = []

    if augmented_states:
        n_states = 6
    else:
        n_states = 5

    for value_dim in range(n_states):
        dgm_times_of_this_dimension = []
        dgm_values_of_this_dimension = []
        for trajectory_id in range(len(times)):
            order = np.argsort(times[trajectory_id][value_dim])
            dgm_times_of_this_dimension.append(jnp.asarray(
                times[trajectory_id][value_dim][order]
            ))
            dgm_values_of_this_dimension.append(jnp.asarray(
                values[trajectory_id][value_dim][order]
            ))
        dgm_times.append(dgm_times_of_this_dimension)
        dgm_values.append(dgm_values_of_this_dimension)
    return dgm_times, dgm_values


def extract_trajectories_with_more_than_n_obs_per_dim(n: int,
                                                      train_times,
                                                      train_observations,
                                                      initial_conditions=None
                                                      ):
    """
    Assume format as returned by format_data_for_dgm
    """
    list_of_unwanted = []
    num_dimensions = len(train_times)
    num_trajectories = len(train_times[0])
    for traj_id in range(num_trajectories):
        for dim_id in range(num_dimensions):
            if len(train_times[dim_id][traj_id]) < n:
                list_of_unwanted.append(traj_id)
                break

    # remove list of undesirables
    if initial_conditions:
        for unwanted_id in sorted(list_of_unwanted, reverse=True):
            for dim_id in range(num_dimensions):
                del train_times[dim_id][unwanted_id]
                del train_observations[dim_id][unwanted_id]
            del initial_conditions[unwanted_id]
        return train_times, train_observations, initial_conditions
    else:
        for unwanted_id in sorted(list_of_unwanted, reverse=True):
            for dim_id in range(num_dimensions):
                del train_times[dim_id][unwanted_id]
                del train_observations[dim_id][unwanted_id]
        return train_times, train_observations


def extract_trajectories_with_ics_within_slack(ic_slack,
                                               times,
                                               observations
                                               ):
    """
    Assume format as returned by format_data_for_dgm
    """
    if not ic_slack > 0:
        return times, observations

    unwanted_indices = []
    for traj_id in range(len(times[0])):
        for dim_id in range(len(times)):
            if min(times[dim_id][traj_id]) > ic_slack:
                unwanted_indices.append(traj_id)
                break

    print(len(unwanted_indices))
    print("Created list of unwanted indices")

    for unwanted_id in sorted(unwanted_indices, reverse=True):
        for dim_id in range(len(times)):
            del times[dim_id][unwanted_id]
            del observations[dim_id][unwanted_id]

    return times, observations
    # test with max([max([min(time) for time in traj_of_this_dim]) for traj_of_this_dim in times])


def extract_trajectories_with_more_than_n_train_obs_per_dim_and_3_val_obs(n: int,
                                                                          train_times, train_observations,
                                                                          test_times, test_observations,
                                                                          initial_conditions=None
                                                                          ):
    """
    Assume format as returned by format_data_for_dgm
    """
    list_of_unwanted = []
    num_dimensions = len(train_times)
    num_trajectories = len(train_times[0])
    for traj_id in range(num_trajectories):
        for dim_id in range(num_dimensions):
            if len(train_times[dim_id][traj_id]) < n or len(test_times[dim_id][traj_id]) < 3:
                list_of_unwanted.append(traj_id)
                break
    # remove list of undesirables
    if initial_conditions:
        for unwanted_id in sorted(list_of_unwanted, reverse=True):
            for dim_id in range(num_dimensions):
                del train_times[dim_id][unwanted_id]
                del train_observations[dim_id][unwanted_id]
                del test_times[dim_id][unwanted_id]
                del test_observations[dim_id][unwanted_id]
            del initial_conditions[unwanted_id]
        return train_times, train_observations, test_times, test_observations, initial_conditions
    else:
        for unwanted_id in sorted(list_of_unwanted, reverse=True):
            for dim_id in range(num_dimensions):
                del train_times[dim_id][unwanted_id]
                del train_observations[dim_id][unwanted_id]
                del test_times[dim_id][unwanted_id]
                del test_observations[dim_id][unwanted_id]
        return train_times, train_observations, test_times, test_observations


def plot_reformatted_data(times, values, path_to_plots, augmented=False):
    """
    Legend has been created using
    >>> a = np.load("label_id_mapping.npy", allow_pickle=True)
    >>> a
    array({'PRCP': 0, 'SNOW': 1, 'SNWD': 2, 'TMAX': 3, 'TMIN': 4},
      dtype=object)
    in the original GRU preprocessing repo and then crossreferencing it with
    https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/ndp070.html#files
    """

    if augmented:
        legend = ["precipitation", "snowfall", "snowdepth", "max_temp", "min_temp", "daylength"]
        n_states = 6
    else:
        legend = ["precipitation", "snowfall", "snowdepth", "max_temp", "min_temp"]
        n_states = 5
    path_to_plots = os.path.join(path_to_plots, "raw_trajectories")
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    for trajectory_id in range(len(times)):
        print("Plotting trajectory {}".format(trajectory_id))
        plt.figure()
        for value_dim in range(n_states):
            order = np.argsort(times[trajectory_id][value_dim])
            plt.plot(times[trajectory_id][value_dim][order],
                     values[trajectory_id][value_dim][order],
                     marker='.',
                     )
        plt.legend(legend)
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_plots, "trajectory{}".format(trajectory_id)))
        plt.close()


def extract_time_and_value_arrays_from_GRU_data(gru_dataframe, augmented=False):
    """
    returns two lists, times and values, which should be called as "times[trajectory_id, value_dimension]"
    """
    n_states = 5
    if augmented:
        n_states = 6

    n_trajectories = len(pd.unique(gru_dataframe.ID))
    all_times = []
    all_values = []
    for trajectory_id in range(n_trajectories):
        current_trajectory = gru_dataframe.loc[gru_dataframe.ID==trajectory_id]
        all_times_of_this_trajectory = []
        all_values_of_this_trajectory = []
        for state_id in range(n_states):
            times_of_this_dimension = \
                current_trajectory.loc[current_trajectory["Mask_{}".format(state_id)]==1, "Time"]
            values_of_this_dimension = \
                current_trajectory.loc[current_trajectory["Mask_{}".format(state_id)]==1, "Value_{}".format(state_id)]
            all_times_of_this_trajectory.append(times_of_this_dimension.reset_index(drop=True))
            all_values_of_this_trajectory.append(values_of_this_dimension.reset_index(drop=True))
        all_times.append(all_times_of_this_trajectory)
        all_values.append(all_values_of_this_trajectory)

    return all_times, all_values


def extract_time_value_and_position_arrays_from_GRU_data(gru_dataframe, position_extractor, augmented=False):
    """
    returns three lists, times, values, initial_conditions, which should be called as "times[trajectory_id, value_dimension]"
    """
    all_times, all_values = extract_time_and_value_arrays_from_GRU_data(gru_dataframe, augmented=augmented)
    all_ics = []
    n_trajectories = len(all_times)

    for trajectory_id in range(n_trajectories):
        position, elevation = position_extractor.get_position_and_elevation_from_index(trajectory_id)
        current_ic = np.asarray(position + elevation)
        all_ics.append(current_ic)

    return all_times, all_values, all_ics


def load_raw_data(path_to_data):
    original_data = pd.read_csv(path_to_data)
    return original_data



