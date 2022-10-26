from dgm.utils.representatives import  InitialConditionExtractorType
from jax import numpy as jnp


def get_initial_condition_extractor(extractor_type: InitialConditionExtractorType):
    if extractor_type == InitialConditionExtractorType.NEAREST_NEIGHBOR:
        return extract_nearest_neighbor_initial_conditions
    if extractor_type == InitialConditionExtractorType.NONE:
        return extract_none
        # return wrong_extract_nearest_neighbor_initial_conditions


def extract_none(times, observations):
    return None


def extract_nearest_neighbor_initial_conditions(times, observations):
    num_dimensions = len(times)
    num_trajectories = len(times[0])

    initial_conditions = []

    for traj_id in range(num_trajectories):
        initial_conditions_for_this_dimension = []
        for dim_id in range(num_dimensions):
            min_time_index = jnp.argmin(times[dim_id][traj_id])
            initial_conditions_for_this_dimension.append(observations[dim_id][traj_id][min_time_index])
        initial_conditions.append(jnp.asarray(initial_conditions_for_this_dimension))

    return initial_conditions


def wrong_extract_nearest_neighbor_initial_conditions(times, observations):
    num_dimensions = len(times)
    num_trajectories = len(times[0])

    initial_conditions = []

    for traj_id in range(num_trajectories):
        initial_conditions_for_this_dimension = []
        for dim_id in range(num_dimensions):
            initial_conditions_for_this_dimension.append(observations[dim_id][traj_id][0])
        initial_conditions.append(jnp.asarray(initial_conditions_for_this_dimension))

    return initial_conditions
