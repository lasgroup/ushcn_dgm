import jax.numpy as jnp
from typing import List, Optional


def create_equidistant_times(start_time: float,
                             end_time: float,
                             num_points_on_trajectory_per_dimension: List[int],
                             num_trajectories: int,
                             num_dimensions: Optional[int] = None):
    """
    Returns lists of times, which should be called as
    times[trajectory_id][dimension_id] and returns a 1D jnp array with length num_points_on_trajectory
    """
    if num_dimensions:
        num_points_on_trajectory_per_dimension = num_points_on_trajectory_per_dimension*jnp.ones([num_dimensions])
        num_points_on_trajectory_per_dimension = num_points_on_trajectory_per_dimension.astype(int)
    all_times = []
    for dim_id in range(len(num_points_on_trajectory_per_dimension)):
        all_times.append(
            [
                jnp.linspace(start_time, end_time, num_points_on_trajectory_per_dimension[dim_id])
                for _ in range(num_trajectories)
            ]
        )

    return all_times


if __name__ == "__main__":
    a = create_equidistant_times(start_time=0,
                                 end_time=1,
                                 num_points_on_trajectory=5,
                                 num_dimensions=2,
                                 num_trajectories=3)
    print(a)
