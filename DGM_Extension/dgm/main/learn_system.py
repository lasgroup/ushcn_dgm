"""
Making functions store stuff in self or assuming that some arguments are stored in self is very bad style.
Disentangle functions such that they return arguments and take arguments as inputs to make structure clear.
Best example is data creation. There, we should probably create something like a data_creation_kwargs, that is then
passed to super.init
"""

import copy
import os
import pickle
import time
from functools import partial
from typing import List, Optional, Dict, Tuple, Any, Callable

from jax.experimental.ode import odeint

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import jit, value_and_grad
from jax.experimental.optimizers import adam, sgd
from jax.tree_util import tree_leaves
from sklearn.preprocessing import StandardScaler


from dgm.utils.helper_functions import make_positive, get_leading_singular_value_and_eigenvector
from wandb_scripts.runDB import RunDB

from dgm.dynamics.dynamics_model import get_dynamics
from dgm.objectives.objective_builder import get_objective_builder
from dgm.plotter.plotter import Plotter
from dgm.schedules.betas import get_betas, BetaSchedule
from dgm.schedules.learning_rate import get_learning_rate
from dgm.schedules.weight_decay import get_weight_decay
from dgm.simulator.simulator import get_simulator
from dgm.smoother.smoother import get_smoother
from dgm.utils.helper_functions import unroll_dictionary, replace_str
from dgm.utils.representatives import Optimizer
from dgm.utils.ic_extractor import get_initial_condition_extractor

from abc import ABC, abstractmethod

from dgm.utils.USHCN_handling import load_raw_data, extract_time_and_value_arrays_from_GRU_data, \
    extract_trajectories_with_more_than_n_obs_per_dim, \
    extract_trajectories_with_more_than_n_train_obs_per_dim_and_3_val_obs, format_data_for_dgm, \
    extract_trajectories_with_ics_within_slack, extract_time_value_and_position_arrays_from_GRU_data

from dgm.utils.position_extractor import PositionExtractor

Schedule = Callable[[int], float]
Range = Optional[Tuple[float, float]]
pytree = Any


class AbstractLearnSystem(ABC):
    def __init__(
            self,
            seed: int,
            smoother: Dict,
            dynamics: Dict,
            betas: Dict,
            optimizer: Dict,
            priors: Dict,
            logging: Dict,
            numerical_correction: float,
            num_derivative_points_per_trajectory: int,
            split_pretraining: Optional[bool] = True,
            create_equidistant_derivative_times: Optional[bool] = True,
            ignore_third_dimension_input_for_deriv_times: Optional[bool] = True,
            obs_std_prior: Optional[Dict] = None,
            spectral_regularization_weight_smoother: Optional[float] = 0,
            spectral_regularization_weight_dynamics: Optional[float] = 0,
            use_spectral_normalization: Optional[bool] = False,
            n_l2_pretraining_for_dynamics: Optional[int] = 0
    ):
        self.n_l2_pretraining_for_dynamics = n_l2_pretraining_for_dynamics
        self.spectral_regularization_weight_smoother = spectral_regularization_weight_smoother
        self.spectral_regularization_weight_dynamics = spectral_regularization_weight_dynamics
        self.use_spectral_normalization = use_spectral_normalization
        self.start_vecs_smoother = None
        self.start_vecs_dynamics = None

        self.obs_std_prior = obs_std_prior
        self.create_equidistant_derivative_times = create_equidistant_derivative_times
        self.ignore_third_dimension_input_for_deriv_times = ignore_third_dimension_input_for_deriv_times
        self.split_pretraining = split_pretraining
        self.numerical_correction = numerical_correction
        self.track_just_loss = logging["track_just_loss"]
        self.core_type = smoother["core"]["type"]
        self.core_kwargs = smoother["core"]["kwargs"]
        # Adding kwargs here seems a bit hacky, it works but should you
        # think of a better solution don't hesitate to implement it
        self.core_kwargs['weight_key'] = "core"
        self.mean_head_type = smoother["mean_head"]["type"]
        self.mean_head_kwargs = smoother["mean_head"]["kwargs"]
        self.mean_head_kwargs['weight_key'] = "mean_head"
        self.kernel_core_type = smoother["kernel_core"]["type"]
        self.kernel_core_kwargs = smoother["kernel_core"]["kwargs"]
        self.kernel_core_kwargs['weight_key'] = "kernel_core"
        self.kernel_head_type = smoother['kernel_head']['type']
        self.kernel_head_kwargs = smoother["kernel_head"]["kwargs"]
        self.kernel_head_kwargs['weight_key'] = "kernel_head"

        self.learning_rate: Schedule = get_learning_rate(optimizer["learning_rate"]["type"],
                                                         optimizer["learning_rate"]["kwargs"])

        self.parameters = None
        self.kernel_type = smoother["kernel"]["type"]
        self.kernel_kwargs = smoother["kernel"]["kwargs"]
        self.optimizer_type = optimizer["type"]
        self.dynamics_model = dynamics["type"]
        self.dynamics_kwargs = dynamics['kwargs']
        self.dynamics_time_dependence = dynamics.get('time_dependent', False)
        self.dynamics_ic_dependence = dynamics.get('ic_dependent', False)

        self.current_rng = jax.random.PRNGKey(seed)
        self.betas: BetaSchedule = get_betas(betas["type"], betas["kwargs"])

        self.wd_core = get_weight_decay(priors['wd_core']['type'], priors['wd_core']['kwargs'])
        self.wd_kernel_core = get_weight_decay(priors['wd_kernel_core']['type'], priors['wd_core']['kwargs'])
        self.wd_kernel_head = get_weight_decay(priors['wd_kernel_head']['type'], priors['wd_core']['kwargs'])
        self.wd_mean_head = get_weight_decay(priors['wd_mean_head']['type'], priors['wd_core']['kwargs'])
        self.wd_obs_noise = get_weight_decay(priors['wd_pure_kernel']['observation_noise']['type'],
                                             priors['wd_pure_kernel']['observation_noise']['kwargs'])
        self.wd_kernel_variance = get_weight_decay(priors['wd_pure_kernel']['kernel_variance']['type'],
                                                   priors['wd_pure_kernel']['kernel_variance']['kwargs'])
        self.wd_kernel_lengthscales = get_weight_decay(priors['wd_pure_kernel']['kernel_lengthscale']['type'],
                                                       priors['wd_pure_kernel']['kernel_lengthscale']['kwargs'])
        self.wd_dynamics: Schedule = get_weight_decay(priors["wd_dynamics"]['type'], priors["wd_dynamics"]['kwargs'])

        self.track_wandb = logging["track_wandb"]
        self.num_derivative_points_per_trajectory = num_derivative_points_per_trajectory
        self._create_data()

        self.time_normalizers = [StandardScaler() for _ in range(self.state_dimension)]
        self.state_normalizers = [StandardScaler() for _ in range(self.state_dimension)]
        self._normalize_data()

        self._prepare_data_for_training()

        self._prepare_smoother()
        self._prepare_dynamics_model()
        self._prepare_objective_builder()
        self._prepare_optimizer()
        self._prepare_spec_reg()

    @abstractmethod
    def _get_ground_truth(self, initial_conditions, times, rng):
        raise NotImplementedError("This function is not implemented for this LearnSystem")

    @abstractmethod
    def _create_data(self):
        raise NotImplementedError("Please implement a data creation function for this type of system.")

    def _normalize_data(self):
        time_before_normalization = time.time()

        for dimension_id in range(self.state_dimension):
            all_times_of_this_dimension = jnp.concatenate(self.times[dimension_id])
            all_observations_of_this_dimension = jnp.concatenate(self.observations[dimension_id])
            self.time_normalizers[dimension_id].fit(all_times_of_this_dimension.reshape(-1, 1))
            self.state_normalizers[dimension_id].fit(all_observations_of_this_dimension.reshape(-1, 1))

        self.normalized_times = []
        self.normalized_observations = []
        self.normalized_initial_conditions = []
        self.normalized_test_times = []
        self.normalized_ground_truth_states = []
        self.normalized_ground_truth_derivatives = []

        for dimension_id in range(self.state_dimension):
            current_normalized_times = []
            current_normalized_test_times = []
            current_normalized_states = []
            current_normalized_ground_truth_states = []
            current_normalized_ground_truth_derivatives = []
            derivative_scale_for_this_dimension = \
                self.time_normalizers[dimension_id].scale_ / self.state_normalizers[dimension_id].scale_
            for trajectory_id in range(self.num_trajectories):
                current_normalized_times.append(
                    self.time_normalizers[dimension_id].transform(
                        self.times[dimension_id][trajectory_id].reshape(-1, 1)
                    ).reshape(-1) if len(self.times[dimension_id][trajectory_id]) > 0 else np.array([]).reshape(-1, 1)
                )
                current_normalized_test_times.append(
                    self.time_normalizers[dimension_id].transform(
                        self.test_times[dimension_id][trajectory_id].reshape(-1, 1)
                    ).reshape(-1) if len(self.test_times[dimension_id][trajectory_id]) > 0 else np.array([]).reshape(-1, 1)
                )
                current_normalized_states.append(
                    self.state_normalizers[dimension_id].transform(
                        self.observations[dimension_id][trajectory_id].reshape(-1, 1)
                    ) if len(self.observations[dimension_id][trajectory_id]) > 0 else np.array([]).reshape(-1, 1)
                )

                current_normalized_ground_truth_states.append(
                    self.state_normalizers[dimension_id].transform(
                        self.ground_truth_states[dimension_id][trajectory_id].reshape(-1, 1)
                    ) if len(self.ground_truth_states[dimension_id][trajectory_id]) > 0 else np.array([]).reshape(-1, 1)
                )
                current_normalized_ground_truth_derivatives.append(
                    derivative_scale_for_this_dimension * self.ground_truth_derivatives[dimension_id][trajectory_id]
                )
            self.normalized_times.append(current_normalized_times)
            self.normalized_test_times.append(current_normalized_test_times)
            self.normalized_observations.append(current_normalized_states)
            self.normalized_ground_truth_states.append(current_normalized_ground_truth_states)
            self.normalized_ground_truth_derivatives.append(current_normalized_ground_truth_derivatives)
        self._normalize_initial_conditions()
        print("Time for normalization", time.time() - time_before_normalization)

    def _normalize_initial_conditions(self):
        for dimension_id in range(self.state_dimension):
            current_normalized_initial_conditions = []
            for trajectory_id in range(self.num_trajectories):
                current_normalized_initial_conditions.append(
                    self.state_normalizers[dimension_id].transform(
                        self.initial_conditions[trajectory_id][dimension_id].reshape(-1, 1)
                    )
                )
            self.normalized_initial_conditions.append(current_normalized_initial_conditions)
        self.ic_normalizers = self.state_normalizers

    def _prepare_data_for_training(self):
        self.joint_normalized_test_times = []
        self.joint_normalized_times = []
        self.joint_normalized_observations = []
        max_times = []
        min_times = []

        for dimension_id in range(self.state_dimension):
            self.joint_normalized_test_times.append(jnp.concatenate(self.normalized_test_times[dimension_id]))
            self.joint_normalized_times.append(jnp.concatenate(self.normalized_times[dimension_id]))
            self.joint_normalized_observations.append(jnp.concatenate(self.normalized_observations[dimension_id]))
            max_times.append(jnp.max(self.joint_normalized_times[dimension_id]))
            min_times.append(jnp.min(self.joint_normalized_times[dimension_id]))

        if self.create_equidistant_derivative_times:
            min_time = min(min_times)
            max_time = max(max_times)

            self.joint_normalized_times_for_derivatives = []
            times_for_derivatives = []
            for dim_id in range(self.state_dimension):
                times_for_this_dim = []
                for trajectory_id in range(self.num_trajectories):
                    times_for_this_dim.append(
                        jnp.linspace(min_time, max_time, self.num_derivative_points_per_trajectory)
                    )
                times_for_derivatives.append(times_for_this_dim)

            self.joint_normalized_times_for_derivatives = []
            for dim_id in range(self.state_dimension):
                self.joint_normalized_times_for_derivatives.append(jnp.concatenate(times_for_derivatives[dim_id]))
        else:
            self.joint_normalized_times_for_derivatives = copy.deepcopy(self.joint_normalized_times)
            times_for_derivatives = copy.deepcopy(self.normalized_times)

        self.joint_repeated_normalized_initial_conditions = []
        self.joint_repeated_normalized_test_initial_conditions = []
        self.joint_repeated_normalized_initial_conditions_derivatives = []

        for dimension_id in range(self.state_dimension):
            ic_to_pass_for_this_dimension = []
            ic_for_test_for_this_dimension = []
            ic_for_deriv_for_this_dimension = []
            for traj_id in range(self.num_trajectories):
                full_train_ic = []
                full_test_ic = []
                full_deriv_ic = []
                for train_ic_dim_id in range(self.state_dimension):
                    full_train_ic.append(self.normalized_initial_conditions[train_ic_dim_id][traj_id])
                    full_test_ic.append(self.normalized_initial_conditions[train_ic_dim_id][traj_id])
                    full_deriv_ic.append(self.normalized_initial_conditions[train_ic_dim_id][traj_id])
                full_train_ic = np.asarray(full_train_ic)
                full_test_ic = np.asarray(full_test_ic)
                full_deriv_ic = np.asarray(full_deriv_ic)
                ic_to_pass_for_this_dimension.append(
                    jnp.repeat(full_train_ic.reshape(1, -1),
                               self.normalized_times[dimension_id][traj_id].size, axis=0)
                )
                ic_for_test_for_this_dimension.append(
                    jnp.repeat(full_test_ic.reshape(1, -1),
                               self.test_times[dimension_id][traj_id].size, axis=0)
                )

                if self.ignore_third_dimension_input_for_deriv_times and dimension_id == 2:
                    # adjust times for derivatives
                    times_for_derivatives[dimension_id][traj_id] = jnp.unique(jnp.squeeze(jnp.concatenate(
                        [times_for_derivatives[0][traj_id].reshape([-1, 1]),
                         times_for_derivatives[1][traj_id].reshape([-1, 1])],
                        axis=0)
                        ))

                ic_for_deriv_for_this_dimension.append(
                    jnp.repeat(full_deriv_ic.reshape(1, -1),
                               times_for_derivatives[dimension_id][traj_id].size, axis=0)
                )

            self.joint_repeated_normalized_initial_conditions.append(
                jnp.concatenate(ic_to_pass_for_this_dimension, axis=0)
            )

            self.joint_repeated_normalized_test_initial_conditions.append(
                jnp.concatenate(ic_for_test_for_this_dimension, axis=0)
            )

            self.joint_repeated_normalized_initial_conditions_derivatives.append(
                jnp.concatenate(ic_for_deriv_for_this_dimension, axis=0)
            )
        if self.ignore_third_dimension_input_for_deriv_times:
            # concatenate times_for_derivatives
            self.joint_normalized_times_for_derivatives[2] = jnp.concatenate(times_for_derivatives[2])

    def _prepare_smoother(self):
        time_smoother = time.time()
        (
            self.smoother_init,
            self.smoother_apply,
            self.smoother_get_means_and_covariances_test,
            self.get_smoother_regularization,
        ) = get_smoother(kernel=self.kernel_type, kernel_kwargs=self.kernel_kwargs,
                         core_type=self.core_type, core_kwargs=self.core_kwargs,
                         mean_head_type=self.mean_head_type, mean_head_kwargs=self.mean_head_kwargs,
                         kernel_core_type=self.kernel_core_type, kernel_core_kwargs=self.kernel_core_kwargs,
                         kernel_head_type=self.kernel_head_type, kernel_head_kwargs=self.kernel_head_kwargs,
                         n_dim=self.state_dimension, numerical_correction=self.numerical_correction)
        print("Time for smoother preparation: ", time.time() - time_smoother)

    def _prepare_dynamics_model(self):
        time_dynamics = time.time()
        (
            self.dynamics_model_init,
            self.dynamics_model_apply,
            self.dynamics_for_plotting,
            self.dynamics_sample_trajectories,
            self.get_dynamics_regularization
        ) = get_dynamics(dynamics_model=self.dynamics_model, state_normalizers=self.state_normalizers,
                         time_normalizers=self.time_normalizers, state_dimension=self.state_dimension,
                         dynamics_kwargs=self.dynamics_kwargs)
        print("Time for dynamics preparation: ", time.time() - time_dynamics)

    def _prepare_l2_pretraining(self, smoother_params):

        # precompute all smoother quantities
        def compute_smoother_stuff_for_one_dimension(dimension):
            # Compute smoother terms
            (
                posterior_means,
                posterior_smoother_derivative_means,
                _,
                _
            ) = self.smoother_apply(
                self.joint_normalized_times,
                [self.joint_normalized_times_for_derivatives[dimension]] * len(
                    self.joint_normalized_times_for_derivatives),
                self.joint_repeated_normalized_initial_conditions,
                [self.joint_repeated_normalized_initial_conditions_derivatives[dimension]]*len(
                    self.joint_repeated_normalized_initial_conditions_derivatives),
                self.joint_normalized_observations,
                smoother_params
            )
            return posterior_means, posterior_smoother_derivative_means

        smoother_states = []
        smoother_derivatives = []
        for dim_id in range(len(self.joint_normalized_observations)):
            posterior_means, posterior_smoother_derivative_means = compute_smoother_stuff_for_one_dimension(dim_id)
            smoother_states.append(jnp.stack(posterior_means).T)
            smoother_derivatives.append(posterior_smoother_derivative_means[dim_id])
        # done with precomputing smoother quantities

        @jit
        def l2_objective(dynamics_parameters: Dict,
                         regularization_weight: float):

            def calculate_dynamics_stuff_for_one_dimension(dimension):
                states_for_dynamics_model = smoother_states[dimension]

                # Compute dynamics terms
                posterior_dynamics_derivative_means, posterior_dynamics_derivatives_covariances = \
                    self.dynamics_model_apply(
                        dynamics_parameters, states_for_dynamics_model
                    )

                return posterior_dynamics_derivative_means[:, dimension]

            averaged_losses = []
            for dim_id in range(len(self.joint_normalized_observations)):
                current_smoother_derivatives = smoother_derivatives[dim_id]
                current_dynamics_derivatives = calculate_dynamics_stuff_for_one_dimension(dim_id)
                curr_loss = jnp.mean((current_smoother_derivatives - current_dynamics_derivatives)**2)
                averaged_losses.append(curr_loss)

            averaged_losses = jnp.array(averaged_losses)

            regularization_loss = self.get_dynamics_regularization(dynamics_parameters,
                                                                   {'dynamics': regularization_weight}
                                                                   )

            return 2*jnp.mean(averaged_losses) + regularization_loss

        self.values_and_grad_l2_objective = jit(value_and_grad(l2_objective, 0))  # dynamics

    def _prepare_objective_builder(self):
        time_objective_builder = time.time()
        self.current_rng, *keys = jax.random.split(self.current_rng, 3)
        dynamics_parameters = self.dynamics_model_init(keys[0])
        smoother_parameters = self.smoother_init(keys[1], self.state_dimension)
        self.parameters = {"smoother": smoother_parameters, "dynamics": dynamics_parameters, }
        self.num_dynamics_parameters = 0
        self.num_smoother_parameters = 0

        for leave in tree_leaves(dynamics_parameters):
            self.num_dynamics_parameters += leave.size
        for leave in tree_leaves(smoother_parameters):
            self.num_smoother_parameters += leave.size
        self.num_parameters = self.num_smoother_parameters + self.num_dynamics_parameters

        number_of_obs_per_dim = []
        for dim_id in range(len(self.observations)):
            number_of_obs_for_this_dim = 0
            for traj_id in range(len(self.observations[dim_id])):
                number_of_obs_for_this_dim += len(self.observations[dim_id][traj_id])
            number_of_obs_per_dim.append(number_of_obs_for_this_dim)
        number_of_obs_per_dim = jnp.array(number_of_obs_per_dim)

        self.objective_builder = get_objective_builder(apply_smoother=self.smoother_apply,
                                                       apply_dynamics=self.dynamics_model_apply,
                                                       get_dynamics_regularization=self.get_dynamics_regularization,
                                                       get_smoother_regularization=self.get_smoother_regularization,
                                                       is_dynamics_time_dependent=self.dynamics_time_dependence,
                                                       is_dynamics_ic_dependent=self.dynamics_ic_dependence,
                                                       return_split_objectives=self.split_pretraining,
                                                       obs_std_prior=self.obs_std_prior,
                                                       number_of_obs_per_dimension=number_of_obs_per_dim)
        print("Time to prepare objective builder", time.time() - time_objective_builder)
        time_objective_builder = time.time()

        if not self.split_pretraining:
            def joint_objective(
                    parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    observations,
                    betas,
                    weights
            ):
                loss_objective = self.objective_builder(
                    parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    observations,
                    betas,
                    weights
                )
                regularization_loss_smoother = self.get_spectral_regularization_smoother(parameters['smoother'])
                regularization_loss_dynamics = self.get_spectral_regularization_dynamics(parameters['dynamics'])
                return loss_objective + regularization_loss_smoother + regularization_loss_dynamics

            self.values_and_grad = jit(value_and_grad(joint_objective, 0))
        else:
            def smoother_objective(
                    smoother_parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,  # List over dimensions of shape n_deriv_obs x n_dim
                    observations,  # List over dimensions of shape n_obs x n_dim
                    weights
            ):
                loss_smoother = self.objective_builder[1](
                    smoother_parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,  # List over dimensions of shape n_deriv_obs x n_dim
                    observations,  # List over dimensions of shape n_obs x n_dim
                    weights
                )
                loss_regularization = self.get_spectral_regularization_smoother(smoother_parameters)
                return loss_regularization + loss_smoother

            def dynamics_objective(
                    dynamics_parameters,
                    smoother_parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                    observations,  # List over dimensions of shape n_obs x n_dim
                    weights
            ):
                loss_dynamics = self.objective_builder[2](
                    dynamics_parameters,
                    smoother_parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                    observations,  # List over dimensions of shape n_obs x n_dim
                    weights
                )
                loss_regularization = self.get_spectral_regularization_dynamics(dynamics_parameters)
                return loss_dynamics + loss_regularization

            def tune_smoother_with_dynamics_objective(
                smoother_parameters,
                dynamics_parameters,
                times,  # List over dimensions of shape n_obs
                times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                initial_conditions,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives,
                observations,
                betas,
                weights
            ):
                parameters = {
                    'smoother': smoother_parameters,
                    'dynamics': dynamics_parameters
                }
                loss_objective = self.objective_builder[0](
                    parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    observations,
                    betas,
                    weights
                )
                regularization_loss_smoother = self.get_spectral_regularization_smoother(parameters['smoother'])
                regularization_loss_dynamics = self.get_spectral_regularization_dynamics(parameters['dynamics'])
                return loss_objective + regularization_loss_smoother + regularization_loss_dynamics

            def joint_objective(
                    parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    observations,
                    betas,
                    weights
            ):
                loss_objective = self.objective_builder[0](
                    parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    observations,
                    betas,
                    weights
                )
                regularization_loss_smoother = self.get_spectral_regularization_smoother(parameters['smoother'])
                regularization_loss_dynamics = self.get_spectral_regularization_dynamics(parameters['dynamics'])
                return loss_objective + regularization_loss_smoother + regularization_loss_dynamics

            def dynamics_objective_for_std_params(
                dynamics_std_parameters: pytree,
                dynamics_mean_parameters: pytree,
                smoother_parameters: pytree,
                times: jnp.array,  # List over dimensions of shape n_obs
                times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
                initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives: jnp.array,
                # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                observations: jnp.array,  # List over dimensions of shape n_obs x n_dim
                weights
            ):
                dynamics_parameters = {
                    'mean': dynamics_mean_parameters,
                    'std': dynamics_std_parameters
                }
                loss_dynamics = self.objective_builder[2](
                    dynamics_parameters,
                    smoother_parameters,
                    times,  # List over dimensions of shape n_obs
                    times_for_derivatives,  # List over dimensions of shape n_deriv_obs
                    initial_conditions,  # List over dimensions of shape n_obs x n_dim
                    initial_conditions_for_derivatives,
                    # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                    observations,  # List over dimensions of shape n_obs x n_dim
                    weights
                )
                loss_regularization = self.get_spectral_regularization_dynamics(dynamics_parameters)
                return loss_dynamics + loss_regularization

            def l2_std_dynamics_objective(
                dynamics_std_parameters: pytree,
                dynamics_mean_parameters: pytree,
                smoother_parameters: pytree,
                times: jnp.array,  # List over dimensions of shape n_obs
                times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
                initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives: jnp.array,
                # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                observations: jnp.array,  # List over dimensions of shape n_obs x n_dim
                weights
            ):
                loss_objective = self.objective_builder[3](
                    dynamics_std_parameters,
                    dynamics_mean_parameters,
                    smoother_parameters,
                    times,
                    times_for_derivatives,
                    initial_conditions,
                    initial_conditions_for_derivatives,
                    observations,
                    weights
                )
                parameters_dynamics = {
                    'mean': dynamics_mean_parameters,
                    'std': dynamics_std_parameters
                }

                regularization_loss_smoother = self.get_spectral_regularization_smoother(smoother_parameters)
                regularization_loss_dynamics = self.get_spectral_regularization_dynamics(parameters_dynamics)
                return loss_objective + regularization_loss_smoother + regularization_loss_dynamics
                pass

            self.values_and_grad = jit(value_and_grad(joint_objective, 0))  # complete
            self.values_and_grad_smoother = jit(value_and_grad(smoother_objective, 0))  # smoother
            self.values_and_grad_dynamics = jit(value_and_grad(dynamics_objective, 0))  # dynamics
            self.values_and_grad_dynamics_for_std_params = jit(value_and_grad(dynamics_objective_for_std_params, 0))
            self.values_and_grad_l2_std_dynamics = jit(value_and_grad(l2_std_dynamics_objective, 0))
            self.values_and_grad_tune_smoother_with_dynamics = jit(
                value_and_grad(tune_smoother_with_dynamics_objective, 0))
        print("Time to jit: ", time.time() - time_objective_builder)

    def _prepare_optimizer(self):
        if self.optimizer_type == Optimizer.ADAM:
            self.optimizer = adam
        elif self.optimizer_type == Optimizer.SGD:
            self.optimizer = sgd

    @staticmethod
    def _get_smoother_nn_matrices_from_parameters(smoother_parameters):
        return [
            # mean head
            # smoother_parameters['mean'][0][0],
            # core
            smoother_parameters['core'][0][0],
            smoother_parameters['core'][2][0],
            smoother_parameters['core'][4][0]
            ]
        #    + [
        #     # kernel head
        #     params for params in smoother_parameters['smoother']['kernel']['kernel'][0]
        # ]

    @staticmethod
    def _get_dynamics_nn_matrices_from_parameters(dynamics_parameters):
        return [
            dynamics_parameters['mean_and_std'][0][0],  # input layer
            dynamics_parameters['mean_and_std'][2][0],
            dynamics_parameters['mean_and_std'][4][0],
            # dynamics_parameters['mean_and_std'][6][0]  # output layer
        ]

    def _prepare_spec_reg(self):
        if self.spectral_regularization_weight_smoother > 0:
            def get_spectral_regularization_smoother(parameters_smoother):
                smoother_matrices = self._get_smoother_nn_matrices_from_parameters(parameters_smoother)
                singular_values_squared = []
                eigenvectors = []
                for i in range(len(smoother_matrices)):
                    singular_value, eigenvector = get_leading_singular_value_and_eigenvector(smoother_matrices[i],
                                                                                             self.start_vecs_smoother[i])
                    singular_values_squared.append(singular_value**2)
                    eigenvectors.append(eigenvector)
                return self.spectral_regularization_weight_smoother*jnp.sum(jnp.array(singular_values_squared))
        else:
            def get_spectral_regularization_smoother(parameters):
                return 0
        self.get_spectral_regularization_smoother = get_spectral_regularization_smoother

        if self.spectral_regularization_weight_dynamics > 0:
            def get_spectral_regularization_dynamics(parameters_dynamics):
                dynamics_matrices = self._get_dynamics_nn_matrices_from_parameters(parameters_dynamics)
                singular_values_squared = []
                eigenvectors = []
                for i in range(len(dynamics_matrices)):
                    singular_value, eigenvector = get_leading_singular_value_and_eigenvector(dynamics_matrices[i], self.start_vecs_dynamics[i])
                    singular_values_squared.append(singular_value**2)
                    eigenvectors.append(eigenvector)
                return self.spectral_regularization_weight_dynamics*jnp.sum(jnp.array(singular_values_squared))
        else:
            def get_spectral_regularization_dynamics(parameters):
                return 0
        self.get_spectral_regularization_dynamics = get_spectral_regularization_dynamics

    def train(self, number_of_steps, num_steps_smoother_pretraining=0, num_steps_dynamics_pretraining=0,
              num_just_std_params_training=0, num_std_l2_loss_training=0, num_smoother_tuning_dynamics=0):
        current_time = time.time()
        initial_time = current_time
        opt_init, opt_update, get_params = self.optimizer(self.learning_rate)

        if self.spectral_regularization_weight_smoother > 0:
            smoother_matrices = self._get_smoother_nn_matrices_from_parameters(self.parameters['smoother'])
            self.start_vecs_smoother = []
            for i in range(len(smoother_matrices)):
                self.start_vecs_smoother.append(jnp.ones(smoother_matrices[i].shape[1]))

        if self.spectral_regularization_weight_dynamics > 0:
            dynamics_matrices = self._get_dynamics_nn_matrices_from_parameters(self.parameters['dynamics'])
            self.start_vecs_dynamics = []
            for i in range(len(dynamics_matrices)):
                self.start_vecs_dynamics.append(jnp.ones(dynamics_matrices[i].shape[1]))

        """ start of ugly code """
        if self.use_spectral_normalization:

            if self.spectral_regularization_weight_smoother > 0 or self.spectral_regularization_weight_dynamics > 0:
                raise AttributeError("Do not use spectral normalization and regularization at the same time.")

            dynamics_matrices = self._get_dynamics_nn_matrices_from_parameters(self.parameters['dynamics'])
            self.start_vecs_dynamics = []
            for i in range(len(dynamics_matrices)):
                self.start_vecs_dynamics.append(jnp.ones(dynamics_matrices[i].shape[1]))
            smoother_matrices = self._get_smoother_nn_matrices_from_parameters(self.parameters['smoother'])
            self.start_vecs_smoother = []
            for i in range(len(smoother_matrices)):
                self.start_vecs_smoother.append(jnp.ones(smoother_matrices[i].shape[1]))

            def do_spectral_normalization_smoother(params_smoother, start_vecs_smoother):
                smoother_matrices = [
                    # mean head
                    # params_smoother['mean'][0][0],
                    # core
                    params_smoother['core'][0][0],
                    params_smoother['core'][2][0],
                    params_smoother['core'][4][0]]
                    # ] + [
                    #     # kernel head
                    #     params for params in params_smoother['smoother']['kernel']['kernel'][0]
                    # ]
                sing_vals_smoother = []
                new_start_vecs_smoother = []
                for i in range(len(smoother_matrices)):
                    singular_value, eigenvector = get_leading_singular_value_and_eigenvector(smoother_matrices[i],
                                                                                             start_vecs_smoother[i])
                    sing_vals_smoother.append(singular_value)
                    new_start_vecs_smoother.append(eigenvector)

                new_tuple = (params_smoother['core'][0][0] / sing_vals_smoother[0], params_smoother['core'][0][1])
                params_smoother['core'][0] = new_tuple
                new_tuple = (params_smoother['core'][2][0] / sing_vals_smoother[1], params_smoother['core'][2][1])
                params_smoother['core'][2] = new_tuple
                new_tuple = (params_smoother['core'][4][0] / sing_vals_smoother[2], params_smoother['core'][4][1])
                params_smoother['core'][4] = new_tuple

                return params_smoother, new_start_vecs_smoother

            def do_spectral_normalization_dynamics(params_dynamics, start_vecs_dynamics):
                dynamics_matrices = [
                    params_dynamics['mean_and_std'][0][0],
                    params_dynamics['mean_and_std'][2][0],
                    params_dynamics['mean_and_std'][4][0],
                    ]

                sing_vals_dynamics = []
                new_start_vecs_dynamics = []
                for i in range(len(dynamics_matrices)):
                    singular_value, eigenvector = get_leading_singular_value_and_eigenvector(dynamics_matrices[i],
                                                                                             start_vecs_dynamics[i])
                    sing_vals_dynamics.append(singular_value)
                    new_start_vecs_dynamics.append(eigenvector)

                new_tuple = (params_dynamics['mean_and_std'][0][0] / sing_vals_dynamics[0],
                             params_dynamics['mean_and_std'][0][1])
                params_dynamics['mean_and_std'][0] = new_tuple

                new_tuple = (params_dynamics['mean_and_std'][2][0] / sing_vals_dynamics[1],
                             params_dynamics['mean_and_std'][2][1])
                params_dynamics['mean_and_std'][2] = new_tuple

                new_tuple = (params_dynamics['mean_and_std'][4][0] / sing_vals_dynamics[2],
                             params_dynamics['mean_and_std'][4][1])
                params_dynamics['mean_and_std'][4] = new_tuple

                return params_dynamics, new_start_vecs_dynamics
        else:
            def do_spectral_normalization_smoother(parameters, vecs_smoother):
                return parameters, vecs_smoother

            def do_spectral_normalization_dynamics(parameters, vecs_dynamics):
                return parameters, vecs_dynamics
        """ end of ugly code """

        @jit
        def do_l2_on_std_step(step, params_dynamics_std, original_params_dynamics_mean, original_params_smoother):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad_l2_std_dynamics(
                get_params(params_dynamics_std),
                original_params_dynamics_mean,
                original_params_smoother,
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                weights
            )
            return loss, opt_update(step, params_grad, params_dynamics_std)

        @jit
        def do_dynamics_std_pretraining_step(step,
                                             params_dynamics_std,
                                             original_params_dynamics_mean,
                                             original_params_smoother):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad_dynamics_for_std_params(
                get_params(params_dynamics_std),
                original_params_dynamics_mean,
                original_params_smoother,
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                weights
            )
            return loss, opt_update(step, params_grad, params_dynamics_std)

        @jit
        def do_pure_smoother_step(step, params_smoother):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad_smoother(
                get_params(params_smoother),
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                weights
            )
            return loss, opt_update(step, params_grad, params_smoother)

        def do_smoother_tuning_with_dynamics_step(step, params_smoother, original_dynamics_params):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad_tune_smoother_with_dynamics(
                get_params(params_smoother),
                original_dynamics_params,
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                self.betas(step),
                weights
            )
            return loss, opt_update(step, params_grad, params_smoother)

        @jit
        def do_pure_dynamics_step(step, params_dynamics, original_smoother_params):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad_dynamics(
                get_params(params_dynamics),
                original_smoother_params,
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                weights
            )
            return loss, opt_update(step, params_grad, params_dynamics)

        def do_l2_dynamics_step(step, params_dynamics):
            regularization_weight = self.wd_dynamics(step)

            loss, params_grad = self.values_and_grad_l2_objective(
                get_params(params_dynamics),
                regularization_weight
            )
            return loss, opt_update(step, params_grad, params_dynamics)

        @jit
        def do_step(step, curr_params):
            weights = {
                "kernel_variance": self.wd_kernel_variance(step),
                "kernel_lengthscale": self.wd_kernel_lengthscales(step),
                "obs_noise": self.wd_obs_noise(step),
                "dynamics": self.wd_dynamics(step),
                "core": self.wd_core(step),
                'kernel_core': self.wd_kernel_core(step),
                "kernel_head": self.wd_kernel_head(step),
                "mean_head": self.wd_mean_head(step)
            }
            loss, params_grad = self.values_and_grad(
                get_params(curr_params),
                self.joint_normalized_times,
                self.joint_normalized_times_for_derivatives,
                self.joint_repeated_normalized_initial_conditions,
                self.joint_repeated_normalized_initial_conditions_derivatives,
                self.joint_normalized_observations,
                self.betas(step),
                weights
            )
            return loss, opt_update(step, params_grad, curr_params)

        def log_wandb_in_training_loop(curr_params, loss):
            if self.track_wandb:
                if self.track_just_loss:
                    variables_dict = dict()
                    variables_dict["Loss"] = float(loss)
                else:
                    variables_dict = unroll_dictionary(get_params(curr_params))
                    variables_dict["Loss"] = float(loss)
                wandb.log(variables_dict)

        if self.split_pretraining:
            " Smoother Pretraining "
            smoother_params = opt_init(self.parameters['smoother'])
            for step in range(num_steps_smoother_pretraining):
                loss, smoother_params = do_pure_smoother_step(step, smoother_params)
                log_wandb_in_training_loop(smoother_params, loss)
                self.parameters['smoother'] = get_params(smoother_params)
                self.parameters['smoother'], self.start_vecs_smoother = do_spectral_normalization_smoother(
                    self.parameters['smoother'], self.start_vecs_smoother
                )
                smoother_params = opt_init(self.parameters['smoother'])

            " L2 Pretraining "
            if self.n_l2_pretraining_for_dynamics > 0:
                self._prepare_l2_pretraining(self.parameters['smoother'])

            dynamics_params = opt_init(self.parameters['dynamics'])
            for step in range(self.n_l2_pretraining_for_dynamics):
                loss, dynamics_params = do_l2_dynamics_step(step,
                                                            params_dynamics=dynamics_params
                                                            )
                log_wandb_in_training_loop(dynamics_params, loss)
                self.parameters['dynamics'] = get_params(dynamics_params)
                self.parameters['dynamics'], self.start_vecs_dynamics = do_spectral_normalization_dynamics(
                    self.parameters['dynamics'], self.start_vecs_dynamics
                )
                dynamics_params = opt_init(self.parameters['dynamics'])

            "Dynamics std on wasserstein pretraining"
            if num_just_std_params_training > 0:
                std_dynamics_params = opt_init(self.parameters['dynamics']['mean'])
            for step in range(num_just_std_params_training):
                loss, new_std_dynamics_params = do_dynamics_std_pretraining_step(
                    step,
                    params_dynamics_std=std_dynamics_params,
                    original_params_dynamics_mean=self.parameters['dynamics']['std'],
                    original_params_smoother=self.parameters['smoother']
                )
                log_wandb_in_training_loop(self.parameters, loss)
                self.parameters['dynamics']['std'] = get_params(new_std_dynamics_params)
                self.parameters['dynamics'], self.start_vecs_dynamics = do_spectral_normalization_dynamics(
                    self.parameters['dynamics'], self.start_vecs_dynamics
                )
                std_dynamics_params = opt_init(self.parameters['dynamics']['mean'])

            " Dynamics std on l2 std loss pretraining "
            if num_std_l2_loss_training > 0:
                std_dynamics_params = opt_init(self.parameters['dynamics']['mean'])
            for step in range(num_std_l2_loss_training):
                loss, new_std_dynamics_params = do_l2_on_std_step(
                    step,
                    params_dynamics_std=std_dynamics_params,
                    original_params_dynamics_mean=self.parameters['dynamics']['std'],
                    original_params_smoother=self.parameters['smoother']
                )
                log_wandb_in_training_loop(self.parameters, loss)
                self.parameters['dynamics']['std'] = get_params(new_std_dynamics_params)
                self.parameters['dynamics'], self.start_vecs_dynamics = do_spectral_normalization_dynamics(
                    self.parameters['dynamics'], self.start_vecs_dynamics
                )
                std_dynamics_params = opt_init(self.parameters['dynamics']['mean'])

            " Other dynamics pretraining "
            dynamics_params = opt_init(self.parameters['dynamics'])
            for step in range(num_steps_dynamics_pretraining):
                loss, dynamics_params = do_pure_dynamics_step(step,
                                                              params_dynamics=dynamics_params,
                                                              original_smoother_params=self.parameters['smoother']
                                                              )
                log_wandb_in_training_loop(dynamics_params, loss)
                self.parameters['dynamics'] = get_params(dynamics_params)
                self.parameters['dynamics'], self.start_vecs_dynamics = do_spectral_normalization_dynamics(
                    self.parameters['dynamics'], self.start_vecs_dynamics
                )
                dynamics_params = opt_init(self.parameters['dynamics'])


        " Proper training "

        params = opt_init(self.parameters)
        for step in range(number_of_steps):
            if step < 10:
                next_time = time.time()
                print("Time for step {}:".format(step), next_time - current_time)
                current_time = next_time
            loss, params = do_step(step, params)
            log_wandb_in_training_loop(params, loss)
            self.parameters = get_params(params)
            self.parameters['smoother'], self.start_vecs_smoother = do_spectral_normalization_smoother(
                self.parameters['smoother'], self.start_vecs_smoother
            )
            self.parameters['dynamics'], self.start_vecs_dynamics = do_spectral_normalization_dynamics(
                self.parameters['dynamics'], self.start_vecs_dynamics
            )
            params = opt_init(self.parameters)

        " smoother learning with dynamics "
        smoother_params = opt_init(self.parameters['smoother'])
        original_dynamics_params = self.parameters['dynamics']
        for step in range(num_smoother_tuning_dynamics):
            loss, new_smoother_params = do_smoother_tuning_with_dynamics_step(
                step, smoother_params, original_dynamics_params)
            log_wandb_in_training_loop(new_smoother_params, loss)
            self.parameters['smoother'] = get_params(new_smoother_params)
            self.parameters['smoother'], self.start_vecs_smoother = do_spectral_normalization_smoother(
                self.parameters['smoother'], self.start_vecs_smoother
            )
            smoother_params = opt_init(self.parameters['smoother'])

        time_spent_for_training = time.time() - initial_time
        print("Time spent for training:", time_spent_for_training, "seconds")
        # Save parameters_for_dgm
        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join('models', 'final_parameters.pkl')
            with open(os.path.join(wandb.run.dir, model_path), 'wb') as handle:
                pickle.dump(get_params(params), handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

    def _compute_nll_per_dimension(self, denormalized_state_means, denormalized_state_variances,
                                   denormalized_derivative_means, denormalized_derivative_variances,
                                   denormalized_dynamics_means, denormalized_dynamics_variances,
                                   true_states, true_derivatives):
        """
        all arguments are lists over dimensions over trajectories

        Returns a list of nlls over trajectories
        """
        nll_state = []
        nll_derivatives_smoother = []
        nll_derivatives_dynamics = []

        # Compute average (over dimension) average NLL score
        # Not over the range of self.num_trajectories but over the range of number of evaluated trajectories
        for trajectory_id in range((len(denormalized_state_means[0]))):
            current_true_states = []
            current_true_derivatives = []
            current_state_means = []
            current_state_variances = []
            current_derivative_means = []
            current_derivative_variances = []
            current_dynamics_means = []
            current_dynamics_variances = []
            for dimension_id in range(len(denormalized_state_means)):
                current_true_states.append(true_states[dimension_id][trajectory_id])
                current_true_derivatives.append(true_derivatives[dimension_id][trajectory_id])
                current_state_means.append(denormalized_state_means[dimension_id][trajectory_id])
                current_state_variances.append(denormalized_state_variances[dimension_id][trajectory_id])
                current_derivative_means.append(denormalized_derivative_means[dimension_id][trajectory_id])
                current_derivative_variances.append(denormalized_derivative_variances[dimension_id][trajectory_id])
                current_dynamics_means.append(denormalized_dynamics_means[dimension_id][trajectory_id])
                current_dynamics_variances.append(denormalized_dynamics_variances[dimension_id][trajectory_id])
            current_true_states = jnp.concatenate(current_true_states)
            current_true_derivatives = jnp.concatenate(current_true_derivatives)
            current_state_means = jnp.concatenate(current_state_means)
            current_state_variances = jnp.concatenate(current_state_variances)
            current_derivative_means = jnp.concatenate(current_derivative_means)
            current_derivative_variances = jnp.concatenate(current_derivative_variances)
            current_dynamics_means = jnp.concatenate(current_dynamics_means)
            current_dynamics_variances = jnp.concatenate(current_dynamics_variances)

            nll_state.append(
                self._mean_nll_of_independent_gaussians(
                    current_true_states, current_state_means, current_state_variances)
            )
            nll_derivatives_smoother.append(
                self._mean_nll_of_independent_gaussians(
                    current_true_derivatives, current_derivative_means, current_derivative_variances)
            )
            nll_derivatives_dynamics.append(
                self._mean_nll_of_independent_gaussians(
                    current_true_derivatives, current_dynamics_means, current_dynamics_variances)
            )
        return nll_state, nll_derivatives_smoother, nll_derivatives_dynamics

    @staticmethod
    def _mean_nll_of_independent_gaussians(test, mean, variance):
        """
        Calculates the negative log likelihood of a test vector, where it is assumed that each entry of the test vector
        is independently Gaussian distributed with mean and variance as specified in the corresponding entry of the
        mean and variance vectors.
        """
        mean_diff = (test - mean)
        nll_state_current = 0.5 * jnp.mean(mean_diff * mean_diff / variance)
        nll_state_current += 0.5 * jnp.mean(jnp.log(variance))
        nll_state_current += 0.5 * jnp.log(2 * jnp.pi)
        return nll_state_current

    @staticmethod
    def _prepare_nll_for_wandb(nll_state, nll_derivatives_smoother, nll_derivatives_dynamics, quantile):
        nll_state = jnp.array(nll_state)
        nll_derivatives_smoother = jnp.array(nll_derivatives_smoother)
        nll_derivatives_dynamics = jnp.array(nll_derivatives_dynamics)

        nll_state_median = jnp.median(nll_state)
        nll_derivatives_smoother_median = jnp.median(nll_derivatives_smoother)
        nll_derivatives_dynamics_median = jnp.median(nll_derivatives_dynamics)

        nll_state_lower_q = jnp.quantile(nll_state, q=1 - quantile)
        nll_derivatives_smoother_lower_q = jnp.quantile(nll_derivatives_smoother, q=1 - quantile)
        nll_derivatives_dynamics_lower_q = jnp.quantile(nll_derivatives_dynamics, q=1 - quantile)

        nll_state_upper_q = jnp.quantile(nll_state, q=quantile)
        nll_derivatives_smoother_upper_q = jnp.quantile(nll_derivatives_smoother, q=quantile)
        nll_derivatives_dynamics_upper_q = jnp.quantile(nll_derivatives_dynamics, q=quantile)

        variables_dict = dict()

        variables_dict['nll_state_mean'] = float(jnp.mean(nll_state))
        variables_dict['nll_derivatives_smoother_mean'] = float(jnp.mean(nll_derivatives_smoother))
        variables_dict['nll_derivatives_dynamics_mean'] = float(jnp.mean(nll_derivatives_dynamics))

        variables_dict['nll_state_median'] = float(nll_state_median)
        variables_dict['nll_derivatives_smoother_median'] = float(nll_derivatives_smoother_median)
        variables_dict['nll_derivatives_dynamics_median'] = float(nll_derivatives_dynamics_median)

        variables_dict['nll_state_lower_q'] = float(nll_state_lower_q)
        variables_dict['nll_derivatives_smoother_lower_q'] = float(nll_derivatives_smoother_lower_q)
        variables_dict['nll_derivatives_dynamics_lower_q'] = float(nll_derivatives_dynamics_lower_q)

        variables_dict['nll_state_upper_q'] = float(nll_state_upper_q)
        variables_dict['nll_derivatives_smoother_upper_q'] = float(nll_derivatives_smoother_upper_q)
        variables_dict['nll_derivatives_dynamics_upper_q'] = float(nll_derivatives_dynamics_upper_q)
        return variables_dict

    def _denormalize(self, state_means, state_variances, derivative_means, derivative_variances, dynamics_means,
                     dynamics_variances):  # all quantities are lists over dimensions

        denormalized_state_means = []
        denormalized_state_variances = []
        denormalized_derivative_means = []
        denormalized_derivative_variances = []
        denormalized_dynamics_means = []
        denormalized_dynamics_variances = []
        for dimension_id in range(len(state_means)):

            denormalized_state_means.append(
                self.state_normalizers[dimension_id].inverse_transform(
                    state_means[dimension_id].reshape([1, -1])
                ).reshape(-1)
            )
            denormalized_state_variances.append(
                self.state_normalizers[dimension_id].scale_ ** 2 * state_variances[dimension_id]
            )

            derivative_scale = self.state_normalizers[dimension_id].scale_ / self.time_normalizers[dimension_id].scale_

            denormalized_derivative_means.append(
                derivative_scale * derivative_means[dimension_id]
            )
            denormalized_derivative_variances.append(
                derivative_scale ** 2 * derivative_variances[dimension_id]
            )
            denormalized_dynamics_means.append(
                derivative_scale * dynamics_means[dimension_id]
            )
            denormalized_dynamics_variances.append(
                derivative_scale ** 2 * dynamics_variances[dimension_id]
            )

        return denormalized_state_means, denormalized_state_variances, denormalized_derivative_means, \
               denormalized_derivative_variances, denormalized_dynamics_means, denormalized_dynamics_variances

    @staticmethod
    def join_trajectories(initial_conditions: List[jnp.array],  # List over trajectories over jnp.array of shape n_dim
                          times: List[jnp.array]  # List over dimensions over trajectories over jnp.array of shape n_obs
                          ) -> Tuple[pytree, jnp.array, jnp.array]:
        all_trajectory_lengths = []
        all_joint_times = []
        all_joint_initial_conditions = []
        for dim_id in range(len(times)):
            trajectory_lengths_of_curr_dim = []
            joint_times_of_curr_dim = jnp.concatenate(times[dim_id])
            joint_initial_conditions_of_curr_dim = []

            for traj_id in range(len(times[dim_id])):
                curr_full_initial_condition = []
                trajectory_lengths_of_curr_dim.append(len(times[dim_id][traj_id]))
                joint_initial_conditions_of_curr_dim.append(
                    jnp.repeat(initial_conditions[traj_id].reshape(1, -1),
                               times[dim_id][traj_id].size,
                               axis=0)
                )
            all_trajectory_lengths.append(trajectory_lengths_of_curr_dim)
            all_joint_initial_conditions.append(jnp.concatenate(joint_initial_conditions_of_curr_dim, axis=0))
            all_joint_times.append(joint_times_of_curr_dim)

        return all_trajectory_lengths, all_joint_times, all_joint_initial_conditions

    @staticmethod
    def split_trajectories(trajectory_lengths, *data  # arguments to this function are lists over dimensions over the desired quantity
                           ) -> List[List[jnp.array]]:
        all_separated_data_objects = []
        for data_object in data:
            separated_data_objects = []
            for dimension_id in range(len(trajectory_lengths)):
                separated_data_object_of_this_dimension = []
                start_index = 0
                for length in trajectory_lengths[dimension_id]:
                    separated_data_object_of_this_dimension.append(
                        data_object[dimension_id][start_index: start_index + length])
                    start_index += length
                separated_data_objects.append(separated_data_object_of_this_dimension)
            all_separated_data_objects.append(separated_data_objects)
        return all_separated_data_objects

    def _get_split_predictions(self,
                               joint_normalized_times,  # List over dimensions of shape n_obs
                               joint_repeated_normalized_initial_conditions,  # List over trajectories of jnp.arrays  (n_test_obs x n_dim)
                               trajectory_lengths):  # List over dimensions over trajectories of integers
        state_means, state_variances, \
        derivative_means, derivative_variances = self.smoother_get_means_and_covariances_test(
            joint_normalized_times,
            self.joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            self.joint_repeated_normalized_initial_conditions,
            self.joint_normalized_observations,
            self.parameters["smoother"],
        )
        dynamics_means = []
        dynamics_variances = []
        for dimension_id in range(len(state_means)):
            evaluation_times = self.state_dimension * [joint_normalized_times[dimension_id]]
            evaluation_ics = self.state_dimension * [joint_repeated_normalized_initial_conditions[dimension_id]]
            dynamics_state_means = self.smoother_get_means_and_covariances_test(
                evaluation_times,  # evaluation
                self.joint_normalized_times,
                evaluation_ics,  # evaluation
                self.joint_repeated_normalized_initial_conditions,
                self.joint_normalized_observations,
                self.parameters["smoother"],
                )[0]
            dynamics_state_means = jnp.asarray(dynamics_state_means).T
            states_for_dynamics = dynamics_state_means
            if self.dynamics_time_dependence:
                states_for_dynamics = jnp.concatenate([dynamics_state_means, evaluation_times[0].reshape([-1, 1])], axis=1)
            elif self.dynamics_ic_dependence:
                states_for_dynamics = jnp.concatenate(
                    [dynamics_state_means, joint_repeated_normalized_initial_conditions[dimension_id]],
                    axis=1
                )
            one_dimensional_dynamics = self.dynamics_model_apply(
                self.parameters["dynamics"], states_for_dynamics)
            dynamics_means.append(one_dimensional_dynamics[0][:, dimension_id])
            dynamics_variances.append(one_dimensional_dynamics[1][:, dimension_id])

        # Denormalize everything
        denormalized_state_means, denormalized_state_variances, denormalized_derivative_means, \
            denormalized_derivative_variances, denormalized_dynamics_means, denormalized_dynamics_variances = \
            self._denormalize(
                state_means, state_variances,
                derivative_means, derivative_variances,
                dynamics_means, dynamics_variances)

        # Here all data are one big jnp.array now we split it since we would like to perform per trajectory analysis
        return self.split_trajectories(trajectory_lengths, denormalized_state_means, denormalized_state_variances,
                                       denormalized_derivative_means, denormalized_derivative_variances,
                                       denormalized_dynamics_means, denormalized_dynamics_variances)

    def normalize_and_reshape_times_and_ics(self, joint_times, joint_repeated_initial_conditions):

        joint_normalized_times = []
        joint_repeated_normalized_initial_conditions = []
        for dimension_id in range(len(joint_times)):
            ic_of_this_dimension = []
            for inner_dimension_id in range(len(joint_times)):
                ic_of_this_dimension.append(
                    self.ic_normalizers[inner_dimension_id].transform(
                        joint_repeated_initial_conditions[dimension_id][:, inner_dimension_id].reshape(-1, 1)
                    ) if joint_repeated_initial_conditions[dimension_id][:, inner_dimension_id].size > 0 else np.array([]).reshape(-1, 1)
                )
            ic_of_this_dimension = np.asarray(ic_of_this_dimension)
            ic_of_this_dimension = np.squeeze(ic_of_this_dimension)
            ic_of_this_dimension = ic_of_this_dimension.T
            joint_repeated_normalized_initial_conditions.append(
                ic_of_this_dimension
            )
            joint_normalized_times.append(
                self.time_normalizers[dimension_id].transform(joint_times[dimension_id].reshape(-1, 1)).reshape(-1)
                if joint_times[dimension_id].size > 0 else np.array([]).reshape(-1)
            )
        return joint_normalized_times, joint_repeated_normalized_initial_conditions

    def evaluate_nll_of_smoother_and_dynamics(
            self, use_training_ics_and_times: bool = True, new_initial_conditions: Optional = None,
            new_times: Optional = None, quantile=0.8):

        if use_training_ics_and_times:
            if not (new_initial_conditions is None) or not (new_times is None):
                print("Warning: initial_conditions and times are overwritten by training defaults")
            new_initial_conditions = self.initial_conditions
            new_times = self.test_times

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            new_initial_conditions, new_times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        # Prepare (not normalized) ground truth prediction
        self.current_rng, subkey = jax.random.split(self.current_rng)
        true_states, true_derivatives = self._get_ground_truth(new_initial_conditions, new_times, subkey)

        # Compute average (per dimension) average NLL score
        nll_state, nll_derivatives_smoother, nll_derivatives_dynamics = self._compute_nll_per_dimension(
            denormalized_state_means, denormalized_state_variances,
            denormalized_derivative_means, denormalized_derivative_variances,
            denormalized_dynamics_means, denormalized_dynamics_variances,
            true_states, true_derivatives)

        variables_dict = self._prepare_nll_for_wandb(nll_state, nll_derivatives_smoother, nll_derivatives_dynamics,
                                                     quantile)
        if self.track_wandb:
            wandb.log(variables_dict)
        return variables_dict

    def plot_training_trajectories_at_test_times(self, add_all_trajectories: bool = False):
        print('Before computing the values for plotting')
        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            self.initial_conditions, self.test_times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                self.test_times,
                denormalized_state_means,
                denormalized_state_variances,
                denormalized_derivative_means,
                denormalized_derivative_variances,
                denormalized_dynamics_means,
                denormalized_dynamics_variances,
                train_times=self.times,
                observations=self.observations,
                all_initial_conditions=self.initial_conditions if add_all_trajectories else None
            )
        figure_smoother_states.tight_layout()
        figure_smoother_derivatives.tight_layout()
        figure_dynamics_derivatives.tight_layout()

        state_filename = 'smoother_states_with_all_trajectories' if add_all_trajectories else 'smoother_states'
        if self.track_wandb:
            wandb.log({state_filename: wandb.Image(figure_smoother_states),
                       'smoother_derivatives': wandb.Image(figure_smoother_derivatives),
                       'dynamics_derivatives': wandb.Image(figure_dynamics_derivatives)})

    def plot_first_few_training_trajectories_at_test_times(self,
                                                           n_trajectories_to_plot: Optional[int] = 1,
                                                           ):
        print('Before computing the values for plotting')
        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            self.initial_conditions, self.test_times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
            denormalized_derivative_means, denormalized_derivative_variances, \
            denormalized_dynamics_means, denormalized_dynamics_variances = self._get_split_predictions(
                joint_normalized_times,
                joint_repeated_normalized_initial_conditions,
                trajectory_lengths)

        n_trajectories = len(denormalized_state_means[0])
        if n_trajectories < n_trajectories_to_plot:
            n_trajectories_to_plot = n_trajectories
            print("Only {} trajectories available. All trajectories will be printed")

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                self.get_first_n_trajectories(n_trajectories_to_plot, self.test_times),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_state_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_state_variances),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_derivative_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_derivative_variances),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_dynamics_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_dynamics_variances),
                train_times=self.get_first_n_trajectories(n_trajectories_to_plot, self.times),
                observations=self.get_first_n_trajectories(n_trajectories_to_plot, self.observations),
                all_initial_conditions=None,
                denormalized_obs_noise_variances=self.get_denormalized_smoother_observation_noise_variance()
            )
        figure_smoother_states.tight_layout()
        figure_smoother_derivatives.tight_layout()
        figure_dynamics_derivatives.tight_layout()

        state_filename = 'smoother_states'
        if self.track_wandb:
            wandb.log({state_filename: wandb.Image(figure_smoother_states),
                       'smoother_derivatives': wandb.Image(figure_smoother_derivatives),
                       'dynamics_derivatives': wandb.Image(figure_dynamics_derivatives)})

    @staticmethod
    def get_first_n_trajectories(n: int, value: List[List[jnp.array]]):
        return_values = []
        for dimension_id in range(len(value)):
            return_values_of_this_dimension = []
            for trajectory_id in range(n):
                return_values_of_this_dimension.append(value[dimension_id][trajectory_id])
            return_values.append(return_values_of_this_dimension)
        return return_values

    def load_parameters_from_wandb(self, entity_string, project_string, run_string, run_args, model_file_suffix,
                                   group_string=None, group_args=[]):

        run_db = RunDB(project_string=project_string,
                       group_string=group_string,
                       run_string=run_string,
                       entity_string=entity_string
                       )
        run_id = run_db.get_run_id(group_args, run_args)
        run_path = os.path.join(entity_string, project_string, run_id)
        wandb.restore(name=f"models/model_{model_file_suffix}.pkl", run_path=run_path)

        directory = os.path.join(wandb.run.dir, 'models')
        data_path = os.path.join(directory, f"model_{model_file_suffix}.pkl")
        with open(data_path, 'rb') as file:
            self.parameters = pickle.load(file)


class LearnUSHCNPredictor(AbstractLearnSystem):

    def __init__(
        self,
        obs_std_prior: Dict,
        seed: int,
        data_generation: Dict,
        smoother: Dict,
        dynamics: Dict,
        betas: Dict,
        optimizer: Dict,
        priors: Dict,
        logging: Dict,
        numerical_correction: float,
        num_derivative_points_per_trajectory: int,
        split_pretraining: Optional[bool]=True,
        create_equidistant_derivative_times: Optional[bool] = True,
        ignore_third_dimension_input_for_deriv_times: Optional[bool] = True,
        spectral_regularization_weight_smoother: Optional[float] = 0,
        spectral_regularization_weight_dynamics: Optional[float] = 0,
        use_spectral_normalization: Optional[bool] = False,
        n_l2_pretraining_for_dynamics: Optional[int] = 0
    ):
        self.plotter = Plotter(simulator=None, initial_conditions=None)

        self.test_time_ratio_random = data_generation.get('test_time_ratio_random', 0)
        self.test_time_ratio_consecutive = data_generation.get('test_time_ratio_consecutive', 0)
        self.simulator_type = data_generation['type']
        self.path_to_data = data_generation['path_to_data']
        self.path_to_mapping = data_generation.get('path_to_mapping', None)
        self.path_to_stations_file = data_generation.get('path_to_stations_file', None)
        self.ic_extraction_method = data_generation['ic_extractor']
        self.n_trajectories_to_consider = data_generation['n_trajectories_to_consider']
        self.dimensions_to_consider = data_generation['dimensions_to_consider']
        self.min_obs_per_trajectory = data_generation['min_obs_per_trajectory']
        self.load_augmented_states = data_generation.get('load_augmented_states', False)
        self.max_ic_time_slack = data_generation.get('max_ic_time_slack', -1)
        self.extract_ic = get_initial_condition_extractor(self.ic_extraction_method)

        super().__init__(seed=seed,
                         obs_std_prior=obs_std_prior,
                         smoother=smoother,
                         dynamics=dynamics,
                         betas=betas,
                         optimizer=optimizer,
                         priors=priors,
                         logging=logging,
                         numerical_correction=numerical_correction,
                         num_derivative_points_per_trajectory=num_derivative_points_per_trajectory,
                         split_pretraining=split_pretraining,
                         create_equidistant_derivative_times=create_equidistant_derivative_times,
                         spectral_regularization_weight_smoother=spectral_regularization_weight_smoother,
                         spectral_regularization_weight_dynamics=spectral_regularization_weight_dynamics,
                         use_spectral_normalization=use_spectral_normalization,
                         n_l2_pretraining_for_dynamics=n_l2_pretraining_for_dynamics,
                         ignore_third_dimension_input_for_deriv_times=ignore_third_dimension_input_for_deriv_times
        )

    def get_denormalized_smoother_observation_noise_variance(self):
        normalized_sigma = make_positive(self.parameters['smoother']['smoother']['noise_std'])
        denormalized_obs_noise_var = []
        for dim_id in range(normalized_sigma.size):
            denormalized_obs_noise_var.append(
                self.state_normalizers[dim_id].scale_ ** 2 * normalized_sigma[dim_id]**2
            )
        return denormalized_obs_noise_var

    def get_metrics(self, initial_conditions, times, observations, quantile=0.8, ignore_third_dimension=False):
        """ initial conditions and times are not normalized """

        """ calculate predictions """

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            initial_conditions, times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        if sum(trajectory_lengths[2]) == 0:
            # here, third dimension is empty because we are in test set
            # as a hack for the rest of the code to work, we introduce a fake time and observation, which is later ignored
            ignore_third_dimension = True  # ignore third dimension when calculating the metric, as it only contains the fake obs
            joint_normalized_times[2] = np.array([1.1])
            joint_repeated_normalized_initial_conditions[2] = 0.1*np.ones_like(joint_repeated_normalized_initial_conditions[0][0]).reshape(1, -1)
            trajectory_lengths[2][0] = 1

        predictions = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        denormalized_state_means = predictions[0]
        denormalized_state_variances = predictions[1]

        # Prepare (not normalized) ground truth prediction
        self.current_rng, subkey = jax.random.split(self.current_rng)

        denormalized_observation_noise_variances = self.get_denormalized_smoother_observation_noise_variance()

        """ calculate nlls """

        # Compute average (per dimension) average NLL score
        nll_state = self._compute_nll_per_dimension_for_state_observations(
            denormalized_state_means, denormalized_state_variances,
            denormalized_observation_noise_variances,
            observations,
            ignore_third_dimension=ignore_third_dimension
        )

        nll_state = jnp.array(nll_state)
        nll_state_median = jnp.median(nll_state)
        nll_state_lower_q = jnp.quantile(nll_state, q=1 - quantile)
        nll_state_upper_q = jnp.quantile(nll_state, q=quantile)
        nll_state_mean = jnp.mean(nll_state)

        """ calculate rmse """
        state_rmses = []
        # Compute average (over dimension) average NLL score
        # Not over the range of self.num_trajectories but over the range of number of evaluated trajectories
        for trajectory_id in range((len(denormalized_state_means[0]))):
            current_observations = []
            current_state_means = []
            for dimension_id in range(len(denormalized_state_means)):
                if ignore_third_dimension and dimension_id==2:
                    continue
                else:
                    current_observations.append(observations[dimension_id][trajectory_id])
                    current_state_means.append(denormalized_state_means[dimension_id][trajectory_id])

            current_observations = jnp.concatenate(current_observations)
            current_state_means = jnp.concatenate(current_state_means)

            state_rmses.append(
                jnp.sqrt(jnp.mean(jnp.square(current_observations - current_state_means)))
            )

        rmse_state = jnp.array(state_rmses)
        rmse_state_median = jnp.median(rmse_state)
        rmse_state_lower_q = jnp.quantile(rmse_state, q=1-quantile)
        rmse_state_upper_q = jnp.quantile(rmse_state, q=quantile)
        rmse_state_mean = jnp.mean(rmse_state)

        return\
            nll_state_median, nll_state_lower_q, nll_state_upper_q, nll_state_mean, \
            rmse_state_median, rmse_state_lower_q, rmse_state_upper_q, rmse_state_mean

    def _compute_nll_per_dimension_for_state_observations(
            self,
            denormalized_state_means,
            denormalized_state_variances,
            denormalized_observation_noise_variances,
            observations,
            ignore_third_dimension=False
    ):
        """
        all arguments are lists over dimensions over trajectories

        Returns a list of nlls over trajectories
        """
        nll_state = []

        # Compute average (over dimension) average NLL score
        # Not over the range of self.num_trajectories but over the range of number of evaluated trajectories
        for trajectory_id in range((len(denormalized_state_means[0]))):
            current_observations = []
            current_state_means = []
            current_state_variances = []
            for dimension_id in range(len(denormalized_state_means)):
                if ignore_third_dimension and (dimension_id == 2):
                    continue
                current_observations.append(observations[dimension_id][trajectory_id])
                current_state_means.append(denormalized_state_means[dimension_id][trajectory_id])
                current_state_variances.append(
                    denormalized_state_variances[dimension_id][trajectory_id]
                    + denormalized_observation_noise_variances[dimension_id]
                )

            current_observations = jnp.concatenate(current_observations)
            current_state_means = jnp.concatenate(current_state_means)
            current_state_variances = jnp.concatenate(current_state_variances)

            nll_state.append(
                self._mean_nll_of_independent_gaussians(
                    current_observations, current_state_means, current_state_variances)
            )
        return nll_state

    @staticmethod
    def store_metrics_in_wandb(metrics, name):
        (
            nll_state_median, nll_state_lower_q, nll_state_upper_q, nll_state_mean,
            rmse_state_median, rmse_state_lower_q, rmse_state_upper_q, rmse_state_mean
        ) = metrics

        variable_dict = dict()

        variable_dict[name + "_nll_state_median"] = nll_state_median
        variable_dict[name + "_nll_state_lower_q"] = nll_state_lower_q
        variable_dict[name + "_nll_state_upper_q"] = nll_state_upper_q
        variable_dict[name + "_nll_state_mean"] = nll_state_mean

        variable_dict[name + "_rmse_state_median"] = rmse_state_median
        variable_dict[name + "_rmse_state_lower_q"] = rmse_state_lower_q
        variable_dict[name + "_rmse_state_upper_q"] = rmse_state_upper_q
        variable_dict[name + "_rmse_state_mean"] = rmse_state_mean

        wandb.log(variable_dict)

    def calculate_metrics_and_push_to_wandb_on_train(self, name, ignore_third_dimension=False):
        train_initial_conditions = self.initial_conditions
        train_times = self.times
        train_observations = self.observations
        metrics = self.get_metrics(train_initial_conditions, train_times, train_observations, quantile=0.8,
                                   ignore_third_dimension=ignore_third_dimension)
        if self.track_wandb:
            self.store_metrics_in_wandb(metrics, name + "_train")

    def calculate_metrics_and_push_to_wandb_on_test(self, name, ignore_third_dimension=False):
        test_initial_conditions = self.test_initial_conditions
        test_times = self.test_times
        test_observations = self.test_observations
        metrics = self.get_metrics(test_initial_conditions, test_times, test_observations, quantile=0.8,
                                   ignore_third_dimension=ignore_third_dimension)
        if self.track_wandb:
            self.store_metrics_in_wandb(metrics, name + "_test")

    def calculate_nll_and_rmse_on_trajectory_by_id(self, trajectory_id: int, ignore_third_dimension=False):
        initial_conditions, times, observations = self.get_information_for_trajectory_of_id(trajectory_id)
        metrics = self.get_metrics(
            initial_conditions, times, observations, quantile=0.8, ignore_third_dimension=ignore_third_dimension)
        return metrics[0], metrics[4]

    def calculate_metrics_and_push_to_wandb_on_all_data(self, name, ignore_third_dimension=False):
        all_initial_conditions = self.all_initial_conditions
        all_times = self.all_times
        all_observations = self.all_observations
        metrics = self.get_metrics(all_initial_conditions, all_times, all_observations, quantile=0.8,
                                   ignore_third_dimension=ignore_third_dimension)
        if self.track_wandb:
            self.store_metrics_in_wandb(metrics, name + "_all")

    def store_parameters(self, name="pretrained"):
        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_path = os.path.join(directory, 'model_' + name + '.pkl')
            with open(data_path, 'wb') as file:
                pickle.dump(self.parameters, file)
            wandb.save(data_path)

    def _get_ground_truth(self, initial_conditions, times, rng):
        print("This is a hack, instead of truth, we will simply return the real observations")
        if len(times[0][0]) == len(self.times[0][0]):
            return self.ground_truth_states, self.ground_truth_derivatives
        elif len(times[0][0]) == len(self.test_times[0][0]):
            return self.test_observations, self.test_observations
        elif len(times[0][0]) == len(self.all_times[0][0]):
            return self.all_observations, self.all_observations
        else:
            raise BaseException("There is no simulator to provide truths for times outside of train / test splits")

    def _create_data(self):
        original_data = load_raw_data(self.path_to_data)
        times, values = extract_time_and_value_arrays_from_GRU_data(original_data, augmented=self.load_augmented_states)
        self.times, self.observations = format_data_for_dgm(times, values, augmented_states=self.load_augmented_states)

        # only use dimensions that are interesting
        self.times = [self.times[i] for i in self.dimensions_to_consider]
        self.observations = [self.observations[i] for i in self.dimensions_to_consider]

        # train / testing split
        train_times, train_obs, test_times, test_obs = self.get_train_test_split(self.times, self.observations)

        # recover initial conditions if necessary, see interpolator

        # only use trajectories with minimal number of obs
        train_times, train_obs, test_times, test_obs = \
            extract_trajectories_with_more_than_n_train_obs_per_dim_and_3_val_obs(
                self.min_obs_per_trajectory,
                train_times, train_obs,
                test_times, test_obs
            )

        # only use the first few trajectories, if desired
        if len(train_times[0]) > self.n_trajectories_to_consider:
            train_times = self.extract_first_n_trajectories(self.n_trajectories_to_consider, train_times)
            train_obs = self.extract_first_n_trajectories(self.n_trajectories_to_consider, train_obs)
            test_times = self.extract_first_n_trajectories(self.n_trajectories_to_consider, test_times)
            test_obs = self.extract_first_n_trajectories(self.n_trajectories_to_consider, test_obs)

        self.all_times = self.merge_train_test_split(train_times, test_times)
        self.all_observations = self.merge_train_test_split(train_obs, test_obs)
        self.all_initial_conditions = self.extract_ic(self.all_times, self.all_observations)

        self.times = train_times
        self.observations = train_obs
        self.initial_conditions = self.extract_ic(self.times, self.observations)
        self.num_trajectories = len(self.initial_conditions)
        self.state_dimension = len(self.initial_conditions[0])

        self.test_times = test_times
        self.test_observations = test_obs
        self.test_initial_conditions = self.extract_ic(self.times, self.observations)

        # TODO: maybe do something different for

        self.ground_truth_states = self.observations
        self.ground_truth_derivatives = self.observations  # hack!! No derivatives available for real data

    def get_information_for_trajectory_of_id(self, trajectory_id):
        # this should not be abstract here. It should be implemented. But not done due to time constraints.
        # initial_conditions = None
        # times = None
        # observations = None
        #
        # return initial_conditions, times, observations

        raise NotImplementedError("Please implement alongside _create_data")

    @staticmethod
    def extract_first_n_trajectories(n, values):
        new_values = []
        for dim_id in range(len(values)):
            new_values.append(values[dim_id][:n])
        return new_values

    @staticmethod
    def get_train_test_split(times, observations, splitting_time=150):
        """
        for each trajectory, extracts all observations before time 150 for training and gets all future values for
        validation
        """
        def get_train_test_split_for_one_value(values):
            train_values = []
            test_values = []
            for dim_id in range(len(times)):
                train_values_for_this_dim = []
                test_values_for_this_dim = []
                for traj_id in range(len(times[dim_id])):
                    train_values_for_this_traj = []
                    test_values_for_this_traj = []
                    for idx, curr_time in enumerate(times[dim_id][traj_id]):
                        if curr_time < splitting_time:
                            train_values_for_this_traj.append(values[dim_id][traj_id][idx])
                        else:
                            test_values_for_this_traj.append(values[dim_id][traj_id][idx])
                    train_values_for_this_dim.append(jnp.asarray(train_values_for_this_traj))
                    test_values_for_this_dim.append(jnp.asarray(test_values_for_this_traj))
                train_values.append(train_values_for_this_dim)
                test_values.append(test_values_for_this_dim)
            return train_values, test_values

        train_times, test_times = get_train_test_split_for_one_value(times)
        train_obs, test_obs = get_train_test_split_for_one_value(observations)

        return train_times, train_obs, test_times, test_obs

    def plot_first_few_trajectories_at_all_times(self, n_trajectories_to_plot, plot_equidistantly=False, n_points=150,
                                                 name_prefix=""):
        print('Before computing the values for plotting')
        if plot_equidistantly:
            raw_times = self.all_times
            plotting_times = []
            for dim_id in range(self.state_dimension):
                plotting_times_for_this_dimension = []
                for traj_id in range(self.num_trajectories):
                    plotting_times_for_this_dimension.append(
                        jnp.linspace(min(raw_times[dim_id][traj_id]),
                                     max(raw_times[dim_id][traj_id]),
                                     n_points)
                    )
                plotting_times.append(plotting_times_for_this_dimension)
        else:
            plotting_times = self.all_times

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            self.all_initial_conditions, plotting_times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
        denormalized_derivative_means, denormalized_derivative_variances, \
        denormalized_dynamics_means, denormalized_dynamics_variances = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        n_trajectories = len(denormalized_state_means[0])
        if n_trajectories < n_trajectories_to_plot:
            n_trajectories_to_plot = n_trajectories
            print("Only {} trajectories available. All trajectories will be printed")

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                self.get_first_n_trajectories(n_trajectories_to_plot, plotting_times),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_state_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_state_variances),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_derivative_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_derivative_variances),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_dynamics_means),
                self.get_first_n_trajectories(n_trajectories_to_plot, denormalized_dynamics_variances),
                train_times=self.get_first_n_trajectories(n_trajectories_to_plot, self.times),
                train_observations=self.get_first_n_trajectories(n_trajectories_to_plot, self.observations),
                test_times=self.get_first_n_trajectories(n_trajectories_to_plot, self.test_times),
                test_observations=self.get_first_n_trajectories(n_trajectories_to_plot, self.test_observations),
                all_initial_conditions=None,
                denormalized_obs_noise_variances=self.get_denormalized_smoother_observation_noise_variance()
            )
        figure_smoother_states.tight_layout()
        figure_smoother_derivatives.tight_layout()
        figure_dynamics_derivatives.tight_layout()

        state_filename = 'smoother_states'
        if self.track_wandb:
            wandb.log({name_prefix + state_filename: wandb.Image(figure_smoother_states),
                       name_prefix + 'smoother_derivatives': wandb.Image(figure_smoother_derivatives),
                       name_prefix + 'dynamics_derivatives': wandb.Image(figure_dynamics_derivatives)})

    @staticmethod
    def merge_train_test_split(train_values, test_values):
        n_dimensions = len(train_values)
        n_trajectories = len(train_values[0])

        merged_values = []
        for dim_id in range(n_dimensions):
            merged_values_for_this_dim = []
            for traj_id in range(n_trajectories):
                merged_values_for_this_dim.append(jnp.concatenate(
                    [
                        train_values[dim_id][traj_id],
                        test_values[dim_id][traj_id]
                    ]))
            merged_values.append(merged_values_for_this_dim)
        return merged_values


class LearnUSHCNInterpolator(LearnUSHCNPredictor):

    def __init__(
        self,
        seed: int,
        data_generation: Dict,
        smoother: Dict,
        dynamics: Dict,
        betas: Dict,
        optimizer: Dict,
        priors: Dict,
        logging: Dict,
        numerical_correction: float,
        num_derivative_points_per_trajectory: int,
        split_pretraining: Optional[bool] = True,
        create_equidistant_derivative_times: Optional[bool] = True,
        ignore_third_dimension_input_for_deriv_times: Optional[bool] = True,
        obs_std_prior: Optional[Dict] = None,
        spectral_regularization_weight_smoother: Optional[float] = 0,
        spectral_regularization_weight_dynamics: Optional[float] = 0,
        use_spectral_normalization: Optional[bool] = False,
        n_l2_pretraining_for_dynamics: Optional[int] = 0
    ):

        super().__init__(seed=seed,
                         data_generation=data_generation,
                         smoother=smoother,
                         dynamics=dynamics,
                         betas=betas,
                         optimizer=optimizer,
                         priors=priors,
                         logging=logging,
                         numerical_correction=numerical_correction,
                         num_derivative_points_per_trajectory=num_derivative_points_per_trajectory,
                         split_pretraining=split_pretraining,
                         create_equidistant_derivative_times=create_equidistant_derivative_times,
                         obs_std_prior=obs_std_prior,
                         spectral_regularization_weight_smoother=spectral_regularization_weight_smoother,
                         spectral_regularization_weight_dynamics=spectral_regularization_weight_dynamics,
                         use_spectral_normalization=use_spectral_normalization,
                         n_l2_pretraining_for_dynamics=n_l2_pretraining_for_dynamics,
                         ignore_third_dimension_input_for_deriv_times=ignore_third_dimension_input_for_deriv_times
                         )

    def _create_data(self):
        original_data = load_raw_data(self.path_to_data)
        times, values = extract_time_and_value_arrays_from_GRU_data(original_data, augmented=self.load_augmented_states)
        self.times, self.observations = format_data_for_dgm(times, values, augmented_states=self.load_augmented_states)

        # only use dimensions that are interesting
        self.times = [self.times[i] for i in self.dimensions_to_consider]
        self.observations = [self.observations[i] for i in self.dimensions_to_consider]

        # kill all trajectories with not enough obs
        self.times, self.observations = extract_trajectories_with_more_than_n_obs_per_dim(
            self.min_obs_per_trajectory,
            self.times, self.observations
        )

        # kill all trajectoriese with not sufficiently early initial conditions
        self.times, self.observations = extract_trajectories_with_ics_within_slack(self.max_ic_time_slack,
                                                                                   self.times,
                                                                                   self.observations
                                                                                   )

        # train / testing split
        train_times, train_obs, test_times, test_obs = self.get_train_test_split(self.times,
                                                                                 self.observations,
                                                                                 None,
                                                                                 )

        # only use the first few trajectories, if desired
        if len(train_times[0]) > self.n_trajectories_to_consider:
            train_times = self.extract_first_n_trajectories(self.n_trajectories_to_consider, train_times)
            train_obs = self.extract_first_n_trajectories(self.n_trajectories_to_consider, train_obs)
            test_times = self.extract_first_n_trajectories(self.n_trajectories_to_consider, test_times)
            test_obs = self.extract_first_n_trajectories(self.n_trajectories_to_consider, test_obs)

        self.all_times = self.merge_train_test_split(train_times, test_times)
        self.all_observations = self.merge_train_test_split(train_obs, test_obs)
        self.all_initial_conditions = self.extract_ic(self.all_times, self.all_observations)

        self.times = train_times
        self.observations = train_obs
        self.initial_conditions = self.extract_ic(self.times, self.observations)
        self.num_trajectories = len(self.initial_conditions)
        self.state_dimension = len(self.initial_conditions[0])

        self.test_times = test_times
        self.test_observations = test_obs
        self.test_initial_conditions = self.extract_ic(self.times, self.observations)
        # use same initial conditions as for training, since we explicitly assume that we only have validation stuff
        # on the trajectories we train on.

        # TODO: maybe do something different for

        self.ground_truth_states = self.observations
        self.ground_truth_derivatives = self.observations  # hack!! No derivatives available for real data

    def get_train_test_split(self, times, observations, test_indices, dims_just_for_training=[2]):
        """
        splits times and observations over time, not over trajectories.
        test_indices as input is irrelevant
        dimensions whose index is in dims_just_for_training will be used exclusively for training
        """
        train_times = []
        train_obs = []
        test_times = []
        test_obs = []
        self.join_test_train_times = False

        # sort times and observations
        for dim_id in range(len(times)):
            for traj_id in range(len(times[0])):
                current_ordering = np.argsort(times[dim_id][traj_id])
                times[dim_id][traj_id] = times[dim_id][traj_id][current_ordering]
                observations[dim_id][traj_id] = observations[dim_id][traj_id][current_ordering]

        if self.test_time_ratio_random > 0:
            self.join_test_train_times = True
            for dim_id in range(len(times)):
                train_times_of_this_dim = []
                train_obs_of_this_dim = []
                test_times_of_this_dim = []
                test_obs_of_this_dim = []
                total_train_time_length_of_this_dim = 0
                total_train_obs_length_of_this_dim = 0
                for traj_id in range(len(times[0])):
                    if dim_id in dims_just_for_training:
                        train_times_of_this_dim.append(np.asarray(times[dim_id][traj_id]))
                        train_obs_of_this_dim.append(np.asarray(observations[dim_id][traj_id]))
                        test_times_of_this_dim.append(np.asarray([]))
                        test_obs_of_this_dim.append(np.asarray([]))
                    else:
                        all_indices = np.arange(1, len(times[dim_id][traj_id]))  # take out initial condition when calculating test indices
                        n_test_indices = int(max(np.floor((len(all_indices)+1)*self.test_time_ratio_random), 1))
                        test_indices = np.random.choice(all_indices, size=n_test_indices, replace=False)
                        all_indices = np.arange(len(times[dim_id][traj_id]))  # consider initial condition for set difference
                        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
                        train_times_of_this_dim.append(times[dim_id][traj_id][train_indices])
                        train_obs_of_this_dim.append(observations[dim_id][traj_id][train_indices])
                        test_times_of_this_dim.append(times[dim_id][traj_id][test_indices])
                        test_obs_of_this_dim.append(observations[dim_id][traj_id][test_indices])
                        total_train_obs_length_of_this_dim += len(train_obs_of_this_dim[-1])
                        total_train_time_length_of_this_dim += len(train_times_of_this_dim[-1])
                train_times.append(train_times_of_this_dim)
                train_obs.append(train_obs_of_this_dim)
                test_times.append(test_times_of_this_dim)
                test_obs.append(test_obs_of_this_dim)
            return train_times, train_obs, test_times, test_obs

        if self.test_time_ratio_consecutive > 0:
            self.join_test_train_times = True
            for dim_id in range(len(times)):
                train_times_of_this_dim = []
                train_obs_of_this_dim = []
                test_times_of_this_dim = []
                test_obs_of_this_dim = []
                # total_train_time_length_of_this_dim = 0
                # total_train_obs_length_of_this_dim = 0
                for traj_id in range(len(times[0])):
                    if dim_id in dims_just_for_training:
                        train_times_of_this_dim.append(np.asarray(times[dim_id][traj_id]))
                        train_obs_of_this_dim.append(np.asarray(observations[dim_id][traj_id]))
                        test_times_of_this_dim.append(np.asarray([]))
                        test_obs_of_this_dim.append(np.asarray([]))
                    else:
                        all_indices = np.arange(1, len(times[dim_id][traj_id]))  # take out ic for test indices
                        n_test_indices = int(max(np.floor((len(all_indices)+1)*self.test_time_ratio_consecutive), 1))
                        # print(f"traj_id: {traj_id} dim_id: {dim_id}")
                        # if traj_id == 319:
                        #     print("Hi")
                        start_test_index = np.random.choice(len(all_indices) - n_test_indices+1)+1
                        test_indices = np.arange(start_test_index, start_test_index + n_test_indices)
                        all_indices = np.arange(len(times[dim_id][traj_id]))
                        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
                        train_times_of_this_dim.append(times[dim_id][traj_id][train_indices])
                        train_obs_of_this_dim.append(observations[dim_id][traj_id][train_indices])
                        test_times_of_this_dim.append(times[dim_id][traj_id][test_indices])
                        test_obs_of_this_dim.append(observations[dim_id][traj_id][test_indices])
                        """
                        print(f"traj_id_{traj_id}_dim_id_{dim_id}")
                        assert len(train_times_of_this_dim[-1]) == len(train_obs_of_this_dim[-1])
                        total_train_obs_length_of_this_dim += len(train_obs_of_this_dim[-1])
                        total_train_time_length_of_this_dim += len(train_times_of_this_dim[-1])
                        """
                train_times.append(train_times_of_this_dim)
                train_obs.append(train_obs_of_this_dim)
                test_times.append(test_times_of_this_dim)
                test_obs.append(test_obs_of_this_dim)
            return train_times, train_obs, test_times, test_obs

        return times, observations, times, observations
        # this code currently does not work, since test and train need same number of trajectories
        # def split_one_value(values):
        #     test_values = []
        #     train_values = []
        #     for dim_id in range(len(values)):
        #         test_values_for_this_dim = []
        #         train_values_for_this_dim = []
        #         for traj_id in range(len(values[0])):
        #             if traj_id in test_indices:
        #                 test_values_for_this_dim.append(values[dim_id][traj_id])
        #             else:
        #                 train_values_for_this_dim.append(values[dim_id][traj_id])
        #         test_values.append(test_values_for_this_dim)
        #         train_values.append(train_values_for_this_dim)
        #     return train_values, test_values
        #
        # train_times, test_times = split_one_value(times)
        # train_obs, test_obs = split_one_value(observations)
        #
        # return train_times, train_obs, test_times, test_obs

    def merge_train_test_split(self, train_values, test_values):
        if self.join_test_train_times:
            merged_values = []
            for dim_id in range(len(train_values)):
                merged_values_for_this_dim = []
                for traj_id in range(len(train_values[0])):
                    merged_values_for_this_dim.append(
                        jnp.array(train_values[dim_id][traj_id].tolist() + test_values[dim_id][traj_id].tolist()))
                merged_values.append(merged_values_for_this_dim)
            return merged_values
        return train_values
        # see function right above
        # n_dimensions = len(train_values)
        #
        # merged_values = []
        # for dim_id in range(n_dimensions):
        #     merged_values.append(train_values[dim_id] + test_values[dim_id])
        # return merged_values


class LearnUSHCNInterpolatorIC(LearnUSHCNInterpolator):

    def __init__(
        self,
        seed: int,
        data_generation: Dict,
        smoother: Dict,
        dynamics: Dict,
        betas: Dict,
        optimizer: Dict,
        priors: Dict,
        logging: Dict,
        numerical_correction: float,
        num_derivative_points_per_trajectory: int,
        split_pretraining: Optional[bool] = True,
        create_equidistant_derivative_times: Optional[bool] = True,
        ignore_third_dimension_input_for_deriv_times: Optional[bool] = True,
        obs_std_prior: Optional[Dict] = None,
        spectral_regularization_weight_smoother: Optional[float] = 0,
        spectral_regularization_weight_dynamics: Optional[float] = 0,
        use_spectral_normalization: Optional[bool] = False,
        n_l2_pretraining_for_dynamics: Optional[int] = 0,
        random_seed_test_train_split: Optional[int] = None
    ):
        self.random_seed_test_train_split = random_seed_test_train_split

        super().__init__(seed=seed,
                         data_generation=data_generation,
                         smoother=smoother,
                         dynamics=dynamics,
                         betas=betas,
                         optimizer=optimizer,
                         priors=priors,
                         logging=logging,
                         numerical_correction=numerical_correction,
                         num_derivative_points_per_trajectory=num_derivative_points_per_trajectory,
                         split_pretraining=split_pretraining,
                         create_equidistant_derivative_times=create_equidistant_derivative_times,
                         obs_std_prior=obs_std_prior,
                         spectral_regularization_weight_smoother=spectral_regularization_weight_smoother,
                         spectral_regularization_weight_dynamics=spectral_regularization_weight_dynamics,
                         use_spectral_normalization=use_spectral_normalization,
                         n_l2_pretraining_for_dynamics=n_l2_pretraining_for_dynamics,
                         ignore_third_dimension_input_for_deriv_times=ignore_third_dimension_input_for_deriv_times
                         )

    def _normalize_initial_conditions(self):
        self.ic_normalizers = [StandardScaler() for _ in range(self.state_dimension)]

        for dimension_id in range(self.state_dimension):
            ic_of_this_dim = [ic[dimension_id] for ic in self.initial_conditions]
            ic_of_this_dim = np.asarray(ic_of_this_dim)
            self.ic_normalizers[dimension_id].fit(ic_of_this_dim.reshape(-1, 1))

            current_normalized_initial_conditions = []
            for trajectory_id in range(self.num_trajectories):
                current_normalized_initial_conditions.append(
                    self.ic_normalizers[dimension_id].transform(
                        self.initial_conditions[trajectory_id][dimension_id].reshape(-1, 1)
                    )
                )
            self.normalized_initial_conditions.append(current_normalized_initial_conditions)

    def plot_trajectories_by_id(self, trajectory_ids: List[int], name_suffix=""):
        initial_conditions = []
        times = [[], [], []]
        observations = [[], [], []]

        for trajectory_id in trajectory_ids:
            new_ic, new_times, new_obs = self.get_information_for_trajectory_of_id(trajectory_id)
            initial_conditions += new_ic
            for dim_id in range(len(times)):
                times[dim_id] += new_times[dim_id]
                observations[dim_id] += new_obs[dim_id]

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            initial_conditions, times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        denormalized_state_means, denormalized_state_variances, \
            denormalized_derivative_means, denormalized_derivative_variances, \
            denormalized_dynamics_means, denormalized_dynamics_variances = self._get_split_predictions(
                joint_normalized_times,
                joint_repeated_normalized_initial_conditions,
                trajectory_lengths)

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                prediction_times=times,
                state_means=denormalized_state_means,
                state_variances=denormalized_state_variances,
                derivatives_means=denormalized_derivative_means,
                derivative_variances=denormalized_derivative_variances,
                dynamics_means=denormalized_dynamics_means,
                dynamics_variances=denormalized_dynamics_variances,
                train_times=times,
                train_observations=observations,
                all_initial_conditions=None,
                denormalized_obs_noise_variances=self.get_denormalized_smoother_observation_noise_variance()
            )
        figure_smoother_states.tight_layout()
        figure_smoother_derivatives.tight_layout()
        figure_dynamics_derivatives.tight_layout()

        state_filename = 'smoother_states_' + name_suffix
        if self.track_wandb:
            wandb.log({state_filename: wandb.Image(figure_smoother_states),
                       'smoother_derivatives': wandb.Image(figure_smoother_derivatives),
                       'dynamics_derivatives': wandb.Image(figure_dynamics_derivatives)})

    @staticmethod
    def extract_value_by_id(value_id, values):
        new_values = []
        for dim_id in range(len(values)):
            new_values.append([values[dim_id][value_id]])
        return new_values

    def _create_data(self):
        original_data = load_raw_data(self.path_to_data)
        position_extractor = PositionExtractor(path_to_stations_file=self.path_to_stations_file,
                                               path_to_mapping=self.path_to_mapping,
                                               )
        times, values, initial_conditions = extract_time_value_and_position_arrays_from_GRU_data(
            original_data, position_extractor, augmented=self.load_augmented_states)
        self.times, self.observations = format_data_for_dgm(times, values, augmented_states=self.load_augmented_states)

        # only use dimensions that are interesting
        self.times = [self.times[i] for i in self.dimensions_to_consider]
        self.observations = [self.observations[i] for i in self.dimensions_to_consider]

        # kill all trajectories with not enough obs
        self.times, self.observations, initial_conditions = extract_trajectories_with_more_than_n_obs_per_dim(
            self.min_obs_per_trajectory,
            self.times, self.observations, initial_conditions=initial_conditions
        )

        # train / testing split
        train_times, train_obs, test_times, test_obs = self.get_train_test_split(self.times,
                                                                                 self.observations,
                                                                                 None,
                                                                                 )

        self.times_of_full_dataset = self.times
        self.obs_of_full_dataset = self.observations
        self.ic_of_full_dataset = initial_conditions

        # only use the first few trajectories, if desired
        if len(train_times[0]) <= self.n_trajectories_to_consider:
            raise ValueError("Too many training trajectories desired")

        # randomly subsample trajectories if necessary
        if self.random_seed_test_train_split:
            rng = jax.random.PRNGKey(self.random_seed_test_train_split)
            self.train_indices = jax.random.choice(key=rng,
                                                   a=len(train_times[0]),
                                                   shape=[self.n_trajectories_to_consider],
                                                   replace=False,
                                                   )
        else:
            self.train_indices = jnp.arange(self.n_trajectories_to_consider)
        self.test_indices = jnp.setdiff1d(jnp.arange(len(train_times[0])),
                                          self.train_indices)

        # extract_by_indices
        train_times = self.extract_trajectories_by_indices(train_times, self.train_indices)
        train_obs = self.extract_trajectories_by_indices(train_obs, self.train_indices)
        test_times = self.extract_trajectories_by_indices(test_times, self.train_indices)
        test_obs = self.extract_trajectories_by_indices(test_obs, self.train_indices)

        initial_conditions = [initial_conditions[a] for a in self.train_indices]

        self.all_times = self.merge_train_test_split(train_times, test_times)
        self.all_observations = self.merge_train_test_split(train_obs, test_obs)
        self.all_initial_conditions = initial_conditions

        self.times = train_times
        self.observations = train_obs
        self.initial_conditions = initial_conditions
        self.num_trajectories = len(self.initial_conditions)
        self.state_dimension = len(self.initial_conditions[0])

        self.test_times = test_times
        self.test_observations = test_obs
        self.test_initial_conditions = initial_conditions
        # use same initial conditions as for training, since we explicitly assume that we only have validation stuff
        # on the trajectories we train on.

        # TODO: maybe do something different for

        self.ground_truth_states = self.observations
        self.ground_truth_derivatives = self.observations  # hack!! No derivatives available for real data

    @staticmethod
    def extract_trajectories_by_indices(quantity, indices):
        return_quantity = []
        for dim_id in range(len(quantity)):
            return_quantity.append([quantity[dim_id][a] for a in indices])
        return return_quantity

    def get_initial_conditions_from_trajectory_ids(self, trajectory_ids, unnormalized_initial_time=0):
        """
        returns tuples containing mean and std of initial condition prediction of state at unnormalized_initial_time
        """
        initial_conditions = [self.ic_of_full_dataset[traj_id] for traj_id in trajectory_ids]

        times = []
        for dim_id in range(3):
            times_for_this_dim = []
            for traj_id in trajectory_ids:
                times_for_this_dim.append(jnp.array([unnormalized_initial_time]))
            times.append(times_for_this_dim)

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            initial_conditions, times)

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        predictions = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        denormalized_state_means = jnp.array(predictions[0]).swapaxes(0, 1)
        denormalized_state_variances = jnp.array(predictions[1]).swapaxes(0, 1)

        return denormalized_state_means, denormalized_state_variances

    def extract_prediction_information_for_trajectory_of_id(self, trajectory_id, start_of_times, end_of_times):
        raise NotImplementedError("Please implement")

    def get_information_for_trajectory_of_id(self, trajectory_id, start_of_times=0, end_of_times=None):
        if end_of_times:
            return self.extract_prediction_information_for_trajectory_of_id(trajectory_id, start_of_times, end_of_times)
        else:
            # returns
            initial_conditions = [self.ic_of_full_dataset[trajectory_id]]
            times = self.extract_value_by_id(trajectory_id, self.times_of_full_dataset)
            observations = self.extract_value_by_id(trajectory_id, self.obs_of_full_dataset)
            return initial_conditions, times, observations

    def plot_trajectories_with_numerical_integration_by_id(
        self,
        trajectory_ids: List[int],
        unnormalized_initial_condition_means,  # List over trajectories with means in it
        unnormalized_initial_condition_variances,  # List over trajectoreis with variances in it
        rng,
        name_suffix="",
        n_samples=100,
        n_dim=3,
        start_of_times=0,
        end_of_times=None
    ):

        times = [[] for dim in range(n_dim)]
        observations = [[] for dim in range(n_dim)]
        list_of_means = []
        list_of_variances = []
        for index, trajectory_id in enumerate(trajectory_ids):

            unnormalized_pseudo_initial_condition, unnorm_obs_times_of_this_traj, unnorm_observations_of_this_traj = \
                self.get_information_for_trajectory_of_id(trajectory_id, start_of_times=start_of_times,
                                                          end_of_times=end_of_times
                                                          )

            for dim_id in range(n_dim):
                times[dim_id].append(unnorm_obs_times_of_this_traj[dim_id][0])
                observations[dim_id].append(unnorm_observations_of_this_traj[dim_id][0])  # 0 because only one trajectory. second dimension is traj_id


            rng, new_rng = jax.random.split(rng)
            current_state_mean, current_state_var = self.get_trajectory_moments_via_numerical_integration(
                unnormalized_pseudo_initial_condition=unnormalized_pseudo_initial_condition[0],
                unnormalized_initial_condition_mean=unnormalized_initial_condition_means[index],
                unnormalized_initial_condition_variance=unnormalized_initial_condition_variances[index],
                unnormalized_evaluation_times=unnorm_obs_times_of_this_traj,
                rng=new_rng,
                n_samples=n_samples)

            list_of_means.append(current_state_mean)
            list_of_variances.append(current_state_var)

        state_means = []
        state_variances = []
        for dim_id in range(len(list_of_means[0])):
            state_means_of_this_dim = []
            state_vars_of_this_dim = []
            for traj_id in range(len(list_of_means)):
                state_means_of_this_dim.append(list_of_means[traj_id][dim_id][0])
                state_vars_of_this_dim.append(list_of_variances[traj_id][dim_id][0])
            state_means.append(state_means_of_this_dim)
            state_variances.append(state_vars_of_this_dim)

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                prediction_times=times,
                state_means=state_means,
                state_variances=state_variances,
                derivatives_means=state_means,
                derivative_variances=state_variances,
                dynamics_means=state_means,
                dynamics_variances=state_variances,
                train_times=times,
                train_observations=observations,
                all_initial_conditions=None,
                denormalized_obs_noise_variances=self.get_denormalized_smoother_observation_noise_variance()
            )
        figure_smoother_states.tight_layout()

        state_filename = 'smoother_states_' + name_suffix
        wandb.log({state_filename: wandb.Image(figure_smoother_states)})

    def get_data_for_final_ushcn_plots(
        self,
        trajectory_ids: List[int],
        unnormalized_initial_condition_means,  # List over trajectories with means in it
        unnormalized_initial_condition_variances,  # List over trajectoreis with variances in it
        rng,
        name_suffix="",
        n_samples=100,
        n_dim=3,
        start_of_times=0,
        end_of_times=None
    ):
        # create lists of times, observations, initial condition means and initial condition variances
        times = [[] for dim in range(n_dim)]
        observations = [[] for dim in range(n_dim)]
        list_of_means = []
        list_of_variances = []
        for index, trajectory_id in enumerate(trajectory_ids):

            unnormalized_pseudo_initial_condition, unnorm_obs_times_of_this_traj, unnorm_observations_of_this_traj = \
                self.get_information_for_trajectory_of_id(trajectory_id, start_of_times=start_of_times,
                                                          end_of_times=end_of_times
                                                          )

            for dim_id in range(n_dim):
                times[dim_id].append(unnorm_obs_times_of_this_traj[dim_id][0])
                observations[dim_id].append(unnorm_observations_of_this_traj[dim_id][0])  # 0 because only one trajectory. second dimension is traj_id

            rng, new_rng = jax.random.split(rng)
            current_state_mean, current_state_var = self.get_trajectory_moments_via_numerical_integration(
                unnormalized_pseudo_initial_condition=unnormalized_pseudo_initial_condition[0],
                unnormalized_initial_condition_mean=unnormalized_initial_condition_means[index],
                unnormalized_initial_condition_variance=unnormalized_initial_condition_variances[index],
                unnormalized_evaluation_times=[jnp.linspace(start_of_times, end_of_times, 100).reshape(1, -1)]*3,
                rng=new_rng,
                n_samples=n_samples)

            list_of_means.append(current_state_mean)
            list_of_variances.append(current_state_var)

        state_means_num_int = []
        state_variances_num_int = []
        for dim_id in range(len(list_of_means[0])):
            state_means_of_this_dim = []
            state_vars_of_this_dim = []
            for traj_id in range(len(list_of_means)):
                state_means_of_this_dim.append(list_of_means[traj_id][dim_id][0])
                state_vars_of_this_dim.append(list_of_variances[traj_id][dim_id][0])
            state_means_num_int.append(state_means_of_this_dim)
            state_variances_num_int.append(state_vars_of_this_dim)

        # create means and variances of smoother predictions
        initial_conditions_for_smoother = [self.ic_of_full_dataset[traj_id] for traj_id in trajectory_ids]

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            initial_conditions=initial_conditions_for_smoother,
            times=[jnp.linspace(start_of_times, end_of_times, 100).reshape([1, -1])]*3
        )

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        predictions = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        state_means_smoother = predictions[0]
        state_variances_smoother = predictions[1]

        return (
            times, observations,
            state_means_num_int, state_variances_num_int,
            state_means_smoother, state_variances_smoother
        )

    def get_obs_nll_via_numerical_integration_by_id_and_ic(self, trajectory_id, unnormalized_initial_condition_mean,
                                                           unnormalized_initial_condition_variance, rng, name,
                                                           n_samples=100, start_of_times=0,
                                                           end_of_times=None):
        unnormalized_pseudo_initial_condition, unnorm_obs_times, unnorm_observations = \
            self.get_information_for_trajectory_of_id(trajectory_id,
                                                      start_of_times=start_of_times,
                                                      end_of_times=end_of_times
                                                      )

        rng, new_rng = jax.random.split(rng)
        unnorm_dynamics_means, unnorm_dynamics_variances = self.get_trajectory_moments_via_numerical_integration(
            unnormalized_pseudo_initial_condition=unnormalized_pseudo_initial_condition[0],
            unnormalized_initial_condition_mean=unnormalized_initial_condition_mean,
            unnormalized_initial_condition_variance=unnormalized_initial_condition_variance,
            unnormalized_evaluation_times=unnorm_obs_times,
            rng=new_rng,
            n_samples=n_samples)

        denormalized_observation_noise_variances = self.get_denormalized_smoother_observation_noise_variance()

        nll_state = self._compute_nll_per_dimension_for_state_observations(
            denormalized_state_means=unnorm_dynamics_means,
            denormalized_state_variances=unnorm_dynamics_variances,
            denormalized_observation_noise_variances=denormalized_observation_noise_variances,
            observations=unnorm_observations,
            ignore_third_dimension=True
        )

        return nll_state[0]

    def integrate_one_trajectory(self, normalized_initial_condition, pseudo_initial_condition,
                                 normalized_times_for_this_dim, rng):
        rng, subkey = jax.random.split(rng)
        sample_from_normal = jax.random.normal(subkey, (normalized_initial_condition.size,))

        @jax.jit
        def compute_derivative(x, _):
            mean, pseudo_covariances = self.dynamics_model_apply(
                self.parameters['dynamics'], jnp.concatenate([x, pseudo_initial_condition]).reshape(1, -1))
            return (mean + jnp.sqrt(pseudo_covariances) * sample_from_normal).reshape(-1)

        trajectory = odeint(
            compute_derivative, normalized_initial_condition.reshape(-1), normalized_times_for_this_dim
        )
        return jnp.array(trajectory)

    def get_trajectory_moments_via_numerical_integration(self,
                                                         unnormalized_pseudo_initial_condition,
                                                         unnormalized_initial_condition_mean,
                                                         unnormalized_initial_condition_variance,
                                                         unnormalized_evaluation_times,
                                                         rng,
                                                         n_samples):
        # normalize time and initial conditions

        normalized_evaluation_times = []
        for dimension_id in range(len(unnormalized_evaluation_times)):
            normalized_times_for_this_dimension = []
            for traj_id in range(len(unnormalized_evaluation_times[dimension_id])):
                normalized_times_for_this_dimension.append(
                    self.time_normalizers[dimension_id].transform(unnormalized_evaluation_times[dimension_id][traj_id].reshape(-1, 1)).reshape(-1)
                )
            normalized_evaluation_times.append(normalized_times_for_this_dimension)

        normalized_pseudo_initial_condition = []
        for dim_id in range(unnormalized_pseudo_initial_condition.size):
            normalized_pseudo_initial_condition.append(self.ic_normalizers[dim_id].transform(
                jnp.array(unnormalized_pseudo_initial_condition[dim_id]).reshape(1, 1)
            ).reshape(-1))
        normalized_pseudo_initial_condition = jnp.squeeze(jnp.array(normalized_pseudo_initial_condition))

        # sample initial conditions

        unnormalized_initial_condition_samples = []
        for i in range(n_samples):
            rng, subkey = jax.random.split(rng)
            sample_from_normal = jax.random.normal(subkey, unnormalized_initial_condition_mean.shape)
            unnormalized_initial_condition_samples.append(
                jnp.sqrt(unnormalized_initial_condition_variance)*sample_from_normal
                + unnormalized_initial_condition_mean
            )
        unnormalized_initial_condition_samples = jnp.squeeze(jnp.array(unnormalized_initial_condition_samples))

        normalized_initial_condition_samples = []
        for dim_id in range(unnormalized_initial_condition_samples.shape[1]):
            normalized_initial_condition_samples.append(
                self.state_normalizers[dim_id].transform(unnormalized_initial_condition_samples[:, dim_id].reshape(-1, 1))
            )
        normalized_initial_condition_samples = jnp.squeeze(jnp.array(normalized_initial_condition_samples))

        # integrate everything

        normalized_trajectory_samples = []
        unnormalized_trajectory_samples = []
        for dim_id in range(len(normalized_evaluation_times)):
            normalized_samples_for_this_dim = []
            unnormalized_samples_for_this_dim = []
            for sample_id in range(n_samples):
                rng, subkey = jax.random.split(rng)
                this_trajectory = self.integrate_one_trajectory(
                    normalized_initial_condition=normalized_initial_condition_samples[:, sample_id],
                    pseudo_initial_condition=normalized_pseudo_initial_condition,
                    normalized_times_for_this_dim=normalized_evaluation_times[dim_id][0],
                    rng=subkey)
                normalized_samples_for_this_dim.append(this_trajectory[:, dim_id])
                unnormalized_samples_for_this_dim.append(self.state_normalizers[dim_id].inverse_transform(
                    this_trajectory[:, dim_id].reshape(-1, 1)
                    ))
            normalized_trajectory_samples.append(normalized_samples_for_this_dim)
            unnormalized_trajectory_samples.append(unnormalized_samples_for_this_dim)

        dynamics_means = []
        dynamics_variances = []
        for dim_id in range(len(unnormalized_trajectory_samples)):
            trajectories_of_this_dim = jnp.array(unnormalized_trajectory_samples[dim_id])
            dynamics_means.append(jnp.mean(trajectories_of_this_dim, axis=0).reshape(1, -1))
            dynamics_variances.append(jnp.var(trajectories_of_this_dim, axis=0).reshape(1, -1))

        return dynamics_means, dynamics_variances


class ShortLearnUSHCNInterpolatorIC(LearnUSHCNInterpolatorIC):
    def __init__(
        self,
        seed: int,
        data_generation: Dict,
        smoother: Dict,
        dynamics: Dict,
        betas: Dict,
        optimizer: Dict,
        priors: Dict,
        logging: Dict,
        numerical_correction: float,
        num_derivative_points_per_trajectory: int,
        split_pretraining: Optional[bool] = True,
        create_equidistant_derivative_times: Optional[bool] = True,
        ignore_third_dimension_input_for_deriv_times: Optional[bool] = True,
        obs_std_prior: Optional[Dict] = None,
        spectral_regularization_weight_smoother: Optional[float] = 0,
        spectral_regularization_weight_dynamics: Optional[float] = 0,
        use_spectral_normalization: Optional[bool] = False,
        n_l2_pretraining_for_dynamics: Optional[int] = 0,
        horizon_split_time: Optional[int] = 50,
        random_seed_test_train_split: Optional[int] = None
    ):
        self.horizon_split_time = horizon_split_time
        self.random_seed_test_train_split = random_seed_test_train_split
        super().__init__(seed=seed,
                         data_generation=data_generation,
                         smoother=smoother,
                         dynamics=dynamics,
                         betas=betas,
                         optimizer=optimizer,
                         priors=priors,
                         logging=logging,
                         numerical_correction=numerical_correction,
                         num_derivative_points_per_trajectory=num_derivative_points_per_trajectory,
                         split_pretraining=split_pretraining,
                         create_equidistant_derivative_times=create_equidistant_derivative_times,
                         obs_std_prior=obs_std_prior,
                         spectral_regularization_weight_smoother=spectral_regularization_weight_smoother,
                         spectral_regularization_weight_dynamics=spectral_regularization_weight_dynamics,
                         use_spectral_normalization=use_spectral_normalization,
                         n_l2_pretraining_for_dynamics=n_l2_pretraining_for_dynamics,
                         ignore_third_dimension_input_for_deriv_times=ignore_third_dimension_input_for_deriv_times
                         )

    def _create_data(self):
        original_data = load_raw_data(self.path_to_data)
        position_extractor = PositionExtractor(path_to_stations_file=self.path_to_stations_file,
                                               path_to_mapping=self.path_to_mapping,
                                               )
        times, values, initial_conditions = extract_time_value_and_position_arrays_from_GRU_data(
            original_data, position_extractor, augmented=self.load_augmented_states)
        self.times, self.observations = format_data_for_dgm(times, values, augmented_states=self.load_augmented_states)

        # only use dimensions that are interesting
        self.times = [self.times[i] for i in self.dimensions_to_consider]
        self.observations = [self.observations[i] for i in self.dimensions_to_consider]

        self.times, self.observations, self.future_times, self.future_obs = \
            self.split_trajectories_at_splitting_time(times=self.times,
                                                      observations=self.observations,
                                                      splitting_time=self.horizon_split_time)

        # kill all trajectories with not enough obs
        self.times, self.observations, self.future_times, self.future_obs, initial_conditions = \
            extract_trajectories_with_more_than_n_train_obs_per_dim_and_3_val_obs(
                n=self.min_obs_per_trajectory,
                train_times=self.times,
                train_observations=self.observations,
                test_times=self.future_times,
                test_observations=self.future_obs,
                initial_conditions=initial_conditions
            )

        # train / testing split
        train_times, train_obs, test_times, test_obs = self.get_train_test_split(self.times,
                                                                                 self.observations,
                                                                                 None,
                                                                                 )

        self.times_of_full_dataset = self.times
        self.obs_of_full_dataset = self.observations
        self.ic_of_full_dataset = initial_conditions

        # only use the first few trajectories, if desired
        if len(train_times[0]) <= self.n_trajectories_to_consider:
            raise ValueError("Too many training trajectories desired")

        # randomly subsample trajectories if necessary
        if self.random_seed_test_train_split:
            rng = jax.random.PRNGKey(self.random_seed_test_train_split)
            self.train_indices = jax.random.choice(key=rng,
                                                   a=len(train_times[0]),
                                                   shape=[self.n_trajectories_to_consider],
                                                   replace=False,
                                                   )
        else:
            self.train_indices = jnp.arange(self.n_trajectories_to_consider)
        self.test_indices = jnp.setdiff1d(jnp.arange(len(train_times[0])),
                                          self.train_indices)

        # extract_by_indices
        train_times = self.extract_trajectories_by_indices(train_times, self.train_indices)
        train_obs = self.extract_trajectories_by_indices(train_obs, self.train_indices)
        test_times = self.extract_trajectories_by_indices(test_times, self.train_indices)
        test_obs = self.extract_trajectories_by_indices(test_obs, self.train_indices)

        initial_conditions = [initial_conditions[a] for a in self.train_indices]

        self.all_times = self.merge_train_test_split(train_times, test_times)
        self.all_observations = self.merge_train_test_split(train_obs, test_obs)
        self.all_initial_conditions = initial_conditions

        self.times = train_times
        self.observations = train_obs
        self.initial_conditions = initial_conditions
        self.num_trajectories = len(self.initial_conditions)
        self.state_dimension = len(self.initial_conditions[0])

        self.test_times = test_times
        self.test_observations = test_obs
        self.test_initial_conditions = initial_conditions
        # use same initial conditions as for training, since we explicitly assume that we only have validation stuff
        # on the trajectories we train on.

        # TODO: maybe do something different for

        self.ground_truth_states = self.observations
        self.ground_truth_derivatives = self.observations  # hack!! No derivatives available for real data

    # @staticmethod
    # def extract_trajectories_by_indices(quantity, indices):
    #     return_quantity = []
    #     for dim_id in range(len(quantity)):
    #         return_quantity.append([quantity[dim_id][a] for a in indices])
    #     return return_quantity

    def get_smoother_generalization_nll_and_plots(self, trajectory_ids, start_of_times, end_of_times, n_dim=3,
                                                  plotting_name_suffix=""):

        initial_conditions_for_smoother = [self.ic_of_full_dataset[traj_id] for traj_id in trajectory_ids]

        unnormalized_test_times = [[] for a in range(n_dim)]
        unnormalized_test_observations = [[] for a in range(n_dim)]
        for trajectory_id in trajectory_ids:
            unnormalized_initial_conditions_for_this_traj, unnormalized_times_for_this_traj,\
                unnormalized_observations_for_this_traj = self.extract_prediction_information_for_trajectory_of_id(
                    trajectory_id=trajectory_id, start_of_times=start_of_times, end_of_times=end_of_times
                )
            for dim_id in range(n_dim):
                unnormalized_test_times[dim_id].append(unnormalized_times_for_this_traj[dim_id][0]+start_of_times)
                unnormalized_test_observations[dim_id].append(unnormalized_observations_for_this_traj[dim_id][0])

        trajectory_lengths, joint_times, joint_repeated_initial_conditions = self.join_trajectories(
            initial_conditions=initial_conditions_for_smoother,
            times=unnormalized_test_times
        )

        joint_normalized_times, joint_repeated_normalized_initial_conditions = self.normalize_and_reshape_times_and_ics(
            joint_times, joint_repeated_initial_conditions)

        predictions = self._get_split_predictions(
            joint_normalized_times,
            joint_repeated_normalized_initial_conditions,
            trajectory_lengths)

        denormalized_state_means = predictions[0]
        denormalized_state_variances = predictions[1]

        # do plots

        print("Plotting")
        figure_smoother_states, figure_smoother_derivatives, figure_dynamics_derivatives = \
            self.plotter.plot_smoother_and_dynamics_predictions(
                prediction_times=unnormalized_test_times,
                state_means=denormalized_state_means,
                state_variances=denormalized_state_variances,
                derivatives_means=denormalized_state_means,
                derivative_variances=denormalized_state_variances,
                dynamics_means=denormalized_state_means,
                dynamics_variances=denormalized_state_variances,
                train_times=unnormalized_test_times,
                train_observations=unnormalized_test_observations,
                all_initial_conditions=None,
                denormalized_obs_noise_variances=self.get_denormalized_smoother_observation_noise_variance()
            )
        figure_smoother_states.tight_layout()

        state_filename = 'smoother_states_' + plotting_name_suffix
        wandb.log({state_filename: wandb.Image(figure_smoother_states)})

        # calculate nlls
        denormalized_observation_noise_variances = self.get_denormalized_smoother_observation_noise_variance()

        nll_state = self._compute_nll_per_dimension_for_state_observations(
            denormalized_state_means=denormalized_state_means,
            denormalized_state_variances=denormalized_state_variances,
            denormalized_observation_noise_variances=denormalized_observation_noise_variances,
            observations=unnormalized_test_observations,
            ignore_third_dimension=True
        )

        return nll_state[0]

    def extract_prediction_information_for_trajectory_of_id(self, trajectory_id, start_of_times, end_of_times):
        initial_conditions = [self.ic_of_full_dataset[trajectory_id]]
        all_future_times = self.extract_value_by_id(trajectory_id, self.future_times)
        all_future_obs = self.extract_value_by_id(trajectory_id, self.future_obs)
        times = []
        observations = []
        for dim_id in range(len(all_future_times)):
            indices_for_this_dim = [start_of_times <= a <= end_of_times for a in all_future_times[dim_id][0]]
            indices_for_this_dim = jnp.array(indices_for_this_dim)
            indices_for_this_dim = jnp.where(indices_for_this_dim)
            times.append([all_future_times[dim_id][0][indices_for_this_dim]-start_of_times])  # 0, because of extraction by id a few lines before
            observations.append([all_future_obs[dim_id][0][indices_for_this_dim]])
            # times.append(np.asarray(all_future_times[dim_id][0][indices_for_this_dim]-start_of_times))  # 0, because of extraction by id a few lines before
            # observations.append(np.asarray(all_future_obs[dim_id][0][indices_for_this_dim]))
        return initial_conditions, times, observations

    @staticmethod
    def split_trajectories_at_splitting_time(times, observations, splitting_time=150):
        """
        for each trajectory, extracts all observations before time 150 for training and gets all future values for
        validation
        """
        def get_train_test_split_for_one_value(values):
            train_values = []
            test_values = []
            for dim_id in range(len(times)):
                train_values_for_this_dim = []
                test_values_for_this_dim = []
                for traj_id in range(len(times[dim_id])):
                    train_values_for_this_traj = []
                    test_values_for_this_traj = []
                    for idx, curr_time in enumerate(times[dim_id][traj_id]):
                        if curr_time < splitting_time:
                            train_values_for_this_traj.append(values[dim_id][traj_id][idx])
                        else:
                            test_values_for_this_traj.append(values[dim_id][traj_id][idx])
                    train_values_for_this_dim.append(jnp.asarray(train_values_for_this_traj))
                    test_values_for_this_dim.append(jnp.asarray(test_values_for_this_traj))
                train_values.append(train_values_for_this_dim)
                test_values.append(test_values_for_this_dim)
            return train_values, test_values

        train_times, test_times = get_train_test_split_for_one_value(times)
        train_obs, test_obs = get_train_test_split_for_one_value(observations)

        return train_times, train_obs, test_times, test_obs

