from typing import Callable, Any, Optional, Dict, List

import jax.numpy as jnp
import jax

pytree = Any
apply_fun = Callable[[pytree, jnp.array], jnp.array]


def get_objective_builder(
        apply_smoother: Callable,
        apply_dynamics: apply_fun,
        get_dynamics_regularization,
        get_smoother_regularization,
        number_of_obs_per_dimension: jnp.array,
        is_dynamics_time_dependent: Optional[bool] = False,
        is_dynamics_ic_dependent: Optional[bool] = False,
        return_split_objectives: Optional[bool] = False,
        obs_std_prior: Optional[Dict] = None,
):
    total_number_of_obs_over_all_dim = jnp.sum(number_of_obs_per_dimension)

    @jax.jit
    def build_complete_objective(
            parameters: pytree,
            times: jnp.array,  # List over dimensions of shape n_obs
            times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
            initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
            initial_conditions_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs x n_dim
            observations: jnp.array,  # List over dimensions of shape n_obs x n_dim
            betas: jnp.array,
            weights
    ) -> float:
        # Compute regularization
        reg_objective = get_dynamics_regularization(parameters['dynamics'], weights)
        reg_objective += get_smoother_regularization(parameters['smoother'], weights)

        obs_prior_value = 0
        if obs_std_prior:
            raw_sigma = parameters['smoother']['smoother']['noise_std']
            raw_sigma = jnp.array(raw_sigma)
            means = jnp.ones_like(raw_sigma) * obs_std_prior['mean']
            stds = jnp.eye(means.size) * obs_std_prior['std']**2
            obs_prior_value = -jax.scipy.stats.multivariate_normal.logpdf(raw_sigma, means, stds)

        def compute_for_one_dimension(dimension):
            # Compute smoother terms
            (
                posterior_means,
                posterior_smoother_derivative_means,
                posterior_smoother_derivatives_covariances,
                data_fit
            ) = apply_smoother(
                times,
                [times_for_derivatives[dimension]] * len(times_for_derivatives),
                initial_conditions,
                [initial_conditions_for_derivatives[dimension]]*len(initial_conditions_for_derivatives),
                observations,
                parameters["smoother"]
            )

            posterior_means = jnp.stack(posterior_means).T
            posterior_smoother_derivative_means = posterior_smoother_derivative_means[dimension]
            # jnp.stack(posterior_smoother_derivative_means).T
            posterior_smoother_derivatives_covariances = posterior_smoother_derivatives_covariances[dimension]
            # jnp.stack(posterior_smoother_derivatives_covariances).T

            states_for_dynamics_model = posterior_means
            if is_dynamics_time_dependent:
                states_for_dynamics_model = jnp.concatenate([posterior_means, times_for_derivatives.reshape(-1, 1)], axis=1)
            elif is_dynamics_ic_dependent:
                states_for_dynamics_model = jnp.concatenate(
                    [posterior_means, initial_conditions_for_derivatives[dimension]], axis=1)

            # Compute dynamics terms
            posterior_dynamics_derivative_means, posterior_dynamics_derivatives_covariances = apply_dynamics(
                parameters["dynamics"], states_for_dynamics_model)

            posterior_dynamics_derivative_means = posterior_dynamics_derivative_means[:, dimension]
            posterior_dynamics_derivatives_covariances = posterior_dynamics_derivatives_covariances[:, dimension]

            # Compute data fit term (marginal log likelihood in the case of full GP)
            # careful, this is not proper log likelihood. Constant and factor -1/2 have been omitted.
            mll_terms = data_fit[dimension]  # jnp.sum(jnp.stack(data_fit))

            # Compute fairness factor
            num_states = initial_conditions[dimension].size
            num_derivatives = initial_conditions_for_derivatives[dimension].size
            # this calculates wrongly, as initial conditions are multi-dimensional. However, this is cancelled when
            # calculating fairness factor.
            fairness_factor = num_states / num_derivatives

            # Compute Wasserstein distance
            wass_regularization = betas[dimension] * jnp.sum(
                (posterior_smoother_derivative_means - posterior_dynamics_derivative_means) ** 2)
            wass_regularization += betas[dimension] * jnp.sum(
                posterior_smoother_derivatives_covariances + posterior_dynamics_derivatives_covariances)
            wass_regularization -= 2 * betas[dimension] * jnp.sum(
                jnp.sqrt(posterior_smoother_derivatives_covariances * posterior_dynamics_derivatives_covariances))

            wass_regularization = 2 * wass_regularization  # the factor 2 is to compensate for the omitted 1/2 factor in the mll of the GP

            return mll_terms + fairness_factor * wass_regularization

        objective = 0
        for dim_id in range(len(observations)):
            objective += compute_for_one_dimension(dim_id) / number_of_obs_per_dimension[dim_id]

        # this could be a source of bugs, depending on the system analyzed. we want to divide by n_dim
        return \
            objective / len(observations) + \
            obs_prior_value + \
            reg_objective

    if not return_split_objectives:
        return build_complete_objective
    else:
        @jax.jit
        def build_smoother_objective(
                smoother_parameters: pytree,
                times: jnp.array,  # List over dimensions of shape n_obs
                times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
                initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs x n_dim
                observations: jnp.array,  # List over dimensions of shape n_obs x n_dim
                weights
        ) -> float:
            obs_prior_value = 0
            if obs_std_prior:
                raw_sigma = smoother_parameters['smoother']['noise_std']
                raw_sigma = jnp.array(raw_sigma)
                means = jnp.ones_like(raw_sigma) * obs_std_prior['mean']
                stds = jnp.eye(means.size) * obs_std_prior['std'] ** 2
                obs_prior_value = -jax.scipy.stats.multivariate_normal.logpdf(raw_sigma, means, stds)

            # Compute regularization
            reg_objective = get_smoother_regularization(smoother_parameters, weights)

            # Compute smoother terms
            (
                posterior_means,
                posterior_smoother_derivative_means,
                posterior_smoother_derivatives_covariances,
                data_fit
            ) = apply_smoother(
                times, times_for_derivatives, initial_conditions, initial_conditions_for_derivatives, observations,
                smoother_parameters
            )

            # Compute data fit term (marginal log likelihood in the case of full GP)

            normed_mll_terms = jnp.mean(jnp.stack(data_fit) / number_of_obs_per_dimension)

            return normed_mll_terms + obs_prior_value + reg_objective

        @jax.jit
        def build_dynamics_objective(
                dynamics_parameters: pytree,
                smoother_parameters: pytree,
                times: jnp.array,  # List over dimensions of shape n_obs
                times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
                initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives: jnp.array,  # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                observations: jnp.array,  # List over dimensions of shape n_obs x n_dim
                weights
        ) -> float:
            # Compute regularization
            reg_objective = get_dynamics_regularization(dynamics_parameters, weights)
            reg_objective += get_smoother_regularization(smoother_parameters, weights)

            def compute_for_one_dimension(dimension):
                # Compute smoother terms
                (
                    posterior_means,
                    posterior_smoother_derivative_means,
                    posterior_smoother_derivatives_covariances,
                    data_fit
                ) = apply_smoother(
                    times,
                    [times_for_derivatives[dimension]] * len(times_for_derivatives),
                    initial_conditions,
                    [initial_conditions_for_derivatives[dimension]] * len(initial_conditions_for_derivatives),
                    observations,
                    smoother_parameters
                )

                posterior_means = jnp.stack(posterior_means).T
                posterior_smoother_derivative_means = posterior_smoother_derivative_means[dimension]
                # jnp.stack(posterior_smoother_derivative_means).T
                posterior_smoother_derivatives_covariances = posterior_smoother_derivatives_covariances[dimension]
                # jnp.stack(posterior_smoother_derivatives_covariances).T

                states_for_dynamics_model = posterior_means
                if is_dynamics_time_dependent:
                    states_for_dynamics_model = jnp.concatenate(
                        [posterior_means, times_for_derivatives.reshape(-1, 1)], axis=1)
                elif is_dynamics_ic_dependent:
                    states_for_dynamics_model = jnp.concatenate(
                        [posterior_means, initial_conditions_for_derivatives[dimension]], axis=1)

                # Compute dynamics terms
                posterior_dynamics_derivative_means, posterior_dynamics_derivatives_covariances = apply_dynamics(
                    dynamics_parameters, states_for_dynamics_model)

                posterior_dynamics_derivative_means = posterior_dynamics_derivative_means[:, dimension]
                posterior_dynamics_derivatives_covariances = posterior_dynamics_derivatives_covariances[:, dimension]

                # Compute data fit term (marginal log likelihood in the case of full GP)

                mll_terms = data_fit[dimension]  # jnp.sum(jnp.stack(data_fit))

                # Compute fairness factor
                num_states = initial_conditions[dimension].size
                num_derivatives = initial_conditions_for_derivatives[dimension].size
                fairness_factor = num_states / num_derivatives

                # Compute Wasserstein distance
                wass_regularization = jnp.sum(
                    jnp.sum(
                        (posterior_smoother_derivative_means - posterior_dynamics_derivative_means)**2,
                        axis=0)
                    )
                wass_regularization += jnp.sum(
                    jnp.sum(
                        posterior_smoother_derivatives_covariances + posterior_dynamics_derivatives_covariances,
                        axis=0))
                wass_regularization -= 2 * jnp.sum(jnp.sum(
                    jnp.sqrt(posterior_smoother_derivatives_covariances * posterior_dynamics_derivatives_covariances),
                    axis=0))

                wass_regularization = 2 * wass_regularization  # the factor 2 is to compensate for the omitted 1/2 factor in the mll of the GP
                return mll_terms + fairness_factor * wass_regularization

            objective = 0
            for dim_id in range(len(observations)):
                objective += compute_for_one_dimension(dim_id) / number_of_obs_per_dimension[dim_id]

            return objective / len(observations) + reg_objective

        @jax.jit
        def build_std_l2_objective(
                dynamics_std_parameters: pytree,
                dynamics_mean_parameters: pytree,
                smoother_parameters: pytree,
                times: jnp.array,  # List over dimensions of shape n_obs
                times_for_derivatives: jnp.array,  # List over dimensions of shape n_deriv_obs
                initial_conditions: jnp.array,  # List over dimensions of shape n_obs x n_dim
                initial_conditions_for_derivatives: jnp.array,
                # List over dimensinos of shape n_deriv_obs x n_dim  #  adapt execution
                observations: jnp.array,  # List over dimensions of shape n_obs x 1
                weights
        ) -> float:
            dynamics_parameters = {
                'mean': dynamics_mean_parameters,
                'std': dynamics_std_parameters
            }

            # Compute regularization
            reg_objective = get_dynamics_regularization(dynamics_parameters, weights)

            def compute_for_one_dimension(dimension):
                # Compute smoother terms
                (
                    posterior_means,
                    posterior_smoother_derivative_means,
                    posterior_smoother_derivatives_covariances,
                    data_fit
                ) = apply_smoother(
                    times,
                    [times_for_derivatives[dimension]] * len(times_for_derivatives),
                    initial_conditions,
                    [initial_conditions_for_derivatives[dimension]] * len(initial_conditions_for_derivatives),
                    observations,
                    smoother_parameters
                )

                posterior_means = jnp.stack(posterior_means).T
                posterior_smoother_derivatives_covariances = posterior_smoother_derivatives_covariances[dimension]

                states_for_dynamics_model = posterior_means
                if is_dynamics_time_dependent:
                    states_for_dynamics_model = jnp.concatenate(
                        [posterior_means, times_for_derivatives.reshape(-1, 1)], axis=1)
                elif is_dynamics_ic_dependent:
                    states_for_dynamics_model = jnp.concatenate(
                        [posterior_means, initial_conditions_for_derivatives.reshape(-1, 1)], axis=1)
                # Compute dynamics terms
                posterior_dynamics_derivative_means, posterior_dynamics_derivatives_covariances = apply_dynamics(
                    dynamics_parameters, states_for_dynamics_model)

                posterior_dynamics_derivatives_covariances = posterior_dynamics_derivatives_covariances[:, dimension]

                # Compute Wasserstein distance
                std_l2_loss = jnp.sum(
                    jnp.sum(
                        posterior_smoother_derivatives_covariances + posterior_dynamics_derivatives_covariances,
                        axis=0))
                std_l2_loss -= 2 * jnp.sum(jnp.sum(
                    jnp.sqrt(posterior_smoother_derivatives_covariances * posterior_dynamics_derivatives_covariances),
                    axis=0))

                return 2*std_l2_loss

            objective = 0
            for dim_id in range(len(observations)):
                objective += compute_for_one_dimension(dim_id) / number_of_obs_per_dimension[dim_id]

            return objective / len(observations) + reg_objective

        return build_complete_objective, build_smoother_objective, build_dynamics_objective, build_std_l2_objective
