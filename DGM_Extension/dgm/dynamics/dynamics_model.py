from typing import Optional, Any, Callable, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import sklearn
from jax import jit
from scipy.integrate import odeint
from scipy.stats.distributions import chi2

from dgm.utils.representatives import DynamicsModel
from dgm.dynamics.dynamics_mean_and_std import parametric_lotka_volterra_mean, \
    parametric_perfectly_adapted_mean, parametric_negative_feedback_oscilator_mean, parametric_lorenz_mean, \
    parametric_cstr_mean, small_node_mean, parametric_lotka_volterra_std, heteroscedastic_std, small_dynamics, \
    parametric_double_pendulum_mean, parametric_quadrocopter_mean, linear_mean, linear_prod_mean, medium_dynamics, \
    big_dynamics, joint_nn_dynamics, joint_nn_dynamics_swish, simple_nn_dynamics
from dgm.utils.helper_functions import make_positive

in_shape = Optional[Tuple[int, int]]
out_shape = Optional[Tuple[int, int]]

Scaler = sklearn.preprocessing._data.StandardScaler
pytree = Any
rng = np.ndarray
init_fun = Callable[[rng, in_shape], Tuple[out_shape, pytree]]
apply_fun = Callable[[pytree, jnp.array], jnp.array]


def get_dynamics(dynamics_model: DynamicsModel, state_normalizers: List[Scaler], time_normalizers: List[Scaler],
                 state_dimension: int = 2, **dynamics_kwargs):
    if dynamics_model == DynamicsModel.PARAMETRIC_LOTKA_VOLTERRA:
        return parametric_lotka_volterra(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.LOTKA_VOLTERRA_WITH_HETEROSCEDASTIC_NOISE:
        return lotka_volterra_with_heteroscedastic_noise(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.PERFECTLY_ADAPTED_HETEROSCEDASTIC_NOISE:
        return perfectly_adapted_with_heteroscedatic_noise(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.NEGATIVE_FEEDBACK_OSCILATOR_HETEROSCEDASTIC_NOISE:
        return negative_feedback_oscilator_with_heteroscedatic_noise(state_normalizers, time_normalizers,
                                                                     **dynamics_kwargs)
    elif dynamics_model == DynamicsModel.LORENZ_HETEROSCEDASTIC_NOISE:
        return lorenz_with_heteroscedastic_noise(state_normalizers, time_normalizers, )
    elif dynamics_model == DynamicsModel.CSTR_HETEROSCEDASTIC_NOISE:
        return cstr_with_heteroscedastic_noise(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.TWO_DIMENSIONAL_NODE:
        return two_dimensional_node(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.JOINT_SMALL_DYNAMICS:
        return joint_small_dynamics(state_normalizers, time_normalizers, state_dimension)
    elif dynamics_model == DynamicsModel.JOINT_MEDIUM_DYNAMICS:
        return joint_medium_dynamics(state_normalizers, time_normalizers, state_dimension)
    elif dynamics_model == DynamicsModel.JOINT_BIG_DYNAMICS:
        return joint_big_dynamics(state_normalizers, time_normalizers, state_dimension)
    elif dynamics_model == DynamicsModel.DOUBLE_PENDULUM_HETEROSCEDASTIC_NOISE:
        return double_pendulum_with_heteroscedastic_noise(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.QUADROCOPTER_HETEROSCEDASTIC_NOISE:
        return quadrocopter_with_heteroscedastic_noise(state_normalizers, time_normalizers)
    elif dynamics_model == DynamicsModel.LINEAR_HETEROSCEDASTIC_NOISE:
        return linear_with_heteroscedastic_noise(state_normalizers, time_normalizers, state_dimension)
    elif dynamics_model == DynamicsModel.LINEAR_PROD_HETEROSCEDASTIC_NOISE:
        return linear_prod_with_heteroscedastic_noise(state_normalizers, time_normalizers, state_dimension, )
    elif dynamics_model == DynamicsModel.JOINT_NN:
        return joint_nn(state_normalizers, time_normalizers, state_dimension, **dynamics_kwargs)
    elif dynamics_model == DynamicsModel.JOINT_NN_SWISH:
        return joint_nn_swish(state_normalizers, time_normalizers, state_dimension, **dynamics_kwargs)
    elif dynamics_model == DynamicsModel.JOINT_NN_TIME_DEPENDENT:
        return joint_time_dependent_nn(state_normalizers, time_normalizers, state_dimension, **dynamics_kwargs)
    elif dynamics_model == DynamicsModel.JOINT_NN_IC_DEPENDENT:
        return joint_ic_dependent_nn(state_normalizers, time_normalizers, state_dimension, **dynamics_kwargs)
    elif dynamics_model == DynamicsModel.SPLIT_NN:
        return split_node(state_normalizers, time_normalizers, state_dimension, **dynamics_kwargs)
    else:
        raise NotImplementedError("Chosen dynamics model has not been implemented yet.")


def joint_ic_dependent_nn(state_normalizers, time_normalizers, state_dimension, **kwargs):
    dynamics_dict = kwargs['dynamics_kwargs']
    init, apply, get_regularization = joint_nn_dynamics(state_dimension, dynamics_dict['hidden_layers'])
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False,
        time_dependent=False,
        ic_dependent=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_time_dependent_nn(state_normalizers, time_normalizers, state_dimension, **kwargs):
    dynamics_dict = kwargs['dynamics_kwargs']
    init, apply, get_regularization = joint_nn_dynamics(state_dimension, dynamics_dict['hidden_layers'])
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False,
        time_dependent=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_nn(state_normalizers, time_normalizers, state_dimension, **kwargs):
    dynamics_dict = kwargs['dynamics_kwargs']
    state_dimension = state_dimension
    init, apply, get_regularization = joint_nn_dynamics(state_dimension, dynamics_dict['hidden_layers'])
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_nn_swish(state_normalizers, time_normalizers, state_dimension, **kwargs):
    dynamics_dict = kwargs['dynamics_kwargs']
    state_dimension = state_dimension
    init, apply, get_regularization = joint_nn_dynamics_swish(state_dimension, dynamics_dict['hidden_layers'])
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_big_dynamics(state_normalizers, time_normalizers, state_dimension):
    state_dimension = state_dimension
    init, apply, get_regularization = big_dynamics(state_dimension)
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_medium_dynamics(state_normalizers, time_normalizers, state_dimension):
    state_dimension = state_dimension
    init, apply, get_regularization = medium_dynamics(state_dimension)
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def joint_small_dynamics(state_normalizers, time_normalizers, state_dimension):
    state_dimension = state_dimension
    init, apply, get_regularization = small_dynamics(state_dimension)
    joint = True

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    ) = prepare_functions_for_learning_and_inference(
        (init,),
        (apply,),
        (get_regularization,),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_parameter_regularization
    )


def split_node(state_normalizers, time_normalizers, state_dimension, **kwargs):
    hidden_layers_mean = kwargs['dynamics_kwargs']['hidden_layers_mean']
    hidden_layers_std = kwargs['dynamics_kwargs']['hidden_layers_std']

    mean_init, mean_apply, mean_regularize = simple_nn_dynamics(state_dimension, hidden_layers_mean)
    std_init, std_apply, std_regularize = simple_nn_dynamics(state_dimension, hidden_layers_std)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def two_dimensional_node(state_normalizers, time_normalizers):
    state_dimension = 2
    mean_init, mean_apply, mean_regularize = small_node_mean(state_dimension)
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=False
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def linear_prod_with_heteroscedastic_noise(state_normalizers, time_normalizers, state_dimension):
    mean_init, mean_apply, mean_regularize = linear_prod_mean(state_dimension)
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def linear_with_heteroscedastic_noise(state_normalizers, time_normalizers, state_dimension):
    mean_init, mean_apply, mean_regularize = linear_mean(state_dimension)
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def lotka_volterra_with_heteroscedastic_noise(state_normalizers, time_normalizers):
    state_dimension = 2
    mean_init, mean_apply, mean_regularize = parametric_lotka_volterra_mean()
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def quadrocopter_with_heteroscedastic_noise(state_normalizers, time_normalizers):
    state_dimension = 12
    mean_init, mean_apply, mean_regularize = parametric_quadrocopter_mean()
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def double_pendulum_with_heteroscedastic_noise(state_normalizers, time_normalizers):
    state_dimension = 4
    mean_init, mean_apply, mean_regularize = parametric_double_pendulum_mean()
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def parametric_lotka_volterra(state_normalizers, time_normalizers):
    state_dimension = 2
    mean_init, mean_apply, mean_regularize = parametric_lotka_volterra_mean()
    std_init, std_apply, std_regularize = parametric_lotka_volterra_std()
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def perfectly_adapted_with_heteroscedatic_noise(state_normalizers, time_normalizers, s=1):
    state_dimension = 2
    mean_init, mean_apply, mean_regularize = parametric_perfectly_adapted_mean(s=s)
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def negative_feedback_oscilator_with_heteroscedatic_noise(state_normalizers, time_normalizers, s=1):
    state_dimension = 3
    mean_init, mean_apply, mean_regularize = parametric_negative_feedback_oscilator_mean(s=s)
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def lorenz_with_heteroscedastic_noise(state_normalizers, time_normalizers):
    state_dimension = 3
    mean_init, mean_apply, mean_regularize = parametric_lorenz_mean()
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


def cstr_with_heteroscedastic_noise(state_normalizers, time_normalizers):
    state_dimension = 2
    mean_init, mean_apply, mean_regularize = parametric_cstr_mean()
    std_init, std_apply, std_regularize = heteroscedastic_std(state_dimension)
    joint = False

    (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    ) = prepare_functions_for_learning_and_inference(
        (mean_init, std_init),
        (mean_apply, std_apply),
        (mean_regularize, std_regularize),
        joint,
        state_dimension,
        state_normalizers,
        time_normalizers,
        parametric_mean=True
    )

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        sample_trajectories,
        get_regularization
    )


"""Helper functions"""


def create_initialize_parameters(joint: bool,
                                 input_dimension: int,
                                 *init_funs: init_fun) -> Callable[[rng], pytree]:
    def initialize_parameters(rng: rng) -> pytree:
        parameters = dict()
        if joint:
            rng, key = jax.random.split(rng)
            _, parameters['mean_and_std'] = init_funs[0](key, (-1, input_dimension))
        else:
            rng, *keys = jax.random.split(rng, 3)
            _, parameters['mean'] = init_funs[0](keys[0], (-1, input_dimension))
            _, parameters['std'] = init_funs[1](keys[1], (-1, input_dimension))
        return parameters

    return initialize_parameters


def create_get_regularization_functions(joint: bool,
                                        *get_regularization_funs):  # TODO: add typing.
    def get_regularization(parameters, weights):
        if joint:
            regularization_term = get_regularization_funs[0](parameters['mean_and_std'], weights)
        else:
            regularization_term = get_regularization_funs[0](parameters['mean'], weights)
            regularization_term += get_regularization_funs[1](parameters['std'], weights)
        return regularization_term

    return get_regularization


def create_get_sample_trajectories(get_dynamics_moments, time_dependent: Optional[bool]=False):

    def sample_trajectories(rng: np.ndarray, parameters: pytree, initial_condition: jnp.array,
                            times: jnp.array, num_samples: int, quantile: float):

        state_dimension = initial_condition.shape[0]
        trajectories = jnp.zeros((num_samples, len(times), state_dimension))
        for i in range(num_samples):
            print(i)
            rng, subkey = jax.random.split(rng)
            sample_from_normal = jax.random.normal(subkey, (state_dimension,))

            if time_dependent:
                @jax.jit
                def compute_derivative(x, t):
                    t = jnp.array(t).reshape(1, -1)
                    x = x.reshape(1, -1)
                    augmented_dynamics_input = jnp.concatenate([t, x], axis=1)
                    mean, pseudo_covariances = get_dynamics_moments(parameters, augmented_dynamics_input)
                    return (mean + make_positive(pseudo_covariances) * sample_from_normal).reshape(-1)
            else:
                @jax.jit
                def compute_derivative(x, _):
                    mean, pseudo_covariances = get_dynamics_moments(parameters, x.reshape(1, -1))
                    return (mean + make_positive(pseudo_covariances) * sample_from_normal).reshape(-1)

            trajectory = odeint(
                compute_derivative, initial_condition.reshape(-1), times
            )
            trajectory = jnp.array(trajectory, dtype=jnp.float32)
            trajectories = jax.ops.index_update(trajectories, jax.ops.index[i, :, :], trajectory)
        trajectories_lower_quantile = jnp.quantile(trajectories, q=1 - quantile, axis=0)
        trajectories_upper_quantile = jnp.quantile(trajectories, q=quantile, axis=0)
        trajectory_median = jnp.median(trajectories, axis=0)
        return trajectory_median, trajectories_lower_quantile, trajectories_upper_quantile

    return sample_trajectories


def create_get_dynamics_moments(joint: bool,
                                *apply_funs: apply_fun) -> \
        Callable[[jnp.array, pytree], Tuple[jnp.array, jnp.array]]:
    @jax.jit
    def get_dynamics_moments(parameters: pytree, xs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        if joint:
            out = apply_funs[0](parameters['mean_and_std'], xs)
            state_dim = int(out.shape[1]/2)
            mean, stds = out[:, :state_dim], out[:, state_dim:]
        else:
            mean = apply_funs[0](parameters['mean'], xs)
            stds = apply_funs[1](parameters['std'], xs)
        covariances = make_positive(stds) ** 2
        return mean, covariances

    return get_dynamics_moments


def prepare_normalized_dynamics(state_normalizers: Scaler, time_normalizers: Scaler,
                                get_dynamics_moments: Callable[[jnp.array, pytree], Tuple[jnp.array, jnp.array]]) -> \
        Callable[[jnp.array, pytree], Tuple[jnp.array, jnp.array]]:
    state_stds = jnp.array([state_normalizers[dim_id].scale_ for dim_id in len(state_normalizers)])
    state_means = jnp.array([state_normalizers[dim_id].mean_ for dim_id in len(state_normalizers)])
    time_stds = jnp.array([time_normalizers[dim_id].scale_ for dim_id in len(state_normalizers)])

    @jit
    def normalized_dynamics_apply(parameters: pytree, xs: jnp.array) -> Tuple[jnp.array, jnp.array]:
        xs = jnp.multiply(state_stds, xs) + state_means
        mean, covariance = get_dynamics_moments(parameters, xs)
        return jnp.multiply(jnp.divide(time_stds, state_stds), mean), \
               jnp.multiply(jnp.divide(time_stds, state_stds), covariance)

    return normalized_dynamics_apply


def volume_of_2d_ellipse(covariance_x, covariance_y, p=0.05):
    # Returns volume of the ellipse described by diagonal matrix with
    # covariance_x and covariance_y on the diagonal and confidence parameter p
    return jnp.pi * chi2.isf(p, 2) * jnp.sqrt(covariance_x * covariance_y)


def create_get_derivatives_for_plotting(state_normalizers: Scaler,
                                        time_normalizers: Scaler,
                                        get_dynamics_moments: Callable[
                                            [jnp.array, pytree], Tuple[jnp.array, jnp.array]]) \
        -> Callable[
            [pytree, jnp.array, jnp.array], Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]]:
    derivative_scales = jnp.array([
        state_normalizers[dim_id].scale_ / time_normalizers[dim_id].scale_ for dim_id in range(len(state_normalizers))])

    def get_derivatives_for_plotting(parameters: pytree, x: jnp.array, y: jnp.array) -> \
            Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        stacked_states = jnp.stack([x, y], axis=2)
        stacked_states_reshaped = stacked_states.reshape(-1, 2)
        normalized_states = [
            state_normalizers[dim_id].transform(stacked_states_reshaped[:, dim_id].reshape(-1, 1)) for dim_id in range(2)
        ]
        x_dot_true, x_dot_covariance = get_dynamics_moments(parameters, normalized_states)
        x_dot_true,= jnp.multiply(derivative_scales, x_dot_true)
        x_dot_covariance = jnp.multiply(derivative_scales, jnp.multiply(derivative_scales, x_dot_covariance))
        x_dot_true = x_dot_true.reshape(stacked_states.shape)
        x_dot_covariance = x_dot_covariance.reshape(stacked_states.shape)
        u_mean, v_mean = x_dot_true[:, :, 0], x_dot_true[:, :, 1]
        norm_mean = jnp.sqrt(u_mean ** 2 + v_mean ** 2)
        u_covariance, v_covariance = x_dot_covariance[:, :, 0], x_dot_covariance[:, :, 1]
        volume_covariance = volume_of_2d_ellipse(u_covariance, v_covariance)
        max_covariance_eigenvalue = jnp.sqrt(jnp.max(jnp.stack([u_covariance, v_covariance], axis=-1), axis=-1))
        return u_mean, v_mean, norm_mean, volume_covariance, max_covariance_eigenvalue

    return get_derivatives_for_plotting


def prepare_functions_for_learning_and_inference(init: Tuple[init_fun, ...],
                                                 apply: Tuple[init_fun, ...],
                                                 get_regularization_funs: Tuple[init_fun, ...],
                                                 joint: bool,
                                                 state_dimension: int,
                                                 state_normalizers: List[Scaler],
                                                 time_normalizers: List[Scaler],
                                                 parametric_mean: bool = True,
                                                 time_dependent: bool = False,
                                                 ic_dependent: bool = False
                                                 ) -> Tuple[Callable, ...]:
    if parametric_mean and time_dependent:
        raise NotImplementedError("current codebase does not support parametric means and time_dependent dynamics models")

    if time_dependent:
        initialize_parameters = create_initialize_parameters(joint, state_dimension+1, *init)
    elif ic_dependent:
        initialize_parameters = create_initialize_parameters(joint, 2*state_dimension, *init)
    else:
        initialize_parameters = create_initialize_parameters(joint, state_dimension, *init)

    get_regularization = create_get_regularization_functions(joint, *get_regularization_funs)

    if parametric_mean:
        get_dynamics_moments = create_get_dynamics_moments(joint, *apply)
        get_dynamics_moments = prepare_normalized_dynamics(state_normalizers, time_normalizers, get_dynamics_moments)
    else:
        get_dynamics_moments = create_get_dynamics_moments(joint, *apply)

    if state_dimension == 2 and not time_dependent:
        get_derivatives_for_plotting = create_get_derivatives_for_plotting(
            state_normalizers, time_normalizers, get_dynamics_moments)
    else:
        def get_derivatives_for_plotting(*args, **kwargs):
            raise NotImplementedError('Dimension of the system is not 2. We only plot 2 dimensional vector fields.')

    get_sample_trajectories = create_get_sample_trajectories(get_dynamics_moments)

    return (
        initialize_parameters,
        get_dynamics_moments,
        get_derivatives_for_plotting,
        get_sample_trajectories,
        get_regularization
    )
