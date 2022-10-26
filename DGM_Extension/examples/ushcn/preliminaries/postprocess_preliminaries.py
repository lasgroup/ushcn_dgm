import argparse
import time
from typing import Callable

import os
os.environ['JAX_ENABLE_X64'] = 'True'

from dgm.plotter.plotter import Plotter

from wandb_scripts.runDB import RunDB
import pickle
import jax
import jax.numpy as jnp
import wandb
from jax.experimental.stax import Dense, Identity, Sigmoid

from dgm.main.learn_system import LearnUSHCNInterpolatorIC
from dgm.schedules.betas import BetasType
from dgm.schedules.learning_rate import LearningRateType
from dgm.schedules.weight_decay import WeightDecayType
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType
from dgm.utils.representatives import KernelType, Optimizer, DynamicsModel, SimulatorType, InitialConditionExtractorType
from matplotlib import pyplot as plt

Schedule = Callable[[int], float]

import numpy as np
np.random.seed(14859)

PATH_TO_DATA = "/home/philippe/ultimate_thesis_plots/preliminaries/data"
PATH_TO_PLOTS = "/home/philippe/ultimate_thesis_plots/preliminaries/plots"
PATH_TO_FINAL_PLOTS = "/home/philippe/ultimate_thesis_plots/preliminaries"


def main():
    # for seed in np.arange(1, 11)*5:
    #     get_data_for_one_seed(seed, 'final')
    #     get_data_for_one_seed(seed, 'smoother')
    # for seed in np.arange(1, 11)*5:
    #     plot_one_seed(seed, 'final')
    #     plot_one_seed(seed, 'smoother')
    # evaluate_all_nlls('final')
    # evaluate_all_nlls('smoother')
    do_one_paper_plot()

def evaluate_all_nlls(prefix):
    path_to_data = os.path.join(PATH_TO_DATA, prefix)
    all_nll_ic = []
    all_nll_ti = []
    for splitting_seed in np.arange(1, 11)*5:
        all_nll_ic.append(
            np.load(os.path.join(path_to_data, f"ic_nll_{splitting_seed}.npy"), allow_pickle=True))
        all_nll_ti.append(
            np.load(os.path.join(path_to_data, f"ti_nll_{splitting_seed}.npy"), allow_pickle=True))
    print(f"{prefix} ic nll: {np.mean(all_nll_ic)} +- {np.std(all_nll_ic)}")
    print(f"{prefix} ti nll: {np.mean(all_nll_ti)} +- {np.std(all_nll_ti)}")


def plot_one_seed(splitting_seed, prefix):
    path_to_data = os.path.join(PATH_TO_DATA, prefix)

    smoother_obs_noise_vars = np.load(
        os.path.join(path_to_data, f"denormalized_smoother_obs_noise_vars_{splitting_seed}.npy"), allow_pickle=True)

    ti_train_times = np.load(os.path.join(path_to_data, f"ti_train_times_{splitting_seed}.npy"), allow_pickle=True)
    ti_train_obs = np.load(os.path.join(path_to_data, f"ti_train_observations_{splitting_seed}.npy"), allow_pickle=True)
    ti_test_times = np.load(os.path.join(path_to_data, f"ti_test_times_{splitting_seed}.npy"), allow_pickle=True)
    ti_test_obs = np.load(os.path.join(path_to_data, f"ti_test_observations_{splitting_seed}.npy"), allow_pickle=True)
    ti_means = np.load(os.path.join(path_to_data, f"ti_means_{splitting_seed}.npy"), allow_pickle=True)
    ti_vars = np.load(os.path.join(path_to_data, f"ti_vars_{splitting_seed}.npy"), allow_pickle=True)
    for traj_id in range(10):
        do_one_plot(means=ti_means[:, traj_id],
                    vars=ti_vars[:, traj_id],
                    obs_vars=smoother_obs_noise_vars,
                    prefix=os.path.join(f"{prefix}_{splitting_seed}", f"ti_{traj_id}"),
                    plot_times=np.linspace(0, 200, 100),
                    original_times=ti_train_times[:, traj_id],
                    original_obs=ti_train_obs[:, traj_id],
                    test_times=ti_test_times[:, traj_id],
                    test_obs=ti_test_obs[:, traj_id]
                    )

    ic_times = np.load(os.path.join(path_to_data, f"ic_times_{splitting_seed}.npy"), allow_pickle=True)
    ic_obs = np.load(os.path.join(path_to_data, f"ic_observations_{splitting_seed}.npy"), allow_pickle=True)
    ic_means = np.load(os.path.join(path_to_data, f"ic_means_{splitting_seed}.npy"), allow_pickle=True)
    ic_vars = np.load(os.path.join(path_to_data, f"ic_vars_{splitting_seed}.npy"), allow_pickle=True)
    for traj_id in range(5):
        do_one_plot(means=ic_means[:, traj_id],
                    vars=ic_vars[:, traj_id],
                    obs_vars=smoother_obs_noise_vars,
                    prefix=os.path.join(f"{prefix}_{splitting_seed}", f"ic_{traj_id}"),
                    plot_times=np.linspace(0, 200, 100),
                    original_times=ic_times[:, traj_id],
                    original_obs=ic_obs[:, traj_id]
                    )

    print("Loading done")


def do_one_plot(means, vars, obs_vars, prefix, plot_times, original_times, original_obs,
                test_times=None, test_obs=None):
    path_to_plots = os.path.join(PATH_TO_PLOTS, prefix)
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    print(prefix)
    plotter = Plotter(simulator=None, initial_conditions=None)
    values_lower_with_on, values_upper_with_on = plotter._create_quantiles_with_obs_noise(
        state_means=means.reshape(means.shape[0], 1, means.shape[1]),
        state_variances=vars.reshape(vars.shape[0], 1, vars.shape[1]),
        denormalized_obs_noise=obs_vars
        )
    _, values_lower_no_on, values_upper_no_on = plotter.create_stds_and_quantiles(
        state_means=means.reshape(means.shape[0], 1, means.shape[1]),
        state_variances=vars.reshape(vars.shape[0], 1, vars.shape[1])
        )

    for dim_id in range(3):
        plt.figure()
        plt.plot(plot_times,
                 means[dim_id][:],
                 "b-")
        plt.fill_between(plot_times,
                         values_lower_no_on[dim_id][0][:],
                         values_upper_no_on[dim_id][0][:],
                         alpha=0.5, fc="b", ec="None")
        plt.fill_between(plot_times,
                         values_lower_with_on[dim_id][0][:],
                         values_upper_with_on[dim_id][0][:],
                         alpha=0.1, fc="r", ec="None")
        plt.plot(original_times[dim_id][:],
                 original_obs[dim_id][:],
                 "r.",
                 markersize=10)
        if dim_id < 2 and test_times is not None:
            plt.plot(test_times[dim_id],
                     test_obs[dim_id],
                     "b.",
                     markersize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_plots, f"state_{dim_id}"), dpi=300)
        plt.close()


def do_one_paper_plot():
    splitting_seed = 5
    plotter = Plotter(simulator=None, initial_conditions=None)

    path_to_data = os.path.join(PATH_TO_DATA, 'final')
    (
        obs_noise_var_final,
        ti_data_final,
        ic_data_final
    ) = (
        [
            np.load(os.path.join(
                path_to_data, f"denormalized_smoother_obs_noise_vars_{splitting_seed}.npy"), allow_pickle=True)
        ],
        [
            np.load(os.path.join(path_to_data, f"ti_train_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_train_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_test_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_test_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_means_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
            np.load(os.path.join(path_to_data, f"ti_vars_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
        ],
        [

            np.load(os.path.join(path_to_data, f"ic_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ic_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ic_means_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
            np.load(os.path.join(path_to_data, f"ic_vars_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :]
        ]
    )

    ti_values_lower_with_on_final, ti_values_upper_with_on_final = plotter._create_quantiles_with_obs_noise(
        state_means=ti_data_final[4].reshape(ti_data_final[4].shape[0], 1, ti_data_final[4].shape[1]),
        state_variances=ti_data_final[5].reshape(ti_data_final[5].shape[0], 1, ti_data_final[5].shape[1]),
        denormalized_obs_noise=obs_noise_var_final[0]
        )
    _, ti_values_lower_no_on_final, ti_values_upper_no_on_final = plotter.create_stds_and_quantiles(
        state_means=ti_data_final[4].reshape(ti_data_final[4].shape[0], 1, ti_data_final[4].shape[1]),
        state_variances=ti_data_final[5].reshape(ti_data_final[5].shape[0], 1, ti_data_final[5].shape[1])
        )
    ic_values_lower_with_on_final, ic_values_upper_with_on_final = plotter._create_quantiles_with_obs_noise(
        state_means=ic_data_final[2].reshape(ic_data_final[2].shape[0], 1, ic_data_final[2].shape[1]),
        state_variances=ic_data_final[3].reshape(ic_data_final[3].shape[0], 1, ic_data_final[3].shape[1]),
        denormalized_obs_noise=obs_noise_var_final[0]
        )
    _, ic_values_lower_no_on_final, ic_values_upper_no_on_final = plotter.create_stds_and_quantiles(
        state_means=ic_data_final[2].reshape(ic_data_final[2].shape[0], 1, ic_data_final[2].shape[1]),
        state_variances=ic_data_final[3].reshape(ic_data_final[3].shape[0], 1, ic_data_final[3].shape[1])
        )

    path_to_data = os.path.join(PATH_TO_DATA, 'smoother')
    (
        obs_noise_var_smoother,
        ti_data_smoother,
        ic_data_smoother
    ) = (
        [
            np.load(os.path.join(
                path_to_data, f"denormalized_smoother_obs_noise_vars_{splitting_seed}.npy"), allow_pickle=True)
        ],
        [
            np.load(os.path.join(path_to_data, f"ti_train_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_train_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_test_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_test_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ti_means_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
            np.load(os.path.join(path_to_data, f"ti_vars_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
        ],
        [

            np.load(os.path.join(path_to_data, f"ic_times_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ic_observations_{splitting_seed}.npy"), allow_pickle=True)[:, 0],
            np.load(os.path.join(path_to_data, f"ic_means_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :],
            np.load(os.path.join(path_to_data, f"ic_vars_{splitting_seed}.npy"), allow_pickle=True)[:, 0, :]
        ]
    )

    ti_values_lower_with_on_smoother, ti_values_upper_with_on_smoother = plotter._create_quantiles_with_obs_noise(
        state_means=ti_data_smoother[4].reshape(ti_data_smoother[4].shape[0], 1, ti_data_smoother[4].shape[1]),
        state_variances=ti_data_smoother[5].reshape(ti_data_smoother[5].shape[0], 1, ti_data_smoother[5].shape[1]),
        denormalized_obs_noise=obs_noise_var_smoother[0]
        )
    _, ti_values_lower_no_on_smoother, ti_values_upper_no_on_smoother = plotter.create_stds_and_quantiles(
        state_means=ti_data_smoother[4].reshape(ti_data_smoother[4].shape[0], 1, ti_data_smoother[4].shape[1]),
        state_variances=ti_data_smoother[5].reshape(ti_data_smoother[5].shape[0], 1, ti_data_smoother[5].shape[1])
        )
    ic_values_lower_with_on_smoother, ic_values_upper_with_on_smoother = plotter._create_quantiles_with_obs_noise(
        state_means=ic_data_smoother[2].reshape(ic_data_smoother[2].shape[0], 1, ic_data_smoother[2].shape[1]),
        state_variances=ic_data_smoother[3].reshape(ic_data_smoother[3].shape[0], 1, ic_data_smoother[3].shape[1]),
        denormalized_obs_noise=obs_noise_var_smoother[0]
        )
    _, ic_values_lower_no_on_smoother, ic_values_upper_no_on_smoother = plotter.create_stds_and_quantiles(
        state_means=ic_data_smoother[2].reshape(ic_data_smoother[2].shape[0], 1, ic_data_smoother[2].shape[1]),
        state_variances=ic_data_smoother[3].reshape(ic_data_smoother[3].shape[0], 1, ic_data_smoother[3].shape[1])
        )

    # do times plot

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    titles = ["min temperature", "max temperature", "day length"]
    ylabels = ["deep GP", 'full DGM']

    # time data
    for dim_id in range(3):
        ## smoother
        # plot mean
        axes[0][dim_id].plot(np.linspace(0, 200, 100), ti_data_smoother[4][dim_id], "b-")
        # plot mean std
        axes[0][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ti_values_lower_with_on_smoother[dim_id][0],
            ti_values_upper_with_on_smoother[dim_id][0],
            alpha=0.1, fc="r", ec="None"
        )
        axes[0][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ti_values_lower_no_on_smoother[dim_id][0],
            ti_values_upper_no_on_smoother[dim_id][0],
            alpha=0.1, fc="b", ec="None"
        )
        # plot red obs
        axes[0][dim_id].plot(ti_data_smoother[0][dim_id],
                             ti_data_smoother[1][dim_id],
                             "r.",
                             markersize=10)
        # plot blue obs
        if dim_id < 2:
            axes[0][dim_id].plot(ti_data_smoother[2][dim_id],
                                 ti_data_smoother[3][dim_id],
                                 "b.",
                                 markersize=10)
        # plot mean
        axes[1][dim_id].plot(np.linspace(0, 200, 100), ti_data_final[4][dim_id], "b-")
        # plot mean std
        axes[1][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ti_values_lower_with_on_final[dim_id][0],
            ti_values_upper_with_on_final[dim_id][0],
            alpha=0.1, fc="r", ec="None"
        )
        axes[1][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ti_values_lower_no_on_final[dim_id][0],
            ti_values_upper_no_on_final[dim_id][0],
            alpha=0.1, fc="b", ec="None"
        )
        # plot red obs
        axes[1][dim_id].plot(ti_data_final[0][dim_id],
                             ti_data_final[1][dim_id],
                             "r.",
                             markersize=10)
        # plot blue obs
        if dim_id < 2:
            axes[1][dim_id].plot(ti_data_final[2][dim_id],
                                 ti_data_final[3][dim_id],
                                 "b.",
                                 markersize=10)
        # set labels
        axes[0][dim_id].set_title(titles[dim_id])
        axes[0][0].set_ylabel(ylabels[0])
        axes[1][0].set_ylabel(ylabels[1])
        axes[1][dim_id].set_xlabel("time")

    fig.tight_layout()
    fig.savefig(os.path.join(PATH_TO_FINAL_PLOTS, "time_interpolation.png"), dpi=300)
    fig.show()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    # do ic plot
    for dim_id in range(3):
        ## smoother
        # plot mean
        axes[0][dim_id].plot(np.linspace(0, 200, 100), ic_data_smoother[2][dim_id], "b-")
        # plot mean std
        axes[0][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ic_values_lower_with_on_smoother[dim_id][0],
            ic_values_upper_with_on_smoother[dim_id][0],
            alpha=0.1, fc="r", ec="None"
        )
        axes[0][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ic_values_lower_no_on_smoother[dim_id][0],
            ic_values_upper_no_on_smoother[dim_id][0],
            alpha=0.1, fc="b", ec="None"
        )
        # plot blue obs
        axes[0][dim_id].plot(ic_data_smoother[0][dim_id],
                             ic_data_smoother[1][dim_id],
                             "b.",
                             markersize=10)
        # plot mean
        axes[1][dim_id].plot(np.linspace(0, 200, 100), ic_data_final[2][dim_id], "b-")
        # plot mean std
        axes[1][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ic_values_lower_with_on_final[dim_id][0],
            ic_values_upper_with_on_final[dim_id][0],
            alpha=0.1, fc="r", ec="None"
        )
        axes[1][dim_id].fill_between(
            np.linspace(0, 200, 100),
            ic_values_lower_no_on_final[dim_id][0],
            ic_values_upper_no_on_final[dim_id][0],
            alpha=0.1, fc="b", ec="None"
        )
        # plot blue obs
        axes[1][dim_id].plot(ic_data_final[0][dim_id],
                             ic_data_final[1][dim_id],
                             "b.",
                             markersize=10)
        # set labels
        axes[0][dim_id].set_title(titles[dim_id])
        axes[0][0].set_ylabel(ylabels[0])
        axes[1][0].set_ylabel(ylabels[1])
        axes[1][dim_id].set_xlabel("time")

    fig.tight_layout()
    fig.savefig(os.path.join(PATH_TO_FINAL_PLOTS, "ic_interpolation.png"), dpi=300)
    fig.show()


def get_data_for_one_seed(splitting_seed: int,
                          experiment_indicator_string: str):
    path_to_data = os.path.join(PATH_TO_DATA, experiment_indicator_string)
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    path_to_plots = os.path.join(PATH_TO_PLOTS, experiment_indicator_string)
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    random_ratio = 0.1

    kernel_seed = 135
    kernel_rng = jax.random.PRNGKey(kernel_seed)

    lr = 0.001
    big_joint_lr = 0.0005
    small_joint_lr = 0.0001
    finetune_lr = 0.00001

    start_step_of_polynomial_decay = 500
    end_step_of_polynomial_decay = 2500
    big_to_small_joint_lr_step = 3500
    lr_switch = 4750
    n_total = 7500

    weight_decay_smoother = 0.000001
    weight_decay_dynamics = 0.000001

    seed = 0

    track_wandb = True
    track_just_loss = True
    numerical_correction = 1e-3

    # scheduling
    final_beta = 1

    n_rff = 50

    num_der_points = 100

    run_dict = {
        'n_l2_pretraining_for_dynamics': -1,  # is overwritten in this file later on anyways.
        'obs_std_prior':  None,
        'seed': seed,
        'data_generation': {
            'type': SimulatorType.USHCN,
            'path_to_data': "~/data/small_chunked_sporadic_with_daylength_ic.csv",
            'path_to_mapping': "~/data/centers_id_mapping_with_daylength_ic.npy",
            'path_to_stations_file': "~/data/ushcn-stations.txt",
            'ic_extractor': InitialConditionExtractorType.NONE,
            'n_trajectories_to_consider': 200,  #
            'dimensions_to_consider': [3, 4, 5],  # {'PRCP': 0, 'SNOW': 1, 'SNWD': 2, 'TMAX': 3, 'TMIN': 4}
            'min_obs_per_trajectory': 3,
            # always use first few trajectories as test_indices, as the plotter will plot the first n regardless of
            # whether they are in test or train.
            'load_augmented_states': True,
            'max_ic_time_slack': 6,
            'test_time_ratio_random': random_ratio,
            'test_time_ratio_consecutive': 0
        },
        'smoother': {
            'kernel': {
                'type': KernelType.RBF_RFF,
                'kwargs': {
                    'feature_rng': kernel_rng,
                    'n_rff': n_rff,
                    'n_features': 5,  # depends on the feature extractor chosen
                },
            },
            'core': {
                'type': TimeAndStatesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(20), Sigmoid, Dense(20), Sigmoid, Dense(20), Sigmoid]}
            },
            'kernel_core': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Identity]}
            },
            'kernel_head': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(5)]}
            },
            'mean_head': {
                'type': FeaturesToFeaturesType.NEURAL_NET,
                'kwargs': {'serial_input': [Dense(3)]}
            }
        },
        'dynamics': {
            'type': DynamicsModel.JOINT_NN_IC_DEPENDENT,
            'kwargs': {'hidden_layers': [20, 50, 20]},
            # 'kwargs': {'hidden_layers': [20, 50, 50, 20]},
            'time_dependent': False,
            'ic_dependent': True
        },
        'betas': {
            'type': BetasType.TRANSITION_BETWEEN_VALUES,
            'kwargs':
                {
                    'transition_start': start_step_of_polynomial_decay,
                    'step_size': 0,
                    'decay_steps': end_step_of_polynomial_decay - start_step_of_polynomial_decay,
                    'final_step_size': 1,
                    'num_dim': 3,
                    'power': 1.0
                },
        },
        'optimizer': {
            'type': Optimizer.ADAM,
            'learning_rate': {
                'type': LearningRateType.PIECEWISE_CONSTANT,
                'kwargs': {'boundaries': [end_step_of_polynomial_decay, big_to_small_joint_lr_step, lr_switch],
                           'values': [lr, big_joint_lr, small_joint_lr, finetune_lr]},
            },
        },
        'priors': {
            'wd_core': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay_smoother}
            },
            'wd_mean_head': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay_smoother}
            },
            'wd_kernel_core': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay_smoother}
            },
            'wd_kernel_head': {
                'type': WeightDecayType.CONSTANT,
                'kwargs': {'step_size': weight_decay_smoother}
            },
            'wd_pure_kernel': {
                'kernel_variance': {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                },
                'kernel_lengthscale': {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                },
                "observation_noise": {
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': 0}
                }
            },
            'wd_dynamics': {
                'type': WeightDecayType.TRANSITION_BETWEEN_VALUESE,
                'kwargs': {
                    'transition_start': start_step_of_polynomial_decay,
                    'step_size': 0,
                    'decay_steps': end_step_of_polynomial_decay - start_step_of_polynomial_decay,
                    'final_step_size': weight_decay_dynamics,
                    'power': 1.0
                }
            }
        },
        'logging': {
            'track_wandb': track_wandb,
            'track_just_loss': track_just_loss,
        },
        'numerical_correction': numerical_correction,
        'num_derivative_points_per_trajectory': num_der_points,
        'split_pretraining': True,
        'create_equidistant_derivative_times': True,
        'random_seed_test_train_split': splitting_seed
    }

    if track_wandb:
        wandb.init(
            project="RS Postprocess",
            # project="Debug_stuff",
            entity="wenkph",
            config=run_dict,
            dir="/cluster/scratch/wenkph/wandb",
            name=f"seed {splitting_seed}"
        )
        config = wandb.config

    print("deriv points: {}".format(num_der_points))

    model = LearnUSHCNInterpolatorIC(**run_dict)

    general_rng, new_rng = jax.random.split(kernel_rng)
    test_trajectory_ids_ic = jax.random.choice(key=new_rng,
                                               a=model.test_indices,
                                               shape=[5],
                                               replace=False)

    """ Calculate initial conditions smoother """

    if experiment_indicator_string == 'final':
        model.load_parameters_from_wandb(
            entity_string='wenkph',
            project_string="RS Sweep Final DGM",
            run_string="seed {}",
            run_args=[splitting_seed],
            model_file_suffix="Final_"
        )
    elif experiment_indicator_string == 'smoother':
        model.load_parameters_from_wandb(
            entity_string='wenkph',
            project_string="RS Sweep Smoother DGM",
            run_string="seed {}",
            run_args=[splitting_seed],
            model_file_suffix="SPre_"
        )

    n_dim = 3
    # generalization across initial conditions
    original_times = [[] for dim in range(n_dim)]
    original_observations = [[] for dim in range(n_dim)]
    for index, trajectory_id in enumerate(test_trajectory_ids_ic):

        unnormalized_pseudo_initial_condition, unnorm_obs_times_of_this_traj, unnorm_observations_of_this_traj = \
            model.get_information_for_trajectory_of_id(trajectory_id)

        for dim_id in range(n_dim):
            original_times[dim_id].append(unnorm_obs_times_of_this_traj[dim_id][0])
            original_observations[dim_id].append(unnorm_observations_of_this_traj[dim_id][
                                            0])  # 0 because only one trajectory. second dimension is traj_id

    initial_conditions_for_smoother = [model.ic_of_full_dataset[traj_id] for traj_id in test_trajectory_ids_ic]

    trajectory_lengths_plotting, joint_times_plotting, joint_repeated_initial_conditions = model.join_trajectories(
        initial_conditions=initial_conditions_for_smoother,
        times=[[jnp.linspace(0, 200, 100)]*len(initial_conditions_for_smoother)] * 3
    )

    joint_normalized_times_plotting, joint_repeated_normalized_initial_conditions_plotting = \
        model.normalize_and_reshape_times_and_ics(joint_times_plotting,
                                                  joint_repeated_initial_conditions
                                                  )

    predictions_plotting = model._get_split_predictions(
        joint_normalized_times_plotting,
        joint_repeated_normalized_initial_conditions_plotting,
        trajectory_lengths_plotting)

    state_means_smoother = predictions_plotting[0]
    state_variances_smoother = predictions_plotting[1]

    np.save(os.path.join(path_to_data, f"ic_times_{splitting_seed}.npy"), np.asarray(original_times))
    np.save(os.path.join(path_to_data, f"ic_observations_{splitting_seed}.npy"), np.asarray(original_observations))
    np.save(os.path.join(path_to_data, f"ic_means_{splitting_seed}.npy"), np.asarray(state_means_smoother))
    np.save(os.path.join(path_to_data, f"ic_vars_{splitting_seed}.npy"), np.asarray(state_variances_smoother))

    _, _, _, nll_state_mean, _, _, _, _ = model.get_metrics(
        initial_conditions=initial_conditions_for_smoother,
        times=original_times,
        observations=original_observations,
        ignore_third_dimension=True
        )

    np.save(os.path.join(path_to_data, f"ic_nll_{splitting_seed}.npy"), nll_state_mean)

    # generalization across time
    initial_conditions_for_smoother = model.test_initial_conditions
    print('Before computing the values for plotting')
    trajectory_lengths, joint_times, joint_repeated_initial_conditions = model.join_trajectories(
        model.test_initial_conditions,
        times=[[jnp.linspace(0, 200, 100)]*len(initial_conditions_for_smoother)] * 3
    )

    joint_normalized_times, joint_repeated_normalized_initial_conditions = model.normalize_and_reshape_times_and_ics(
        joint_times, joint_repeated_initial_conditions)

    if sum(trajectory_lengths[2]) == 0:
        # here, third dimension is empty because we are in test set
        # as a hack for the rest of the code to work, we introduce a fake time and observation, which is later ignored
        ignore_third_dimension = True  # ignore third dimension when calculating the metric, as it only contains the fake obs
        joint_normalized_times[2] = np.array([1.1])
        joint_repeated_normalized_initial_conditions[2] = 0.1 * np.ones_like(
            joint_repeated_normalized_initial_conditions[0][0]).reshape(1, -1)
        trajectory_lengths[2][0] = 1

    denormalized_state_means, denormalized_state_variances, _, _, _, _ = model._get_split_predictions(
        joint_normalized_times,
        joint_repeated_normalized_initial_conditions,
        trajectory_lengths)

    np.save(os.path.join(path_to_data, f"ti_train_times_{splitting_seed}.npy"), np.asarray(model.times))
    np.save(os.path.join(path_to_data, f"ti_train_observations_{splitting_seed}.npy"), np.asarray(model.observations))
    np.save(os.path.join(path_to_data, f"ti_test_times_{splitting_seed}.npy"), np.asarray(model.test_times))
    np.save(os.path.join(path_to_data, f"ti_test_observations_{splitting_seed}.npy"), np.asarray(model.test_observations))
    np.save(os.path.join(path_to_data, f"ti_means_{splitting_seed}.npy"), np.asarray(denormalized_state_means))
    np.save(os.path.join(path_to_data, f"ti_vars_{splitting_seed}.npy"), np.asarray(denormalized_state_variances))

    np.save(os.path.join(path_to_data, f"denormalized_smoother_obs_noise_vars_{splitting_seed}.npy"),
            model.get_denormalized_smoother_observation_noise_variance())

    _, _, _, nll_state_mean, _, _, _, _ = model.get_metrics(
        initial_conditions=model.test_initial_conditions,
        times=model.test_times,
        observations=model.test_observations
    )

    np.save(os.path.join(path_to_data, f"ti_nll_{splitting_seed}.npy"), nll_state_mean)


if __name__ == '__main__':
    main()
