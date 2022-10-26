from typing import Callable

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import jax.numpy as jnp
import wandb
from jax.experimental.stax import Dense, Identity, Sigmoid

from dgm.main.learn_system import ShortLearnUSHCNInterpolatorIC
from dgm.schedules.betas import BetasType
from dgm.schedules.learning_rate import LearningRateType
from dgm.schedules.weight_decay import WeightDecayType
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType
from dgm.utils.representatives import KernelType, Optimizer, DynamicsModel, SimulatorType, InitialConditionExtractorType

from dgm.plotter.plotter import Plotter

from matplotlib import pyplot as plt

Schedule = Callable[[int], float]

import numpy as np
import argparse
np.random.seed(14859)

PATH_TO_HYPERPARAMETERS = os.path.expanduser('~/GRU_Playground/new_splits/200/best_hypers')
PATH_TO_RESULTS = os.path.expanduser('~/ultimate_thesis_plots/dgm/data')
PATH_TO_PLOTS = os.path.expanduser('~/ultimate_thesis_plots/dgm/plots')

if not os.path.exists(PATH_TO_RESULTS):
    os.makedirs(PATH_TO_RESULTS)


def main():
    # create_data()
    # plot_all_data()
    plot_joint_paper_plot()


def create_data():
    best_wd_smoother_pre = np.load(os.path.join(PATH_TO_HYPERPARAMETERS, 'best_wd_smoother_pre.npy'))
    best_wd_dynamics_pre = np.load(os.path.join(PATH_TO_HYPERPARAMETERS, 'best_wd_dynamics_pre.npy'))
    best_wd_smoother_final = np.load(os.path.join(PATH_TO_HYPERPARAMETERS, 'best_wd_smoother_final.npy'))
    best_wd_dynamics_final = np.load(os.path.join(PATH_TO_HYPERPARAMETERS, 'best_wd_dynamics_final.npy'))
    all_random_seeds = 5 * np.arange(1, 11)

    print("Start pre stuff")

    index_int = 7
    print(f"doing run {index_int}")

    with jax.disable_jit():

        for i in range(len(all_random_seeds)):
            if i != index_int:
                print(f"pass {i}")
                continue
            print(f"doing run {index_int}")
            print(f"with seed {all_random_seeds[i]}")

            random_seed = all_random_seeds[i]
            wd_smoother = best_wd_smoother_pre[i]
            wd_dynamics = best_wd_dynamics_pre[i]

            print("Doing pre")

            store_data(wd_smoother=wd_smoother,
                       wd_dynamics=wd_dynamics,
                       random_seed=random_seed,
                       label="pre"
                       )

            print("Doing final")

            random_seed = all_random_seeds[i]
            wd_smoother = best_wd_smoother_final[i]
            wd_dynamics = best_wd_dynamics_final[i]

            store_data(wd_smoother=wd_smoother,
                       wd_dynamics=wd_dynamics,
                       random_seed=random_seed,
                       label="final"
                       )


def store_data(wd_smoother, wd_dynamics, random_seed, label):

    with jax.disable_jit():
        random_ratio = 0.05

        np.random.seed(random_seed)
        general_rng = jax.random.PRNGKey(random_seed)
        kernel_rng, general_rng = jax.random.split(general_rng)

        n_smoother_pre = 5000
        n_l2_pre = 0
        n_wasser_pre = 2000
        n_final = 3000

        lr = 0.001
        finetune_lr = 0.0001
        lr_switch = n_smoother_pre + n_l2_pre + n_wasser_pre + n_final - 1500

        print(f"smoother: {n_smoother_pre}, L2: {n_l2_pre}, Wasser: {n_wasser_pre}, final: {n_final}")

        weight_decay_smoother = wd_smoother
        weight_decay_dynamics = wd_dynamics

        seed = 0

        track_wandb = True
        track_just_loss = True
        numerical_correction = 1e-3

        final_beta = 1

        n_rff = 50

        num_der_points = 100

        run_dict = {
            'n_l2_pretraining_for_dynamics': -1,  # is overwritten in this file later on anyways.
            'obs_std_prior': None,
            'seed': seed,
            'data_generation': {
                'type': SimulatorType.USHCN,
                'path_to_data': "~/data/small_chunked_sporadic_with_daylength_ic.csv",
                'path_to_mapping': "~/data/centers_id_mapping_with_daylength_ic.npy",
                'path_to_stations_file': "~/data/ushcn-stations.txt",
                'ic_extractor': InitialConditionExtractorType.NONE,
                'n_trajectories_to_consider': 200,
                'dimensions_to_consider': [3, 4, 5],  # {'PRCP': 0, 'SNOW': 1, 'SNWD': 2, 'TMAX': 3, 'TMIN': 4}
                'min_obs_per_trajectory': 3,
                # always use first few trajectories as test_indices, as the plotter will plot the first n regardless of
                # whether they are in test or train.
                'load_augmented_states': True,
                'max_ic_time_slack': 6,
                'test_time_ratio_random': random_ratio,  # .1,
                'test_time_ratio_consecutive': 0,
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
                'kwargs': {'hidden_layers': [20, 50, 50, 20]},
                'time_dependent': False,
                'ic_dependent': True
            },
            'betas': {
                'type': BetasType.CONSTANT,
                'kwargs': {'value': 1, 'num_dim': 3},
            },
            'optimizer': {
                'type': Optimizer.ADAM,
                # 'learning_rate': {
                #     'type': LearningRateType.CONSTANT,
                #     'kwargs': {'step_size': lr},
                # },
                'learning_rate': {
                    'type': LearningRateType.PIECEWISE_CONSTANT,
                    'kwargs': {'boundaries': [lr_switch], 'values': [lr, finetune_lr]},
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
                    'type': WeightDecayType.CONSTANT,
                    'kwargs': {'step_size': weight_decay_dynamics}
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
            'horizon_split_time': 50,
            'random_seed_test_train_split': random_seed
        }

        if track_wandb:
            wandb.init(
                project=f"Create paper plots DGM",
                # project="Debug",
                group=f"{random_seed}",
                entity="wenkph",
                config=run_dict,
                dir="/cluster/scratch/wenkph/wandb",
                name=f"wds_{weight_decay_smoother}_wdd_{weight_decay_dynamics}"
            )
            config = wandb.config

        print("deriv points: {}".format(num_der_points))

        model = ShortLearnUSHCNInterpolatorIC(**run_dict)

        n_steps_for_integration = 100
        general_rng, new_rng = jax.random.split(general_rng)
        test_trajectory_ids = jax.random.choice(key=new_rng,
                                                a=model.train_indices,
                                                shape=[5],
                                                replace=False)
        test_trajectory_ids = jnp.array([570, 571])
        time_of_generalization_start = 50
        time_of_generalization_end = 100

        """ Calculate initial conditions smoother """

        if label == "pre":

            model.load_parameters_from_wandb(
                entity_string='wenkph',
                project_string=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
                run_string="wds_{}_wdd_{}",
                run_args=[weight_decay_smoother, weight_decay_dynamics],
                group_string="{}",
                group_args=[random_seed],
                model_file_suffix="WPre_"
            )
        elif label == "final":
            model.load_parameters_from_wandb(
                entity_string='wenkph',
                project_string=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
                run_string="wds_{}_wdd_{}",
                run_args=[weight_decay_smoother, weight_decay_dynamics],
                group_string="{}",
                group_args=[random_seed],
                model_file_suffix="Final_"
            )
        else:
            raise ValueError("label is not a correct label")

        initial_conditions_mean, initial_conditions_variances = \
            model.get_initial_conditions_from_trajectory_ids(trajectory_ids=test_trajectory_ids,
                                                             unnormalized_initial_time=time_of_generalization_start)

        smoother_obs_noise_vars = model.get_denormalized_smoother_observation_noise_variance();

        # general_rng, new_rng = jax.random.split(general_rng)  # TODO uncomment all after here
        # (
        #     original_times, original_obs,
        #     num_int_mean, num_int_vars,
        #     smoother_mean, smoother_vars
        # ) = model.get_data_for_final_ushcn_plots(
        #     trajectory_ids=test_trajectory_ids,
        #     unnormalized_initial_condition_means=initial_conditions_mean,
        #     # List over trajectories with means in it
        #     unnormalized_initial_condition_variances=initial_conditions_variances,
        #     # List over trajectoreis with variances in it
        #     rng=new_rng,
        #     name_suffix="final_final",
        #     n_samples=n_steps_for_integration,
        #     start_of_times=time_of_generalization_start,
        #     end_of_times=time_of_generalization_end
        # )

        path_to_results = os.path.join(PATH_TO_RESULTS, label)
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        # np.save(os.path.join(path_to_results, "original_times"), original_times)
        # np.save(os.path.join(path_to_results, "original_obs"), original_obs)
        # np.save(os.path.join(path_to_results, "num_int_mean"), num_int_mean)
        # np.save(os.path.join(path_to_results, "num_int_vars"), num_int_vars)
        # np.save(os.path.join(path_to_results, "smoother_mean"), smoother_mean)
        # np.save(os.path.join(path_to_results, "smoother_vars"), smoother_vars)

        np.save(os.path.join(path_to_results, "denormed_smoother_obs_noise_vars"), smoother_obs_noise_vars)

        print("saved all data")


def plot_joint_paper_plot():
    original_times = np.load(os.path.join(PATH_TO_RESULTS, "pre", "original_times.npy"), allow_pickle=True)
    original_times = original_times / 100 + 0.5
    original_obs = np.load(os.path.join(PATH_TO_RESULTS, "pre", "original_obs.npy"), allow_pickle=True)
    num_int_mean_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "num_int_mean.npy"), allow_pickle=True)
    num_int_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "num_int_vars.npy"), allow_pickle=True)
    smoother_mean_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "smoother_mean.npy"), allow_pickle=True)
    smoother_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "smoother_vars.npy"), allow_pickle=True)
    num_int_mean_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "num_int_mean.npy"), allow_pickle=True)
    num_int_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "num_int_vars.npy"), allow_pickle=True)
    smoother_mean_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "smoother_mean.npy"), allow_pickle=True)
    smoother_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "smoother_vars.npy"), allow_pickle=True)
    obs_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "denormed_smoother_obs_noise_vars.npy"),
                           allow_pickle=True)
    obs_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "denormed_smoother_obs_noise_vars.npy"),
                             allow_pickle=True)

    plot_times = np.linspace(0.5, 1, 100)

    all_data_raw = [
        [smoother_mean_pre, smoother_vars_pre, obs_vars_pre],
        [smoother_mean_final, smoother_vars_final, obs_vars_final],
        [num_int_mean_pre, num_int_vars_pre, obs_vars_pre],
        [num_int_mean_final, num_int_vars_final, obs_vars_final]
    ]

    plotter = Plotter(simulator=None, initial_conditions=None)
    all_data = []
    for data_triple in all_data_raw:
        values_lower_with_on, values_upper_with_on = plotter._create_quantiles_with_obs_noise(
            state_means=data_triple[0],
            state_variances=data_triple[1],
            denormalized_obs_noise=data_triple[2])
        _, values_lower_no_on, values_upper_no_on = plotter.create_stds_and_quantiles(
            state_means=data_triple[0],
            state_variances=data_triple[1]
        )
        all_data.append([
            data_triple[0],  # state_means
            values_lower_with_on,
            values_upper_with_on,
            values_lower_no_on,
            values_upper_no_on
        ])

    fig, axes = plt.subplots(4, 3, figsize=(10, 13))

    titles = ["min temperature", "max temperature", "day length"]
    ylabels = ["nip", "nif", "wip", "wif"]

    for dim_id in range(3):
        for model_id in range(4):
            axes[model_id][dim_id].plot(plot_times*100,
                                        all_data[model_id][0][dim_id][0][:],
                                        "b-")
            axes[model_id][dim_id].fill_between(plot_times*100,
                                                all_data[model_id][1][dim_id][0][:],
                                                all_data[model_id][2][dim_id][0][:],
                                                alpha=0.1, fc="r", ec="None")
            axes[model_id][dim_id].fill_between(plot_times*100,
                                                all_data[model_id][3][dim_id][0][:],
                                                all_data[model_id][4][dim_id][0][:],
                                                alpha=0.5, fc="b", ec="None")
            axes[model_id][dim_id].plot(original_times[dim_id][0][:]*100,
                                        original_obs[dim_id][0][:],
                                        "b.", markersize=10)
            axes[0][dim_id].set_title(titles[dim_id])
            axes[model_id][0].set_ylabel(ylabels[model_id])
        axes[3][dim_id].set_xlabel("time")

    fig.tight_layout()
    fig.savefig(os.path.join('/home/philippe/ultimate_thesis_plots/dgm', 'time_extrapolation.png'), dpi=300)
    fig.show()


def plot_all_data():
    original_times = np.load(os.path.join(PATH_TO_RESULTS, "pre", "original_times.npy"), allow_pickle=True)
    original_times = original_times / 100 + 0.5
    original_obs = np.load(os.path.join(PATH_TO_RESULTS, "pre", "original_obs.npy"), allow_pickle=True)
    num_int_mean_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "num_int_mean.npy"), allow_pickle=True)
    num_int_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "num_int_vars.npy"), allow_pickle=True)
    smoother_mean_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "smoother_mean.npy"), allow_pickle=True)
    smoother_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "smoother_vars.npy"), allow_pickle=True)
    num_int_mean_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "num_int_mean.npy"), allow_pickle=True)
    num_int_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "num_int_vars.npy"), allow_pickle=True)
    smoother_mean_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "smoother_mean.npy"), allow_pickle=True)
    smoother_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "smoother_vars.npy"), allow_pickle=True)
    obs_vars_pre = np.load(os.path.join(PATH_TO_RESULTS, "pre", "denormed_smoother_obs_noise_vars.npy"), allow_pickle=True)
    obs_vars_final = np.load(os.path.join(PATH_TO_RESULTS, "final", "denormed_smoother_obs_noise_vars.npy"), allow_pickle=True)

    plot_times = np.linspace(0.5, 1, 100)

    traj_id = 0
    for dim_id in range(3):
        path_to_plots = os.path.join(PATH_TO_PLOTS, "original")
        if not os.path.exists(path_to_plots):
            os.makedirs(path_to_plots)
        plt.figure()
        plt.plot(original_times[dim_id][traj_id][:], original_obs[dim_id][traj_id][:], 'x')
        plt.savefig(os.path.join(path_to_plots, f"state_{dim_id}"))

    do_one_plot(means=num_int_mean_pre, vars=num_int_vars_pre, obs_vars=obs_vars_pre, prefix="wip",
                plot_times=plot_times, original_times=original_times, original_obs=original_obs)
    do_one_plot(means=num_int_mean_final, vars=num_int_vars_final, obs_vars=obs_vars_final, prefix="wif",
                plot_times=plot_times, original_times=original_times, original_obs=original_obs)
    do_one_plot(means=smoother_mean_pre, vars=smoother_vars_pre, obs_vars=obs_vars_pre, prefix="nip",
                plot_times=plot_times, original_times=original_times, original_obs=original_obs)
    do_one_plot(means=smoother_mean_final, vars=smoother_vars_final, obs_vars=obs_vars_final, prefix="nif",
                plot_times=plot_times, original_times=original_times, original_obs=original_obs)


def do_one_plot(means, vars, obs_vars, prefix, plot_times, original_times, original_obs):
    path_to_plots = os.path.join(PATH_TO_PLOTS, prefix)
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    plotter = Plotter(simulator=None, initial_conditions=None)
    values_lower_with_on, values_upper_with_on = plotter._create_quantiles_with_obs_noise(state_means=means,
                                                                                          state_variances=vars,
                                                                                          denormalized_obs_noise=obs_vars
                                                                                          )
    _, values_lower_no_on, values_upper_no_on = plotter.create_stds_and_quantiles(state_means=means,
                                                                                  state_variances=vars
                                                                                  )

    traj_id = 0
    for dim_id in range(3):
        plt.figure()
        plt.plot(plot_times,
                 means[dim_id][traj_id][:],
                 "b-")
        plt.fill_between(plot_times,
                         values_lower_no_on[dim_id][traj_id][:],
                         values_upper_no_on[dim_id][traj_id][:],
                         alpha=0.5, fc="b", ec="None")
        plt.fill_between(plot_times,
                         values_lower_with_on[dim_id][traj_id][:],
                         values_upper_with_on[dim_id][traj_id][:],
                         alpha=0.1, fc="r", ec="None")
        plt.plot(original_times[dim_id][traj_id][:],
                 original_obs[dim_id][traj_id][:],
                 "r.",
                 markersize=10)
        plt.savefig(os.path.join(path_to_plots, f"state_{dim_id}"), dpi=300)


if __name__ == "__main__":
    main()
