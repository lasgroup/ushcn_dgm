import argparse
import time
from typing import Callable

import os
os.environ['JAX_ENABLE_X64'] = 'True'

from wandb_scripts.runDB import RunDB
import pickle
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

Schedule = Callable[[int], float]

import numpy as np
np.random.seed(14859)
# IMPORTANT: Need same random seed as used for training. RFF sampling depends on random seed. And the deep
# feature extractor adapts during training to the RFF chosen to approximate the kernel. Choosing different
# RFF for the same feature extractor will lead to significant errors, as the RFF approximation is not exact.

if __name__ == '__main__':

    random_ratio = 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument('--wd_dynamics', type=float, required=True)
    parser.add_argument('--wd_smoother', type=float, required=True)
    parser.add_argument('--random_seed', type=int, required=True)
    args = parser.parse_args()

    random_seed = args.random_seed
    np.random.seed(random_seed)
    general_rng = jax.random.PRNGKey(random_seed)
    kernel_rng, general_rng = jax.random.split(general_rng)

    n_smoother_pre = 5000
    n_l2_pre = 0
    n_wasser_pre = 2000
    n_final = 3000

    # n_smoother_pre = 1
    # n_l2_pre = 0
    # n_wasser_pre = 1
    # n_final = 1

    lr = 0.001
    finetune_lr = 0.0001
    lr_switch = n_smoother_pre + n_l2_pre + n_wasser_pre + n_final - 1500

    print(f"smoother: {n_smoother_pre}, L2: {n_l2_pre}, Wasser: {n_wasser_pre}, final: {n_final}")

    weight_decay_smoother = args.wd_smoother
    weight_decay_dynamics = args.wd_dynamics

    seed = 0

    track_wandb = True
    track_just_loss = True
    numerical_correction = 1e-3

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
            'n_trajectories_to_consider': 200,
            'dimensions_to_consider': [3, 4, 5],  # {'PRCP': 0, 'SNOW': 1, 'SNWD': 2, 'TMAX': 3, 'TMIN': 4}
            'min_obs_per_trajectory': 3,
            # always use first few trajectories as test_indices, as the plotter will plot the first n regardless of
            # whether they are in test or train.
            'load_augmented_states': True,
            'max_ic_time_slack': 6,
            'test_time_ratio_random': random_ratio,  #.1,
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
            # project=f"Final_Prediction_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",  # TODO
            project="Debug",
            group=f"{random_seed}",
            entity="wenkph",
            config=run_dict,
            dir="/cluster/scratch/wenkph/wandb",
            name=f"wds_{weight_decay_smoother}_wdd_{weight_decay_dynamics}"
        )
        config = wandb.config

    print("deriv points: {}".format(num_der_points))

    with jax.disable_jit():
        model = ShortLearnUSHCNInterpolatorIC(**run_dict)

        n_steps_for_integration = 100
        general_rng, new_rng = jax.random.split(general_rng)
        test_trajectory_ids = jax.random.choice(key=new_rng,
                                                a=model.train_indices,
                                                shape=[5],
                                                replace=False)
        time_of_generalization_start = 50
        time_of_generalization_end = 100

        """ Calculate initial conditions smoother """

        model.load_parameters_from_wandb(
            entity_string='wenkph',
            project_string=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
            run_string="wds_{}_wdd_{}",
            run_args=[weight_decay_smoother, weight_decay_dynamics],
            group_string="{}",
            group_args=[random_seed],
            model_file_suffix="WPre_"
        )

        pre_smoother_nll = model.get_smoother_generalization_nll_and_plots(trajectory_ids=test_trajectory_ids,
                                                                           start_of_times=time_of_generalization_start,
                                                                           end_of_times=time_of_generalization_end,
                                                                           n_dim=3,
                                                                           plotting_name_suffix="pre_no_int")

        initial_conditions_smoother_mean, initial_conditions_smoother_variances = \
            model.get_initial_conditions_from_trajectory_ids(trajectory_ids=test_trajectory_ids,
                                                             unnormalized_initial_time=time_of_generalization_start)

        print("Initial conditions of smoother calculated.")

        """ Calculate initial conditions final """

        model.load_parameters_from_wandb(
            entity_string='wenkph',
            project_string=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
            run_string="wds_{}_wdd_{}",
            run_args=[weight_decay_smoother, weight_decay_dynamics],
            group_string="{}",
            group_args=[random_seed],
            model_file_suffix="Final_"
        )

        final_smoother_nll = model.get_smoother_generalization_nll_and_plots(trajectory_ids=test_trajectory_ids,
                                                                             start_of_times=time_of_generalization_start,
                                                                             end_of_times=time_of_generalization_end,
                                                                             n_dim=3,
                                                                             plotting_name_suffix="final_no_int")

        initial_conditions_final_mean, initial_conditions_final_variances = \
            model.get_initial_conditions_from_trajectory_ids(trajectory_ids=test_trajectory_ids,
                                                             unnormalized_initial_time=time_of_generalization_start)

        print("Initial conditions of final calculated.")

        """ Calculate test metrics smoother """

        general_rng, new_rng = jax.random.split(general_rng)
        model.plot_trajectories_with_numerical_integration_by_id(
            trajectory_ids=test_trajectory_ids,
            unnormalized_initial_condition_means=initial_conditions_smoother_mean,  # List over trajectories with means in it
            unnormalized_initial_condition_variances=initial_conditions_smoother_variances,  # List over trajectoreis with variances in it
            rng=new_rng,
            name_suffix="pre_final",
            n_samples=n_steps_for_integration,
            start_of_times=time_of_generalization_start,
            end_of_times=time_of_generalization_end
        )

        print("pre initial conditions final parameters plotted.")

        general_rng, new_rng = jax.random.split(general_rng)
        model.plot_trajectories_with_numerical_integration_by_id(
            trajectory_ids=test_trajectory_ids,
            unnormalized_initial_condition_means=initial_conditions_final_mean,  # List over trajectories with means in it
            unnormalized_initial_condition_variances=initial_conditions_final_variances,  # List over trajectoreis with variances in it
            rng=new_rng,
            name_suffix="final_final",
            n_samples=n_steps_for_integration,
            start_of_times=time_of_generalization_start,
            end_of_times=time_of_generalization_end
        )

        print("final initial conditions final parameters plotted.")

        test_nll_final_with_pre_ic = []
        test_nll_final_with_final_ic = []
        for trajectory_id, pre_ic_mean, pre_ic_var, final_ic_mean, final_ic_var in zip(
                test_trajectory_ids, initial_conditions_smoother_mean, initial_conditions_smoother_variances,
                initial_conditions_final_mean, initial_conditions_final_variances):
            general_rng, new_rng = jax.random.split(general_rng)
            test_nll_final_with_pre_ic.append(model.get_obs_nll_via_numerical_integration_by_id_and_ic(
                trajectory_id, pre_ic_mean, pre_ic_var, new_rng, name="pre_pre", n_samples=n_steps_for_integration,
                start_of_times=time_of_generalization_start, end_of_times=time_of_generalization_end
            ))
            general_rng, new_rng = jax.random.split(general_rng)
            test_nll_final_with_final_ic.append(model.get_obs_nll_via_numerical_integration_by_id_and_ic(
                trajectory_id, final_ic_mean, final_ic_var, new_rng, name="pre_final", n_samples=n_steps_for_integration,
                start_of_times=time_of_generalization_start, end_of_times=time_of_generalization_end
            ))

        print("metrics calculated for final parameters")

        model.load_parameters_from_wandb(
            entity_string='wenkph',
            project_string=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
            run_string="wds_{}_wdd_{}",
            run_args=[weight_decay_smoother, weight_decay_dynamics],
            group_string="{}",
            group_args=[random_seed],
            model_file_suffix="WPre_"
        )

        print("pretrained parameters loaded")

        general_rng, new_rng = jax.random.split(general_rng)
        model.plot_trajectories_with_numerical_integration_by_id(
            trajectory_ids=test_trajectory_ids,
            unnormalized_initial_condition_means=initial_conditions_smoother_mean,  # List over trajectories with means in it
            unnormalized_initial_condition_variances=initial_conditions_smoother_variances,  # List over trajectoreis with variances in it
            rng=new_rng,
            name_suffix="pre_pre",
            n_samples=n_steps_for_integration,
            start_of_times=time_of_generalization_start,
            end_of_times=time_of_generalization_end
        )

        print("pre initial conditions, pre parameters plotted.")

        general_rng, new_rng = jax.random.split(general_rng)
        model.plot_trajectories_with_numerical_integration_by_id(
            trajectory_ids=test_trajectory_ids,
            unnormalized_initial_condition_means=initial_conditions_final_mean,  # List over trajectories with means in it
            unnormalized_initial_condition_variances=initial_conditions_final_variances,  # List over trajectoreis with variances in it
            rng=new_rng,
            name_suffix="final_pre",
            n_samples=n_steps_for_integration,
            start_of_times=time_of_generalization_start,
            end_of_times=time_of_generalization_end
        )

        print("pre initial conditions, pre parameters plotted")

        test_nll_pretrained_with_pre_ic = []
        test_nll_pretrained_with_final_ic = []
        for trajectory_id, pre_ic_mean, pre_ic_var, final_ic_mean, final_ic_var in zip(
                test_trajectory_ids, initial_conditions_smoother_mean, initial_conditions_smoother_variances,
                initial_conditions_final_mean, initial_conditions_final_variances):
            general_rng, new_rng = jax.random.split(general_rng)
            test_nll_pretrained_with_pre_ic.append(model.get_obs_nll_via_numerical_integration_by_id_and_ic(
                trajectory_id, pre_ic_mean, pre_ic_var, new_rng, name="pre_pre", n_samples=n_steps_for_integration,
                start_of_times=time_of_generalization_start, end_of_times=time_of_generalization_end
            ))
            general_rng, new_rng = jax.random.split(general_rng)
            test_nll_pretrained_with_final_ic.append(model.get_obs_nll_via_numerical_integration_by_id_and_ic(
                trajectory_id, final_ic_mean, final_ic_var, new_rng, name="pre_final", n_samples=n_steps_for_integration,
                start_of_times=time_of_generalization_start, end_of_times=time_of_generalization_end
            ))

        print("metrics for pretrained parameters calculated")

        wandb.log({
            "test nll pre pre": jnp.mean(jnp.array(test_nll_pretrained_with_pre_ic)),
            "test nll pre final": jnp.mean(jnp.array(test_nll_pretrained_with_final_ic)),
            "test nll final pre": jnp.mean(jnp.array(test_nll_final_with_pre_ic)),
            "test nll final final": jnp.mean(jnp.array(test_nll_final_with_final_ic)),
            "pre nll no int": jnp.mean(jnp.array(pre_smoother_nll)),
            "final nll no int": jnp.mean(jnp.array(final_smoother_nll))
        })
