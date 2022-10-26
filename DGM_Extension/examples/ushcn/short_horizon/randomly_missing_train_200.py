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
            project=f"Final_Sweep_200_{random_ratio}_{run_dict['data_generation']['n_trajectories_to_consider']}",
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

    model.n_l2_pretraining_for_dynamics = 0
    model.train(num_steps_smoother_pretraining=n_smoother_pre,
                num_steps_dynamics_pretraining=0,
                number_of_steps=0)
    model.store_parameters(name="SPre_")
    model.plot_first_few_trajectories_at_all_times(n_trajectories_to_plot=10, plot_equidistantly=True,
                                                   name_prefix='SPre_')
    model.calculate_metrics_and_push_to_wandb_on_train('SPre_', ignore_third_dimension=True)
    model.calculate_metrics_and_push_to_wandb_on_test('SPre_', ignore_third_dimension=True)
    model.calculate_metrics_and_push_to_wandb_on_all_data('SPre_', ignore_third_dimension=True)
    print("Smoother_pretrained")

    # model.n_l2_pretraining_for_dynamics = n_l2_pre
    # model.train(num_steps_smoother_pretraining=0,
    #             num_steps_dynamics_pretraining=0,
    #             number_of_steps=0)
    # model.store_parameters(name="L2Pre_")
    # # model.plot_first_few_trajectories_at_all_times(n_trajectories_to_plot=10, plot_equidistantly=True,
    # #                                                name_prefix='L2Pre_')
    # # model.calculate_metrics_and_push_to_wandb_on_train('L2Pre_', ignore_third_dimension=True)
    # # model.calculate_metrics_and_push_to_wandb_on_test('L2Pre_', ignore_third_dimension=True)
    # # model.calculate_metrics_and_push_to_wandb_on_all_data('L2Pre_', ignore_third_dimension=True)
    # print("L2 pretrained")

    model.n_l2_pretraining_for_dynamics = 0
    model.train(num_steps_smoother_pretraining=0,
                num_steps_dynamics_pretraining=n_wasser_pre,
                number_of_steps=0)
    model.store_parameters(name="WPre_")
    # model.plot_first_few_trajectories_at_all_times(n_trajectories_to_plot=10, plot_equidistantly=True,
    #                                                name_prefix='WPre_')
    # model.calculate_metrics_and_push_to_wandb_on_train('WPre_', ignore_third_dimension=True)
    # model.calculate_metrics_and_push_to_wandb_on_test('WPre_', ignore_third_dimension=True)
    # model.calculate_metrics_and_push_to_wandb_on_all_data('WPre_', ignore_third_dimension=True)
    print("Wasserstein pretrained")

    model.train(num_steps_smoother_pretraining=0,
                num_steps_dynamics_pretraining=0,
                number_of_steps=n_final)
    model.store_parameters(name="Final_")
    model.plot_first_few_trajectories_at_all_times(n_trajectories_to_plot=10, plot_equidistantly=True,
                                                   name_prefix='Final_')
    model.calculate_metrics_and_push_to_wandb_on_train('Final_', ignore_third_dimension=True)
    model.calculate_metrics_and_push_to_wandb_on_test('Final_', ignore_third_dimension=True)
    model.calculate_metrics_and_push_to_wandb_on_all_data('Final_', ignore_third_dimension=True)
    print("Final training completed.")
