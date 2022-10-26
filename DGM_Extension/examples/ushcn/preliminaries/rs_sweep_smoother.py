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

from dgm.main.learn_system import LearnUSHCNInterpolatorIC
from dgm.schedules.betas import BetasType
from dgm.schedules.learning_rate import LearningRateType
from dgm.schedules.weight_decay import WeightDecayType
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType
from dgm.utils.representatives import KernelType, Optimizer, DynamicsModel, SimulatorType, InitialConditionExtractorType

Schedule = Callable[[int], float]

import numpy as np
np.random.seed(14859)

if __name__ == '__main__':

    random_ratio = 0.1

    kernel_seed = 135
    kernel_rng = jax.random.PRNGKey(kernel_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    splitting_seed = args.seed

    lr = 0.001
    big_joint_lr = 0.0005
    small_joint_lr = 0.0001
    finetune_lr = 0.00001

    start_step_of_polynomial_decay = 500
    end_step_of_polynomial_decay = 2500
    big_to_small_joint_lr_step = 3500
    lr_switch = 4750
    n_total = 7500

    weight_decay_smoother = 0.00001
    weight_decay_dynamics = 0

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
            project="RS Sweep Smoother DGM",
            # project="Debug_stuff",
            entity="wenkph",
            config=run_dict,
            dir="/cluster/scratch/wenkph/wandb",
            name=f"seed {splitting_seed}"
        )
        config = wandb.config

    print("deriv points: {}".format(num_der_points))

    model = LearnUSHCNInterpolatorIC(**run_dict)

    model.train(num_steps_smoother_pretraining=5000,
                num_steps_dynamics_pretraining=0,
                number_of_steps=0)
    model.store_parameters(name="SPre_")
    model.plot_first_few_trajectories_at_all_times(n_trajectories_to_plot=10, plot_equidistantly=True,
                                                   name_prefix='SPre_')
    model.calculate_metrics_and_push_to_wandb_on_train('SPre_', ignore_third_dimension=True)
    model.calculate_metrics_and_push_to_wandb_on_test('SPre_', ignore_third_dimension=True)
