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

PATH_TO_STORAGE = '/home/philippe/GRU_Playground/new_splits/'
n_trajectories_to_consider = 200

for file_name_suffix in ['pre', 'final']:
    all_wds_smoother = np.load(os.path.join(PATH_TO_STORAGE,
                                            str(n_trajectories_to_consider),
                                            'best_hypers',
                                            'best_wd_smoother_' + file_name_suffix + '.npy')
                               )
    all_wds_dynamics = np.load(os.path.join(PATH_TO_STORAGE,
                                            str(n_trajectories_to_consider),
                                            'best_hypers',
                                            'best_wd_dynamics_' + file_name_suffix + '.npy'))
    for random_seed, weight_decay_smoother, weight_decay_dynamics in zip(
            5 * np.arange(1, 11), all_wds_smoother, all_wds_dynamics):

        random_ratio = 0.05

        print(file_name_suffix)
        print(random_seed)

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
            'seed': seed,  # seed used for initializations of everything.
            'data_generation': {
                'type': SimulatorType.USHCN,
                'path_to_data': "~/data/small_chunked_sporadic_with_daylength_ic.csv",
                'path_to_mapping': "~/data/centers_id_mapping_with_daylength_ic.npy",
                'path_to_stations_file': "~/data/ushcn-stations.txt",
                'ic_extractor': InitialConditionExtractorType.NONE,
                'n_trajectories_to_consider': n_trajectories_to_consider,
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

        # if track_wandb:
        #     wandb.init(
        #         project=f"Hyperparameter Extraction",
        #         # project="Debug",
        #         group=f"{random_seed}",
        #         entity="wenkph",
        #         config=run_dict,
        #         dir="/cluster/scratch/wenkph/wandb",
        #         name=f"wds_{weight_decay_smoother}_wdd_{weight_decay_dynamics}"
        #     )
        #     config = wandb.config
        #
        # print("deriv points: {}".format(num_der_points))

        with jax.disable_jit():
            model = ShortLearnUSHCNInterpolatorIC(**run_dict)

            train_times = model.times
            train_obs = model.observations
            validation_times = model.test_times
            validation_obs = model.test_observations
            train_indices = model.train_indices

            new_test_times = model.extract_trajectories_by_indices(model.future_times, train_indices)
            new_test_obs = model.extract_trajectories_by_indices(model.future_obs, train_indices)

            # filter by start and end time
            filtered_test_times = []
            filtered_test_observations = []
            start_of_times = 50
            end_of_times = 100
            for dim_id in range(len(new_test_times)):
                filtered_obs_of_this_dim = []
                filtered_times_of_this_dim = []
                for traj_id in range(len(new_test_times[0])):
                    indices_for_this_dim = [start_of_times <= a <= end_of_times for a in new_test_times[dim_id][traj_id]]
                    indices_for_this_dim = jnp.array(indices_for_this_dim)
                    indices_for_this_dim = jnp.where(indices_for_this_dim)
                    filtered_times_of_this_dim.append(np.asarray(new_test_times[dim_id][traj_id][indices_for_this_dim] - start_of_times))
                    filtered_obs_of_this_dim.append(np.asarray(new_test_obs[dim_id][traj_id][indices_for_this_dim]))
                filtered_test_times.append(filtered_times_of_this_dim)
                filtered_test_observations.append(filtered_obs_of_this_dim)



            path_to_data_files = os.path.join(PATH_TO_STORAGE, str(n_trajectories_to_consider), "raw_data",
                                              str(random_seed), file_name_suffix)
            if not os.path.exists(path_to_data_files):
                os.makedirs(path_to_data_files)


            print(f"extraction of seed {random_seed} done.")
            np.save(os.path.join(path_to_data_files, 'train_times'), train_times)
            np.save(os.path.join(path_to_data_files, 'train_obs'), train_obs)
            np.save(os.path.join(path_to_data_files, 'validation_times'), validation_times)
            np.save(os.path.join(path_to_data_files, 'validation_obs'), validation_obs)
            np.save(os.path.join(path_to_data_files, 'test_times'), filtered_test_times)
            np.save(os.path.join(path_to_data_files, 'test_obs'), filtered_test_observations)
