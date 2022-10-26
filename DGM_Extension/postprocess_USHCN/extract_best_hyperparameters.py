import numpy as np
import os

from wandb_scripts.runDB import RunDB


random_ratio = 0.05
n_trajectories_to_consider = 200

PATH_TO_STORAGE = f"/home/philippe/GRU_Playground/splits/{n_trajectories_to_consider}/best_hypers"

all_best_wd_smoother_pre = []
all_best_wd_dynamics_pre = []
all_best_wd_smoother_final = []
all_best_wd_dynamics_final = []

for random_seed in 5*np.arange(1, 11):
    try:
        run_db = RunDB(project_string=f"Final_Sweep_{n_trajectories_to_consider}_{random_ratio}_{n_trajectories_to_consider}",
                       group_string=f"{random_seed}",
                       run_string="wds_{}_wdd_{}"
                       )

        min_nll_pre = np.inf
        min_nll_final = np.inf
        best_wd_smoother_pre = -1
        best_wd_dynamics_pre = -1
        best_wd_smoother_final = -1
        best_wd_dynamics_final = -1

        for curr_wd_smoother in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0]:
            for curr_wd_dynamics in [0.0, 0.000001, 0.00001, 0.0001]:

                run_id = run_db.get_run_id(group_args=[],
                                           run_args=[curr_wd_smoother, curr_wd_dynamics]
                                           )

                curr_validation_nll_pre = run_db.get_variable_value(variable_name="SPre__test_nll_state_mean",
                                                                    run_id=run_id
                                                                    )
                if curr_validation_nll_pre < min_nll_pre:
                    min_nll_pre = curr_validation_nll_pre
                    best_wd_smoother_pre = curr_wd_smoother
                    best_wd_dynamics_pre = curr_wd_dynamics

                curr_validation_nll_final = run_db.get_variable_value(variable_name="Final__test_nll_state_mean",
                                                                      run_id=run_id
                                                                      )
                if curr_validation_nll_final < min_nll_final:
                    min_nll_final = curr_validation_nll_final
                    best_wd_smoother_final = curr_wd_smoother
                    best_wd_dynamics_final = curr_wd_dynamics

        all_best_wd_smoother_pre.append(best_wd_smoother_pre)
        all_best_wd_dynamics_pre.append(best_wd_dynamics_pre)
        all_best_wd_smoother_final.append(best_wd_smoother_final)
        all_best_wd_dynamics_final.append(best_wd_dynamics_final)
    except LookupError:
        print(f"no run for wd_smoother {curr_wd_smoother}, wd_dynamics {curr_wd_dynamics} for seed {random_seed}")

if not os.path.exists(PATH_TO_STORAGE):
    os.makedirs(PATH_TO_STORAGE)

print("smoother pre")
print(all_best_wd_smoother_pre)
np.save(os.path.join(PATH_TO_STORAGE, 'best_wd_smoother_pre'), np.asarray(all_best_wd_smoother_pre))

print("dynamics pre")
print(all_best_wd_dynamics_pre)
np.save(os.path.join(PATH_TO_STORAGE, 'best_wd_dynamics_pre'), np.asarray(all_best_wd_dynamics_pre))

print("smoother final")
print(all_best_wd_smoother_final)
np.save(os.path.join(PATH_TO_STORAGE, 'best_wd_smoother_final'), np.asarray(all_best_wd_smoother_final))

print("dynamics final")
print(all_best_wd_dynamics_final)
np.save(os.path.join(PATH_TO_STORAGE, 'best_wd_dynamics_final'), np.asarray(all_best_wd_dynamics_final))
