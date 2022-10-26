import numpy as np
from matplotlib import pyplot as plt
import os

PATH_TO_DATA = "/home/philippe/GRU_Playground/DGM_nlls/nll_predictions/200"

no_int_pre_nlls = []
no_int_final_nlls = []
int_pre_nlls = []
int_final_nlls = []

for seed in 5*np.arange(1, 9, 1):
    path_to_current_files = os.path.join(PATH_TO_DATA, str(seed))
    int_final_nlls.append(np.loadtxt(os.path.join(path_to_current_files, "final_final_nlls.csv")))
    int_pre_nlls.append(np.loadtxt(os.path.join(path_to_current_files, "pre_pre_nlls.csv")))
    no_int_final_nlls.append(np.loadtxt(os.path.join(path_to_current_files, "no_int_final.csv")))
    no_int_pre_nlls.append(np.loadtxt(os.path.join(path_to_current_files, "no_int_pre.csv")))

no_int_pre_nlls = np.asarray(no_int_pre_nlls)
no_int_final_nlls = np.asarray(no_int_final_nlls)
int_pre_nlls = np.asarray(int_pre_nlls)
int_final_nlls = np.asarray(int_final_nlls)

print(f"nip: {np.mean(no_int_pre_nlls)} +- {np.std(no_int_pre_nlls)}")
print(f"nif: {np.mean(no_int_final_nlls)} +- {np.std(no_int_final_nlls)}")
print(f"wip: {np.mean(int_pre_nlls)} +- {np.std(int_pre_nlls)}")
print(f"wif: {np.mean(int_final_nlls)} +- {np.std(int_final_nlls)}")

