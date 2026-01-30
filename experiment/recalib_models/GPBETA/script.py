import argparse

parser = argparse.ArgumentParser(
    description="Runs recalibration model GPBETA on the given split.")
parser.add_argument("X_train", type=str)
parser.add_argument("X_validation", type=str)
parser.add_argument("X_test", type=str)
parser.add_argument("y_train", type=str)
parser.add_argument("y_validation", type=str, help="original data")
parser.add_argument("preds_train", type=str, help="predictions to be recalibrated")
parser.add_argument("preds_validation", type=str)
parser.add_argument("preds_test", type=str)
parser.add_argument("preds_format", type=str, help="path to .txt file describing the format of the predictions")
parser.add_argument("recal_preds_test_out", type=str, help="path to save the recalibrated test set predictions")

args = parser.parse_args()


import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "DistCal"))

# fix TF cuda device
import os
if "GPU_DEVICE" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_DEVICE"]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

print("Using GPU_DEVICE: " + os.environ["CUDA_VISIBLE_DEVICES"])
# Limit to 4 cpu threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# ------------ code partially from DistCal/example.ipynb -----------

import numpy
import scipy.stats
import datetime
import matplotlib.pyplot
from exp import load_data
import sklearn.model_selection
import utils
from GP_Beta_cal import GP_Beta
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

with open(args.preds_format, "r") as f:
    PREDS_FORMAT = f.readline().strip()

y_train = numpy.loadtxt(args.y_train, skiprows = 1)
y_val = numpy.loadtxt(args.y_validation, skiprows = 1)

if PREDS_FORMAT == "WeightedFixSampleEmpirical":
    W_pred_val = numpy.loadtxt(args.preds_validation, skiprows = 1, delimiter = ",")
    W_pred_test = numpy.loadtxt(args.preds_test, skiprows = 1, delimiter = ",")
    

    pred_mean_val = W_pred_val @ y_train
    pred_std_val = np.sqrt(W_pred_val @ (y_train ** 2) - pred_mean_val ** 2)

    pred_mean_test = W_pred_test @ y_train
    pred_std_test = np.sqrt(W_pred_test @ (y_train ** 2) - pred_mean_test ** 2)

    mu_cal, sigma_cal = pred_mean_val.reshape(-1, 1), pred_std_val.reshape(-1, 1) # use variable names from original implementation
    mu_base, sigma_base = pred_mean_test.reshape(-1, 1), pred_std_test.reshape(-1, 1)

elif PREDS_FORMAT == "gaussianmixture":
    pred_val = numpy.loadtxt(args.preds_validation, skiprows = 1, delimiter = ",")
    pred_test = numpy.loadtxt(args.preds_test, skiprows = 1, delimiter = ",")
    
    nc = pred_val.shape[1] // 3
    loc_val, var_val, pi_val = pred_val[:, :nc], pred_val[:, nc:2*nc], pred_val[:, 2*nc:]
    loc_test, var_test, pi_test = pred_test[:, :nc], pred_test[:, nc:2*nc], pred_test[:, 2*nc:]
    
    pi_val = pi_val / np.sum(pi_val, axis = 1, keepdims=True)
    pi_test = pi_test / np.sum(pi_test, axis = 1, keepdims=True)

    pred_mean_val = np.sum(loc_val * pi_val, axis = 1)
    pred_var_val = np.sum(var_val * pi_val, axis = 1) + (np.sum((loc_val ** 2) * pi_val, axis = 1) - np.sum(loc_val * pi_val, axis = 1) ** 2)
    pred_std_val = np.sqrt(pred_var_val)

    pred_mean_test = np.sum(loc_test * pi_test, axis = 1)
    pred_var_test = np.sum(var_test * pi_test, axis = 1) + (np.sum((loc_test ** 2) * pi_test, axis = 1) - np.sum(loc_test * pi_test, axis = 1) ** 2)
    pred_std_test = np.sqrt(pred_var_test)

    print(f"debug shapes: {pred_mean_val.shape}, {pred_mean_test.shape}, {y_val.shape}")

    mu_cal, sigma_cal = pred_mean_val.reshape(-1, 1), pred_std_val.reshape(-1, 1) # use variable names from original implementation
    mu_base, sigma_base = pred_mean_test.reshape(-1, 1), pred_std_test.reshape(-1, 1)

else:
    raise ValueError(f"Not implemented pred format: {args.preds_format}")

# ---------------------------- fit GPB calibrator ---------------------------------
n_u = 8 # number of induced points
print('training size: ' + str(len(y_val)))
print('induced points: ' + str(n_u))
GP_Beta_mdl = GP_Beta()
start_time = datetime.datetime.now()
GP_Beta_mdl.fit(y_val.reshape(-1, 1), mu_cal, sigma_cal, n_u=n_u, 
                plot_loss=False, print_info=False)
end_time = datetime.datetime.now()
print('training time: ' + str((end_time - start_time).total_seconds()) + ' seconds')

# ---------------------------- pred with GPB calibrator ---------------------------
n_t_test = 1024

t_list_test = numpy.linspace(numpy.min(mu_base) - 16.0 * numpy.max(sigma_base),
                            numpy.max(mu_base) + 16.0 * numpy.max(sigma_base),
                            n_t_test).reshape(1, -1)

s_gp, q_gp = GP_Beta_mdl.predict(t_list_test, mu_base, sigma_base)

# --------------------------- save results ---------------------------------
# pandas save t_list_test, s_gp, q_gp
pd.DataFrame(s_gp).to_csv(os.path.join(args.recal_preds_test_out, "test_recal_s.csv"), index=False)
pd.DataFrame(q_gp).to_csv(os.path.join(args.recal_preds_test_out, "test_recal_q.csv"), index=False)
pd.DataFrame(t_list_test.T).to_csv(os.path.join(args.recal_preds_test_out, "test_recal_t.csv"), index=False)
print(f"Recalibrated predictions saved to {args.recal_preds_test_out}")