import argparse

parser = argparse.ArgumentParser(
    description="Runs model bnn on the given split.")

parser.add_argument("X_train", type=str)
parser.add_argument("X_validation", type=str)
parser.add_argument("X_test", type=str)
parser.add_argument("y_train", type=str)
parser.add_argument("y_validation", type=str)
parser.add_argument("preds_train", type=str)
parser.add_argument("preds_validation", type=str)
parser.add_argument("preds_test", type=str)
# default arguments added from run_mdn.py 
parser.add_argument('--n_components', type=int, default=1, help='Number of components') 
parser.add_argument('--device', type=str, default='auto', help='Device to use (default: auto)')

args = parser.parse_args()

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "mdn", "torch-naut")) # add torch-naut to path

## --------------------------- code from torch-naut/uci/run_mdn_bnn.py ---------------------

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import argparse
import os
import torch
# from tqdm.auto import tqdm
from glob import glob
import lib.mdn as mdn
import lib.utils as utils
import torch.nn as nn
from lib import mdn
from lib.bnn import BayesianLayer
import time

start_time = time.time()

class MDN(nn.Module):
    def __init__(self, input_dim, n_components, min_std=0):
        super(MDN, self).__init__()
        self.n_components = n_components
        self.min_std = min_std
        self.layers = nn.Sequential(
            BayesianLayer(input_dim, 50),
            nn.ReLU(),
            BayesianLayer(50, 50),
            nn.ReLU(),
            BayesianLayer(50, 3 * n_components)
        )

    def forward(self, x):
        x = self.layers(x).view(x.shape[0], self.n_components, 3)
        return mdn.transform_output(x, min_std=self.min_std)


if args.device == 'auto':
    device="cuda:" + str(utils.select_gpu_with_low_usage())
else:
    device=args.device
print("using device", device)


default_params = {
    'l2reg': 1e-6,
    'splits': 20,
}

dataset_param_overrides = {
    'protein-tertiary-structure': {
        'splits': 5,
    },
    'concrete': {
        'l2reg': 1e-4,
    },
    'bostonHousing': {
        'l2reg': 1e-4,
    },
    'energy': {
        'l2reg': 1e-4,
    },
    'yacht': {
        'l2reg': 1e-4,
    },
}



# todo: recover dataset name
dataset_name = os.path.basename(os.path.dirname(os.path.dirname(args.X_train)))
params = dict(default_params, **dataset_param_overrides.get(dataset_name, {}))
print(f"start {dataset_name=}")

np.random.seed(1)

# Load dataset 
X_train =  np.loadtxt(args.X_train, skiprows = 1, delimiter = ",")
y_train = np.loadtxt(args.y_train, skiprows = 1)
X_valset = np.loadtxt(args.X_validation, skiprows = 1, delimiter = ",")
# y_valset = np.loadtxt(args.y_validation, skiprows = 1)
X_testset = np.loadtxt(args.X_test, skiprows = 1, delimiter = ",")

# fix scalar feature X to be a 1D array
if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 1)
    X_testset = X_testset.reshape(-1, 1)
    X_valset = X_valset.reshape(-1, 1)

# the validation set and test set are both considered 
# as test sets in our setup.
X_to_predict = np.vstack((X_train, X_valset, X_testset))

# Split training data into train and validation sets (train (80%), validation (20%))
X_train_original = X_train
y_train_original = y_train
num_training_examples = int(0.8 * X_train.shape[0])
X_validation = X_train[num_training_examples:, :]
y_validation = y_train[num_training_examples:]
X_train = X_train[0:num_training_examples, :]
y_train = y_train[0:num_training_examples]


net = MDN(X_train.shape[1], args.n_components, 0.0).to(device) # set min std to 0.0

net, scalers = mdn.bnn_train(
    net,
    X_train,
    y_train,
    X_validation,
    y_validation,
    batch_size=int(np.sqrt(X_train.shape[0])-5),
    l2reg=params['l2reg'],
    max_patience=50,
    kl_coef=0, # much better scores than with kl
    verbose=True,
)

net.eval()
num_evals = 100 # number of bnn evaluations
eval_pred = [mdn.predict(net, scalers, X_to_predict) for i in range(num_evals)]


# returns a tensor of [num_datapoints x (num_evals x num_components x 3)] with the last
# dims corresponding to 
# (num_evals x num_components x mu, num_evals x num_components x var, num_evals x num_components x pi)
def _get_params(pp):
    mmu, sstd, ppi = [], [], []
    for p in pp:
        p = torch.tensor(p, device=device)
        mixture_coefs = nn.functional.softmax(p[:, :, 2], dim=1)
        mu = p[:, :, 0]
        std = p[:, :, 1]
        
        ppi.append(mixture_coefs)
        mmu.append(mu)
        sstd.append(std)

    mu = torch.cat(mmu, dim=1)
    std = torch.cat(sstd, dim=1)
    mixture_coefs = torch.cat(ppi, dim=1)


    return torch.cat([
        mu,
        std ** 2,
        mixture_coefs
    ], dim=1)
    
params_pred = _get_params(eval_pred)
params_pred_trainset = params_pred[:X_train_original.shape[0], :].cpu().numpy()
params_pred_valset = params_pred[X_train_original.shape[0]:X_train_original.shape[0]+X_valset.shape[0], :].cpu().numpy()
params_pred_testset = params_pred[X_train_original.shape[0]+X_valset.shape[0]:, :].cpu().numpy()
pd.DataFrame(params_pred_trainset).to_csv(args.preds_train, index=False)
pd.DataFrame(params_pred_valset).to_csv(args.preds_validation, index=False)
pd.DataFrame(params_pred_testset).to_csv(args.preds_test, index=False)

print(f"saved done split {args.X_train}")
