import argparse

parser = argparse.ArgumentParser(
    description="Runs a recalibration of a given model model on all the datasets.")
# parser.add_argument("model_name", type=str, help="Original model's name , like `drf`")
# parser.add_argument("recalibration_name", type=str, help="Recalibration model's name like `CKMEKDO`")
parser.add_argument("model_exported_preds", type=str, help="Path to original model's exported preds dir")
parser.add_argument("recal_model_exported_preds", type=str, help="Path to recalibration model's exported preds dir")
parser.add_argument("recal_model_name", type=str, help="name of recalibration model's dir (as in the `recalib_models` folder)")
parser.add_argument("--original_model_runner", type=str, help="original model's runner script path e.g. `models/drf/run_model.py`", default=None)



args = parser.parse_args()

if args.original_model_runner:
    print(f"Using original model runner script: {args.original_model_runner}")

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

import sys
sys.path.append("../data/")
import data_split_iterator

import logging, os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = Path(SCRIPT_DIR).parent.as_posix()
# MODEL_DIR = os.path.join(EXPERIMENT_DIR, "models", args.model_name)
RECALIB_DIR = os.path.join(EXPERIMENT_DIR, "recalib_models", args.recal_model_name)
SCRIPT_RUN_PATH = os.path.join(RECALIB_DIR, "run_recalibration.py")
RECAL_PREDS_EXPORT_PATH = args.recal_model_exported_preds
ORIGINAL_PREDS_PATH = args.model_exported_preds

# Set up logger
logger = logging.getLogger("recalibration_runner")
logger.setLevel(logging.INFO)
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(EXPERIMENT_DIR, "logs", f"recal_{args.recal_model_name}_runner_{log_time}.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Walk datasets and `experiment/models/drf/exported_predictions/` 
# and run model if it doesn't already has a result. 

print(f"Follow progress at {log_file}")

logger.info(f"------- start run for {args.recal_model_name} on {ORIGINAL_PREDS_PATH} -------")
logger.info(f"{SCRIPT_RUN_PATH = }")

# export_path = Path(RECALIB_DIR).joinpath("exported_predictions")
export_path = Path(RECAL_PREDS_EXPORT_PATH)
# model_preds_path = Path(MODEL_DIR).joinpath("exported_predictions")
model_preds_path = Path(ORIGINAL_PREDS_PATH)
for (data_repo, data_set, split) in data_split_iterator.Loader(return_arrays = False):
    export_repo = export_path.joinpath(data_repo.name)
    model_preds_repo = model_preds_path.joinpath(data_repo.name)
    if not export_repo.exists():
        logger.info(f"Create {export_repo = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_repo])
    export_set = export_repo.joinpath(data_set.name)
    model_preds_set = model_preds_repo.joinpath(data_set.name)
    if not export_set.exists():
        logger.info(f"Create {export_set = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_set])
    export_split = export_set.joinpath(split.name)
    model_preds_split = model_preds_set.joinpath(split.name)
    if not export_split.exists():
        logger.info(f"Create {export_split = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_split])

    # export_pred_test = export_split.joinpath("predictions_test.csv")
    # export_pred_valid = export_split.joinpath("predictions_valid.csv")
    # export_pred_train = export_split.joinpath("predictions_train.csv")
    
    # if export_split.joinpath("test_recalibration.jld2").exists():
    if any(export_split.iterdir()):
        logger.info(f"skip {export_split}, it's already has value(s): {list(export_split.iterdir())}")
    else:
        logger.info(f"run {args.recal_model_name} on {ORIGINAL_PREDS_PATH}: {export_split}")
        # run the model code
        result = subprocess.run([
            "conda", "run", "-n", "run_exp", "python", SCRIPT_RUN_PATH, 
            split.joinpath("X_train.csv"), 
            split.joinpath("X_validation.csv"), 
            split.joinpath("X_test.csv"),
            split.joinpath("y_train.csv"), 
            split.joinpath("y_validation.csv"), 
            model_preds_split.joinpath("predictions_train.csv"), 
            model_preds_split.joinpath("predictions_valid.csv"), 
            model_preds_split.joinpath("predictions_test.csv"),
            model_preds_path.joinpath("format.txt"),
            export_split,
            log_file
            ] + (["--original_model_runner", args.original_model_runner] if args.original_model_runner else []), # cmd
            capture_output=True,           # capture both stdout and stderr
            text=True                      # decode bytes to str
        )
        if result.returncode == 0:
            logger.info(f"result returncode: {result.returncode}")
        else:
            logger.error(f"result returncode: {result.returncode}")

        if result.stdout:
            logger.info(f"result stdout: {result.stdout}")
            
        if result.stderr:
            logger.warning(f"result stderr: {result.stderr}")

