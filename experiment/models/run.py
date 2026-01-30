import argparse

parser = argparse.ArgumentParser(
    description="Runs a model on all the datasets.")
parser.add_argument("model_name", type=str, help="Path to the folder containing `model_run.py`")
args = parser.parse_args()


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
MODEL_DIR = os.path.join(SCRIPT_DIR, args.model_name)
SCRIPT_RUN_PATH = os.path.join(MODEL_DIR, "run_model.py")

# Set up logger
logger = logging.getLogger("Model_runner")
logger.setLevel(logging.INFO)
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(SCRIPT_DIR, f"../logs/model_{args.model_name}_runner_{log_time}.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Walk datasets and `experiment/models/drf/exported_predictions/` 
# and run model if it doesn't already has a result. 

print(f"Follow progress at {log_file}")

logger.info(f"------- start run for {args.model_name} -------")
logger.info(f"{SCRIPT_RUN_PATH = }")


export_path = Path(MODEL_DIR).joinpath("exported_predictions")
for (data_repo, data_set, split) in data_split_iterator.Loader(return_arrays = False):
    export_repo = export_path.joinpath(data_repo.name)
    if not export_repo.exists():
        logger.info(f"Create {export_repo = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_repo])
    export_set = export_repo.joinpath(data_set.name)
    if not export_set.exists():
        logger.info(f"Create {export_set = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_set])
    export_split = export_set.joinpath(split.name)
    if not export_split.exists():
        logger.info(f"Create {export_split = }, since it doesn't exist yet.")
        subprocess.run(["mkdir", export_split])

    export_pred_test = export_split.joinpath("predictions_test.csv")
    export_pred_valid = export_split.joinpath("predictions_valid.csv")
    export_pred_train = export_split.joinpath("predictions_train.csv")
    if all(map(lambda f : f.exists(), [export_pred_test, export_pred_valid, export_pred_train])):
        logger.debug(f"skip {export_split}, it's already done")
    else:
        logger.info(f"run {args.model_name}: {export_split}")
        # run the model code
        result = subprocess.run([
            "conda", "run", "-n", "run_exp", "python", SCRIPT_RUN_PATH, 
            split.joinpath("X_train.csv"), 
            split.joinpath("X_validation.csv"), 
            split.joinpath("X_test.csv"),
            split.joinpath("y_train.csv"), 
            split.joinpath("y_validation.csv"), 
            export_pred_train, export_pred_valid, export_pred_test,
            log_file
            ], # cmd
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

