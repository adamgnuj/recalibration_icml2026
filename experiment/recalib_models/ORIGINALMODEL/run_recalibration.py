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
parser.add_argument("log_file", type=str)
parser.add_argument("--original_model_runner", type=str, help="original model's runner script path e.g. `models/drf/run_model.py`", default=None)

args = parser.parse_args()


from pathlib import Path
import subprocess


import logging, os

# Set up logger
logger = logging.getLogger("RecalibrationModel_ORIGINAL_runner")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FILE = os.path.join(SCRIPT_DIR, "script.py")

# ORIGINAL_MODEL_RUNNER = "/home/jungadam/idms/spring/experiment/models/drf/run_model.py" # TODO
ORIGINAL_MODEL_RUNNER = args.original_model_runner

# merge X_train X_validation and y

from pathlib import Path
import pandas as pd

path_X_train_and_val = Path(args.X_test).parent.joinpath("X_train_and_validation.csv")
path_y_train_and_val = Path(args.X_test).parent.joinpath("y_train_and_validation.csv")

if path_X_train_and_val.exists() and path_y_train_and_val.exists():
    print("both already exists:", path_X_train_and_val)
else:
    print("doesn't exist:", path_X_train_and_val)
    X_train = pd.read_csv(args.X_train)
    X_validation = pd.read_csv(args.X_validation)
    y_train = pd.read_csv(args.y_train)
    y_validation = pd.read_csv(args.y_validation)

    combined_df_X = pd.concat([X_train, X_validation], axis=0, ignore_index=True)
    combined_df_y = pd.concat([y_train, y_validation], axis=0, ignore_index=True)
    combined_df_X.to_csv(path_X_train_and_val, index=False)
    combined_df_y.to_csv(path_y_train_and_val, index=False)
    print("done concat")




logger.info(f"starting ORIGINAL recalibration (running on ) {path_y_train_and_val}")

result = subprocess.run([
    "conda", "run", "-n", "run_exp",
    "python", ORIGINAL_MODEL_RUNNER,
    path_X_train_and_val, 
    args.X_validation,
    args.X_test,
    path_y_train_and_val,
    args.y_validation,
    os.path.join(args.recal_preds_test_out, "preds_train.csv"),
    os.path.join(args.recal_preds_test_out, "preds_valid.csv"),
    os.path.join(args.recal_preds_test_out, "preds_test.csv"),
    args.log_file
    ], # cmd
    capture_output=True,           # capture both stdout and stderr
    text=True                      # decode bytes to str
)


if result.returncode == 0:
    logger.info("Return code: %s", result.returncode)
else:
    logger.error("Return code: %s", result.returncode)

logger.info("STDOUT:\n%s", result.stdout)
if result.stderr:
    logger.warning("STDERR:\n%s", result.stderr)


