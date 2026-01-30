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

args = parser.parse_args()


from pathlib import Path
import subprocess


import logging, os

# Set up logger
logger = logging.getLogger("RecalibrationModel_GPBETA_runner")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FILE = os.path.join(SCRIPT_DIR, "script.py")
# GPB_LIB_PATH = os.path.join(SCRIPT_DIR, "DistCal")

logger.info("starting GPBETA recalibration script")

result = subprocess.run([
    "conda", "run", "-n", "distcal",
    "python", SCRIPT_FILE,
    args.X_train, args.X_validation, args.X_test,
    args.y_train, args.y_validation, 
    args.preds_train, args.preds_validation, args.preds_test,
    args.preds_format,
    args.recal_preds_test_out,
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


