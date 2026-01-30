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
parser.add_argument("log_file", type=str)

args = parser.parse_args()


from pathlib import Path
import subprocess


import logging, os

# Set up logger
logger = logging.getLogger("Model_bnn_runner")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(args.log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PY_PATH = os.path.join(SCRIPT_DIR, "script.py")

result = subprocess.run([
    "conda", "run", "-n", "pytorch_gpu", "python", SCRIPT_PY_PATH, 
    args.X_train,
    args.X_validation,
    args.X_test,
    args.y_train, 
    args.y_validation, 
    args.preds_train, args.preds_validation, args.preds_test, 
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


