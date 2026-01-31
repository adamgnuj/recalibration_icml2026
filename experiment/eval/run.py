import argparse

parser = argparse.ArgumentParser(
    description="Runs all evaluation code on a given model.")
parser.add_argument("model_exported_preds", type=str, help="Path to model's exported preds dir")
parser.add_argument("save_file_preamble", type=str, help="like `bnn_recal_PIT`)")
parser.add_argument("--gpu", type=int, help="which gpu to use", default=0)

args = parser.parse_args()
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess, os, logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = Path(SCRIPT_DIR).parent.as_posix()


logger = logging.getLogger("eval_runner")
logger.setLevel(logging.INFO)
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(EXPERIMENT_DIR, "logs", f"eval_{args.save_file_preamble}_runner_{log_time}.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

print(f"Follow progress at {log_file}")



JULIA_EVAL_CRPS_SCRIPT = os.path.join(SCRIPT_DIR, "eval_CRPS.jl")
JULIA_RUN_PIT_TEST_SCRIPT = os.path.join(SCRIPT_DIR, "run_PIT_test.jl")
# JULIA_RUN_HSIC_TEST_SCRIPT = os.path.join(SCRIPT_DIR, "run_HSIC_test.jl")
JULIA_RUN_SKCE_TEST_SCRIPT = os.path.join(SCRIPT_DIR, "run_SKCE_test.jl")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

scripts = [JULIA_EVAL_CRPS_SCRIPT, JULIA_RUN_PIT_TEST_SCRIPT, JULIA_RUN_SKCE_TEST_SCRIPT]
eval_names = ["CRPS", "PIT", "SKCE"]

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = ""
env["JULIA_PROJECT"] = "<absolute path to julia project file>"
env["OPENBLAS_NUM_THREADS"] = "5"
env["GPU_DEVICE"] = f"{args.gpu}"


for script, eval_name in zip(scripts, eval_names):
    logger.info(f"Running evaluation: {eval_name}")
    save_file = os.path.join(RESULTS_DIR, f"{args.save_file_preamble}_{eval_name}.csv")
    if os.path.exists(save_file):
        logger.info(f"Evaluation {eval_name} already done at {save_file}, skipping.")
        continue
    else:
        logger.info(f"start evaluations on: {args.save_file_preamble}")

    cmd = ["julia", script, args.model_exported_preds, save_file]
    result = subprocess.run(cmd, capture_output=True, text=True, env = env)
    if result.returncode == 0:
        logger.info(f"result returncode: {result.returncode}")
    else:
        logger.error(f"result returncode: {result.returncode}")

    if result.stdout:
        logger.info(f"result stdout: {result.stdout}")
        
    if result.stderr:
        logger.warning(f"result stderr: {result.stderr}")