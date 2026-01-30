import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set up logger
logger = logging.getLogger("DataSplitIterator")
logger.setLevel(logging.INFO)
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(SCRIPT_DIR, f"../logs/data_split_iterator_{log_time}.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Loader:
    """
    Return path iterator for all data and splits, yielding `(data_repo, data_set, split)`.
    If `return_arrays` then also load the datasets as `np` array and return 
    `(X_train, X_validation, X_test, y_train, y_validation, y_test)` as well.
    """
    def __init__(self, return_arrays = True):
        self._path = Path(SCRIPT_DIR)
        self.return_arrays = return_arrays

    def __iter__(self):
        return self._iterate_on_splits()

    def _iterate_on_splits(self):
        logger.info(f"---------- start data iteration --------")
        for data_repo in self._path.iterdir():
            if (not data_repo.is_dir()) or data_repo.name.startswith('.') or data_repo.name.startswith('_'):
                continue  # skip hidden and non folders
            logger.info(f"data_repo.name = {data_repo.name}")
            exported_splits = data_repo.joinpath('exported_splits')
            if not exported_splits.exists():
                logger.warning(f"exported_splits not found in {data_repo}")
                continue
            for data_set in exported_splits.iterdir():
                logger.info(f"    data_set.name = {data_set.name}")
                for split in sorted(data_set.iterdir(), key=lambda d: int(d.name.removeprefix("split_"))):
                    logger.info(f"        split = {split.name}")
                    if self.return_arrays:
                        X_train = pd.read_csv(split.joinpath("X_train.csv")).to_numpy()
                        X_validation = pd.read_csv(split.joinpath("X_validation.csv")).to_numpy()
                        X_test = pd.read_csv(split.joinpath("X_test.csv")).to_numpy()
                        y_train = pd.read_csv(split.joinpath("y_train.csv")).to_numpy()
                        y_validation = pd.read_csv(split.joinpath("y_validation.csv")).to_numpy()
                        y_test = pd.read_csv(split.joinpath("y_test.csv")).to_numpy()
                        arrays = (X_train, X_validation, X_test, y_train, y_validation, y_test)
                        yield (data_repo, data_set, split), arrays
                    else:
                        yield (data_repo, data_set, split)
