# Base Models

Find the implementations of the base models (i.e., models to be recalibrated) in the subdirectories of this directory.

In each subdirectory there is a file `run_model.py` which
executes model fitting and inference using the given model and given dataset.
The results are exported to `<model name>/exported_predictions/`.

The file [run.py](run.py) in this directory runs
a given model on all the datasets defined under [../data/](../data/).

## Environment and codebase installations

Some of the model implementations are based on
the experiment code in [this](https://github.com/proto-n/torch-naut/blob/iclr2025/) repository.

Please run:
`git clone --branch iclr2025 https://github.com/proto-n/torch-naut.git mdn/
`

To have all `conda` environments ready please run the
following commands:
```
conda env create -f environment_pytorch_gpu.yml
conda env create -f environment_r_env.yml
```


