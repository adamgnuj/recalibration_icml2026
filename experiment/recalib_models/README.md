# Recalibration algorithms

Find the code in the subdirectories, needed to run the recalibration algorithms.

Note that we have the special "recalibrtaion" algorithm
[ORIGINALMODEL](ORIGINALMODEL), which just uses
the base model trained on the bigger (training + calibration) dataset, without any actual recalibration.

In each subdirectory there is a file `run_recalibration.py` which
executes the given _recalibration algorithm_ on the given _base model_ and a given dataset, to produce recalibrated predictions on the test set.
The results are exported to `<recalibration algorithm>/exported_recalibrated_models/<base model>/`.

The file [run.py](run.py) in this directory runs
a given recalibration model on a given base model, on all the datasets defined under [../data/](../data/).


## Installations
Please run
`git clone https://github.com/Srceh/DistCal.git GPBETA/
`
to have the neccessary codebase for running 
the code of [Distribution Calibration for Regression (ICML'2019)](https://github.com/Srceh/DistCal)