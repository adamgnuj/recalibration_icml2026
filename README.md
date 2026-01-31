# recalibration_icml2026
Experiment code for the ICML'2026 submission _Nonparametric Distribution Regression Re-calibration_.


The structure of this repository is the following:
```
recalbration_icml2026
├── Calibration
│   ├── src
│   ├── Project.toml
│   ├── Manifest.toml
├── experiment
│   ├── data
│   ├── models
│   ├── recalib_models
│   ├── eval
│   ├── logs
├── README.md
├── Project.toml
├── Manifest.toml
├── conda_environment.yml
```

The directory [Calibration](Calibration/) contains an
implementation of the proposed nonparametric recalibration algorithm in the form of a [Julia](https://julialang.org/) package.

The directory [experiment](experiment/) contains the
code for 
1. obtainig datasets ([data](experiment/data/))
2. train base models ([models](experiment/models/))
3. running re-calibration algorithms ([recalib_models](experiment/recalib_models/)), 
4. and evaluate the results ([eval](experiment/eval/)).

To reproduce our benchmark results, begin by installing the required environments (see instructions below). Next, follow the steps in the individual README files, following the links presented in the numbered list above.

## Environment installations
### Julia environment
1. Install the Julia version `1.11.5` from [julialang.org](https://julialang.org/downloads/).
2. Istantiate the required julia environment via
    - starting a julia REPL in this directory (`recalibration_icml2026`), 
    - and execute
        ```
        using Pkg
        Pkg.activate(".")  
        Pkg.instantiate()
        ```

### Conda environment 
To create and activate the required conda environment, execute the following lines:
```
conda env create -f conda_environment.yml
conda activate run_exp
```