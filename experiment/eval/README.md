# run evaluation routines
This directorty contatatins the implementation
of followings:
1. CRPS loss evaluation ([eval_CRPS.jl](eval_CRPS.jl))
2. PIT calibration hypothesis test ([run_PIT_test.jl](run_PIT_test.jl))
3. Auto-calibration hypothesis test ([run_SKCE_test.jl](run_SKCE_test.jl))

The file [run.py](run.py) executes the three evaluation metrics on the results of a given model + recalibration method, and exports the results under the [results/](results/) folder.

Modify line `45` of [run.py](run.py)
```
env["JULIA_PROJECT"] = "<absolute path to julia project file>"
```
to execute the scripts with the appropriate julia environment.
For example: `env["JULIA_PROJECT"] = "/home/<user>/.../recalibration_icml2025"`


The file [experiment_walker.jl](experiment_walker.jl)
is just a utility tool, to iterate over all exported dataset splits and
predictions in `julia`.