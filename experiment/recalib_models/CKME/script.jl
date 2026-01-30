# run with julia --project=../../ script.jl 
@info "env:" Base.active_project()

using DataFrames, CSV, Distributions, CUDA
using Calibration

if "GPU_DEVICE" in keys(ENV)
    GPU_DEVICE = parse(Int, ENV["GPU_DEVICE"])
    CUDA.device!(GPU_DEVICE)
else
    GPU_DEVICE = CUDA.device()
end

@info "gpu:" Base.active_project() GPU_DEVICE ARGS


begin
    args_parsed = NamedTuple{(
        :X_train, :X_validation, :X_test, 
        :y_train, :y_validation, 
        :preds_train, :preds_validation, :preds_test,
        :preds_format, :recal_preds_test_out)}(ARGS)


    y_train = CSV.read(args_parsed.y_train, DataFrame) |> Matrix |> vec .|> Float64
    y_valid = CSV.read(args_parsed.y_validation, DataFrame) |> Matrix |> vec .|> Float64


    PREDS_FORMAT = first(readlines(args_parsed.preds_format))
    @info "parsed:" PREDS_FORMAT

    if PREDS_FORMAT == "WeightedFixSampleEmpirical"
        preds_W_train = CSV.read(args_parsed.preds_train, DataFrame) |> Matrix
        preds_W_valid = CSV.read(args_parsed.preds_validation, DataFrame) |> Matrix
        preds_W_test = CSV.read(args_parsed.preds_test, DataFrame) |> Matrix

        preds_repr_train = (preds_W_train, y_train)
        preds_repr_valid = (preds_W_valid, y_train)
        preds_repr_test = (preds_W_test, y_train)

        preds_representation = Calibration.WeightedFixSampleEmpirical(Laplace())
    elseif PREDS_FORMAT == "gaussianmixture"
        preds_train = CSV.read(args_parsed.preds_train, DataFrame) |> Matrix
        preds_valid = CSV.read(args_parsed.preds_validation, DataFrame) |> Matrix
        preds_test = CSV.read(args_parsed.preds_test, DataFrame) |> Matrix

        nc = size(preds_test, 2) ÷ 3
        _n(P) = P ./ sum(P, dims=2)
        preds_repr_train = (preds_train[:, 1:nc], preds_train[:, nc+1:2nc], _n(preds_train[:, 2nc+1:end]))
        preds_repr_valid = (preds_valid[:, 1:nc], preds_valid[:, nc+1:2nc], _n(preds_valid[:, 2nc+1:end]))
        preds_repr_test = (preds_test[:, 1:nc], preds_test[:, nc+1:2nc], _n(preds_test[:, 2nc+1:end]))

        
        
        preds_representation = Calibration.GaussianMixture(Laplace())

    else
        error("Not implemented PREDS_FORMAT: ", PREDS_FORMAT)
    end

    conf = Calibration.ReCalibrationConfiguration(
        "test",
        args_parsed.recal_preds_test_out,
        y_train,
        preds_repr_train,
        y_valid,
        preds_repr_valid,
        preds_repr_test,
        # logrange(10^-18, 1, 50), # lt grid
        42, # cv mask seed
        preds_representation,
        Calibration.MedianHeuristic(Laplace()), # obs repr
        # Calibration.KernelDensityOperator.MinMaxGrid(10_000, 1.3) # n_z, d
    )

    Calibration.run_recalibration(conf)
end