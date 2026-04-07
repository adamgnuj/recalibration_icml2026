using ArgParse, HypothesisTests, JLD2, ProgressMeter, DataFrames
using CSV, Distributions, Logging, Dates, StatsBase, Calibration
using Base.Threads

using Plots, CUDA, LinearAlgebra
include("compare_results.jl")

"return (preds_Q_cal, preds_Q_test, obs_calib, obs_test)"
function _handle_format(format::String, pred_dir::String, obs_test_file::String, obs_train_file::String)
    obs_test = CSV.read(obs_test_file, DataFrame)[:,1] .|> Float64
    obs_train = CSV.read(obs_train_file, DataFrame)[:,1] .|> Float64
    obs_calib = CSV.read(joinpath(first(splitdir(obs_train_file)), "y_validation.csv"), DataFrame)[:,1] .|> Float64

    if format == "WeightedFixSampleEmpirical"
        pred_cal_fn = joinpath(pred_dir, "predictions_valid.csv")
        pred_test_fn = joinpath(pred_dir, "predictions_test.csv")
        preds_W_valid = CSV.read(pred_cal_fn, DataFrame) |> Matrix
        preds_W_test = CSV.read(pred_test_fn, DataFrame) |> Matrix

        preds_Q_cal = [Calibration.Empirical(obs_train, w) for w in eachrow(preds_W_valid)]
        preds_Q_test = [Calibration.Empirical(obs_train, w) for w in eachrow(preds_W_test)]
        return preds_Q_cal, preds_Q_test, obs_calib, obs_test

    elseif format == "gaussianmixture"
        pred_cal_fn = joinpath(pred_dir, "predictions_valid.csv")
        pred_test_fn = joinpath(pred_dir, "predictions_test.csv")
        preds_test = CSV.read(pred_test_fn, DataFrame) |> Matrix
        preds_cal = CSV.read(pred_cal_fn, DataFrame) |> Matrix

        # process test set predictions
        nc = size(preds_test, 2) ÷ 3
        preds_repr_test = (preds_test[:, 1:nc], preds_test[:, nc+1:2nc], preds_test[:, 2nc+1:end])
        L, V, P = preds_repr_test
        P ./= sum(P, dims = 2)
        preds_Q_test = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]

        # proces calibration set predictions
        nc = size(preds_cal, 2) ÷ 3
        preds_repr_cal = (preds_cal[:, 1:nc], preds_cal[:, nc+1:2nc], preds_cal[:, 2nc+1:end])
        L, V, P = preds_repr_cal
        P ./= sum(P, dims = 2)
        preds_Q_cal = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]

        return preds_Q_cal, preds_Q_test, obs_calib, obs_test
    else
        error("not implemented format: ", format)
    end
end


function run_cp_procedure(format, pred_dir, obs_test_file, obs_train_file, α = 0.05)
    preds_Q_cal, preds_Q_test, obs_calib, obs_test = _handle_format(
        format, pred_dir, obs_test_file, obs_train_file)

    ψ(x::Real) = abs(x - 1/2)
    U_cal = cdf.(preds_Q_cal, obs_calib)
    V_cal = ψ.(U_cal)
    Q_V_cal = Calibration.Empirical(V_cal)
    Q̂_cal = quantile(Q_V_cal, min((1-α) * (1 + 1 / length(obs_calib)), 1))

    # find intervals:
    cp_intervals = zeros(length(preds_Q_test), 2)
    for (i, Q_pred) in enumerate(preds_Q_test)
        # ψ(x) = |x - 1/2|
        try
            cp_intervals[i, 1] = quantile(Q_pred, 1/2 - Q̂_cal)
        catch e # 0 and 1 quantiles somethimes throw an error:
            if isa(e, ArgumentError)
                cp_intervals[i, 1] = minimum(Q_pred)
            else
                throw(e)
            end
        end

        try
            cp_intervals[i, 2] = quantile(Q_pred, 1/2 + Q̂_cal)
        catch e # 0 and 1 quantiles somethimes throw an error:
            if isa(e, ArgumentError)
                cp_intervals[i, 2] = maximum(Q_pred)
            else
                throw(e)
            end
        end
    end
    Z_PIT = cdf.(preds_Q_test, obs_test)
    cp_intervals, obs_test, Z_PIT
end




if abspath(PROGRAM_FILE) == @__FILE__
    _argparse_setting = ArgParseSettings()
    @add_arg_table! _argparse_setting begin
        "EXPORTED_PREDS_DIR"
            help = "path to the `exported_predictions` folder under the given model's folder"
            required = true

        "out_file"
            help = "filepath to save the result as a .csv"
            required = true

        "--gpu", "-g"
            help = "optinal gpu index to use"
            arg_type = Int
            default = 0
    end

    pargs = parse_args(ARGS, _argparse_setting)
    CUDA.device!(pargs["gpu"]) 
    @info "run CP on model $(pargs["EXPORTED_PREDS_DIR"])" CUDA.device()

    EVAL_DIR = @__DIR__ 
    EVAL_DIR *= "/../eval"
    include(joinpath(EVAL_DIR, "experiment_walker.jl"))

    file_list = data_path_collecter(pargs["EXPORTED_PREDS_DIR"])
    # file_list = file_list[1:5]
    N = length(file_list)
    results = Vector{Tuple{String, String, Int, Float64, Float64, Float64, Float64}}(undef, N)

    error_log = []
    err_lock = ReentrantLock()
    @showprogress desc = "run CP" showspeed = true @threads for i in 1:N
    # @showprogress desc = "run CP" showspeed = true for i in 1:N
        f = file_list[i]
        try
            cp_intervals, obs_test, Z_PIT = run_cp_procedure(f.PRED_FORMAT, f.pred_dir, 
                f.obs_test_file, f.obs_train_file)

            res = eval_interval_result(cp_intervals, obs_test, Z_PIT)

            results[i] = (f.repo, f.dataset, parse(Int, f.split_idx), res...)
        catch e
            lock(err_lock) do
                # We save the index 'i', a useful identifier like the repo name, and the error 'e'
                push!(error_log, (i, f.dataset, f.split_idx, e))
            end
        end
    end
    if length(error_log) > 0
        @error "there were errors" length(error_log)
        for e in error_log
            @warn "error:" e
        end
    else
        @info "no errors"
    end
    df = DataFrame(results, [:repo, :dataset, :split, :coverage, :average_int_len, :median_int_len, :dcov])
    CSV.write(pargs["out_file"], df)
    @info "done, results saved to $(pargs["out_file"])"
end