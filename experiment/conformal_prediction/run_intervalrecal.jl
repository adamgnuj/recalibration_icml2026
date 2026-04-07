using ArgParse, HypothesisTests, JLD2, ProgressMeter, DataFrames
using CSV, Distributions, Logging, Dates, StatsBase, Calibration
using Base.Threads

using Plots, CUDA, LinearAlgebra
include("compare_results.jl")

"return (preds_Q_test, obs_test)"
function _handle_format(format::String, pred_dir::String, obs_test_file::String, obs_train_file::String)
    obs_test = CSV.read(obs_test_file, DataFrame)[:,1] .|> Float64
    obs_train = CSV.read(obs_train_file, DataFrame)[:,1] .|> Float64
    obs_calib = CSV.read(joinpath(first(splitdir(obs_train_file)), "y_validation.csv"), DataFrame)[:,1] .|> Float64

    if format == "WeightedFixSampleEmpirical"
        # pred_fn = joinpath(pred_dir, first(readdir(pred_dir)))
        fl = readdir(pred_dir)
        if "predictions_test.csv" in fl
            pred_fn = joinpath(pred_dir, "predictions_test.csv")
        elseif "preds_test.csv" in fl
            pred_fn = joinpath(pred_dir, "preds_test.csv")
        else
            @warn "using as prediction" first(fl)
            pred_fn = joinpath(pred_dir, first(fl)) 
            # recalibrated models are only evaluated at the test set
        end
        
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        preds_Q = [Calibration.Empirical(obs_train, w) for w in eachrow(pred_test)]
        return preds_Q, obs_test

    elseif format == "WeightedFixSampleEmpirical_train_and_validation"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir)))
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        n_clipped = size(pred_test, 2) # there is a clip at 20k in the training setup
        preds_Q = [Calibration.Empirical([obs_train; obs_calib][1:n_clipped], w) for w in eachrow(pred_test)]
        return preds_Q, obs_test

    elseif format == "GPB"
        pred_test = Matrix(CSV.read(joinpath(pred_dir, "test_recal_s.csv"), DataFrame))
        pred_test ./= sum(pred_test, dims = 2)
        grid = CSV.read(joinpath(pred_dir, "test_recal_t.csv"), DataFrame) |> Matrix |> vec
        preds_Q = [Calibration.Empirical(grid, w) for w in eachrow(pred_test)]
        return preds_Q, obs_test

    elseif format == "KDO.jld2"
        # r = load(joinpath(pred_dir, "test_recalibration.jld2"))
        # pred_test = r["preds_calib_test_B"][:, length(obs_calib)+1:end]'
        _B = load(joinpath(pred_dir, "test_recalibration.jld2"), "preds_calib_test_B")
        pred_test = _B[:, length(obs_calib)+1:end]'
        preds_Q = [Calibration.Empirical(obs_calib, w) for w in eachrow(pred_test)]
        return preds_Q, obs_test

    elseif format == "gaussianmixture"
        fl = readdir(pred_dir)
        if "predictions_test.csv" in fl
            pred_fn = joinpath(pred_dir, "predictions_test.csv")
        else
            pred_fn = joinpath(pred_dir, first(fl)) 
            # recalibrated models are only evaluated at the test set
        end
        # preds_train = CSV.read(args_parsed.preds_train, DataFrame) |> Matrix
        # preds_valid = CSV.read(args_parsed.preds_validation, DataFrame) |> Matrix
        preds_test = CSV.read(pred_fn, DataFrame) |> Matrix

        nc = size(preds_test, 2) ÷ 3
        preds_repr_test = (preds_test[:, 1:nc], preds_test[:, nc+1:2nc], preds_test[:, 2nc+1:end])
        L, V, P = preds_repr_test
        P ./= sum(P, dims = 2)
        Q = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]
        return Q, obs_test

    elseif format == "uniformweightempirical"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir))) 
        preds_test = CSV.read(pred_fn, DataFrame) |> Matrix
        
        Q_test = [Calibration.Empirical(q) for q in eachrow(preds_test)]
        return Q_test, obs_test
    else
        error("not implemented format: ", format)
    end
end


function create_intervals(Q_preds, α = 0.05)
    pred_intervals = zeros(length(Q_preds), 2)
    for (i, Q_pred) in enumerate(Q_preds)
        try
            pred_intervals[i, 1] = quantile(Q_pred, α / 2) # lower end
        catch e # 0 and 1 quantiles somethimes throw an error:
            if isa(e, ArgumentError)
                pred_intervals[i, 1] = minimum(Q_pred)
            else
                throw(e)
            end
        end

        try
            pred_intervals[i, 2] = quantile(Q_pred, 1 - α / 2)
        catch e # 0 and 1 quantiles somethimes throw an error:
            if isa(e, ArgumentError)
                pred_intervals[i, 2] = maximum(Q_pred)
            else
                throw(e)
            end
        end
    end
    pred_intervals
end


if abspath(PROGRAM_FILE) == @__FILE__
    _argparse_setting = ArgParseSettings()
    @add_arg_table! _argparse_setting begin
        "EXPORTED_PREDS_DIR"
            help = "path to the `exported_predictions` folder under the given recalibration model's folder"
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
    @info "run interval pred on recal model $(pargs["EXPORTED_PREDS_DIR"])" CUDA.device()

    EVAL_DIR = @__DIR__ 
    EVAL_DIR *= "/../eval"
    include(joinpath(EVAL_DIR, "experiment_walker.jl"))

    file_list = data_path_collecter(pargs["EXPORTED_PREDS_DIR"])
    # file_list = file_list[1:5]
    N = length(file_list)
    results = Vector{Tuple{String, String, Int, Float64, Float64, Float64, Float64}}(undef, N)

    error_log = []
    err_lock = ReentrantLock()
    @showprogress desc = "run interval pred" showspeed = true @threads for i in 1:N
    # @showprogress desc = "run interval pred" showspeed = true for i in 1:N
        f = file_list[i]
        try
            Q_preds, obs_test = _handle_format(f.PRED_FORMAT, f.pred_dir, 
                f.obs_test_file, f.obs_train_file)

            intervals = create_intervals(Q_preds)
            Z_PIT = cdf.(Q_preds, obs_test)
            res = eval_interval_result(intervals, obs_test, Z_PIT)

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
    df = DataFrame(results, [:repo, :dataset, :split, :coverage, :average_int_len, :median_int_len, :dcor])
    CSV.write(pargs["out_file"], df)
    @info "done, results saved to $(pargs["out_file"])"
end