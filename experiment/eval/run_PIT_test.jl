using ArgParse, HypothesisTests, JLD2, ProgressMeter, DataFrames
using CSV, Distributions, Logging, Dates, StatsBase, Calibration

function _PIT_handle_format(format::String, pred_dir::String, obs_test_file::String, obs_train_file::String)
    obs_test = CSV.read(obs_test_file, DataFrame)[:,1] .|> Float64
    obs_train = CSV.read(obs_train_file, DataFrame)[:,1] .|> Float64
    obs_calib = CSV.read(joinpath(first(splitdir(obs_train_file)), "y_validation.csv"), DataFrame)[:,1] .|> Float64

    if format == "WeightedFixSampleEmpirical"
        # pred_fn = joinpath(pred_dir, first(readdir(pred_dir)))
        fl = readdir(pred_dir)
        if "predictions_test.csv" in fl
            pred_fn = joinpath(pred_dir, "predictions_test.csv")
        else
            pred_fn = joinpath(pred_dir, first(fl)) 
            # recalibrated models are only evaluated at the test set
        end
        
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        preds_Q = [Calibration.Empirical(obs_train, w) for w in eachrow(pred_test)]
        cdf.(preds_Q, obs_test)

    elseif format == "WeightedFixSampleEmpirical_train_and_validation"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir)))
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        preds_Q = [Calibration.Empirical([obs_train; obs_calib], w) for w in eachrow(pred_test)]
        cdf.(preds_Q, obs_test)

    elseif format == "GPB"
        pred_test = Matrix(CSV.read(joinpath(pred_dir, "test_recal_s.csv"), DataFrame))
        pred_test ./= sum(pred_test, dims = 2)
        grid = CSV.read(joinpath(pred_dir, "test_recal_t.csv"), DataFrame) |> Matrix |> vec
        preds_Q = [Calibration.Empirical(grid, w) for w in eachrow(pred_test)]
        cdf.(preds_Q, obs_test)

    elseif format == "KDO.jld2"
        r = load(joinpath(pred_dir, "test_recalibration.jld2"))
        # Calibration._PIT_transform(r["z"], r["Q_kernel"], r["G_test"], StatsBase.transform(r["standardizer"], obs_test))
        pred_test = r["preds_calib_test_B"][:, length(obs_calib)+1:end]'
        preds_Q = [Calibration.Empirical(obs_calib, w) for w in eachrow(pred_test)]
        cdf.(preds_Q, obs_test)

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
        cdf.(Q, obs_test)

    elseif format == "uniformweightempirical"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir))) 
        preds_test = CSV.read(pred_fn, DataFrame) |> Matrix
        
        Q_test = [Calibration.Empirical(q) for q in eachrow(preds_test)]
        cdf.(Q_test, obs_test)
    else
        error("not implemented format: ", format)
    end
end

function run_pval_test(t)
    Z = _PIT_handle_format(t.PRED_FORMAT, t.pred_dir, t.obs_test_file, t.obs_train_file)
    with_logger(NullLogger()) do
        t = ExactOneSampleKSTest(Z, Uniform())
        t.δ, pvalue(t) # stat, pval
    end
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
    end

    pargs = parse_args(ARGS, _argparse_setting)
    @info "run PIT test on model $(pargs["EXPORTED_PREDS_DIR"])"

    EVAL_DIR = @__DIR__
    include(joinpath(EVAL_DIR, "experiment_walker.jl"))

    file_list = data_path_collecter(pargs["EXPORTED_PREDS_DIR"])

    df = DataFrame(
        repo = String[], 
        dataset = String[],
        split = Int[],
        stat = Float64[],
        pval = Float64[])

    pr = Progress(length(file_list); desc = "run PIT test", showspeed = true)
    _start = time()
    for f in file_list
        _stat, _pval = run_pval_test(f)
        push!(df, (f.repo, f.dataset, parse(Int, f.split_idx), _stat, _pval))
        next!(pr; showvalues = [
            (:repo, f.repo), 
            (:dataset, f.dataset), 
            (:split, f.split_idx), 
            (:pvalue, _pval),
            (:ellapsed, Time(0) + Second(round(Int, time() - _start)))])
    end

    CSV.write(pargs["out_file"], df)
    @info "done, results saved to $(pargs["out_file"])"
end