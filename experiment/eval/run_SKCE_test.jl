using ArgParse, HypothesisTests, JLD2, ProgressMeter, DataFrames
using CSV, Distributions, Logging, Dates, StatsBase, Calibration
using CUDA, LinearAlgebra
using Tullio, KernelAbstractions

CUDA.allowscalar(false)

function _gauss_skce(y_obs, L, V, P)
    # @info "start with" length(y_obs)
    Q = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]
    
    ns = 1000
    # @info "start sampling"
    M = cu(hcat([rand(q, ns) for q in Q]...)') # size: n_test x ns 
    # @info "finished sampling"

    L, V, P = cu(L), cu(V), cu(P)

    pw_dist_obs = abs.(y_obs .- y_obs')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel = Laplace(0, med_obs)
    K_ytest = pdf.(Q_kernel, y_obs .- y_obs')

    pw_mae = Calibration.batched_gaussian_pairwise_mae(cu(L), cu(V), cu(P))
    pw_edist = pw_mae - 0.5(diag(pw_mae) .+ diag(pw_mae)')
    med_pred = Calibration.median_distance(pw_edist)
    Q_kernel_pred = Laplace(0, med_pred)
    K_qtest = pdf.(Q_kernel_pred, pw_edist)
    # @info "start tullio L_qq"

    @tullio L_qq[i,j] := pdf(Q_kernel, M[i, k] - M[j, l])
    L_qq ./= ns * (ns - 1)
    # @info "start tullio L_qy"
    y_obs_cu = cu(y_obs)
    @tullio L_qy[i,j] := pdf(Q_kernel, M[i, k] - y_obs_cu[j])
    L_qy ./= ns
    prec_skce_kernel = Calibration.PrecomputedSKCEKernel(Array(K_qtest), K_ytest,Array(L_qy), Array(L_qq));
    # @info "start test"
    Calibration.AsymptoticSKCETest(prec_skce_kernel, 1:length(y_obs), 1:length(y_obs))
end


function _unifweight_skce(y_obs, M)
    # @info "start with" length(y_obs)
    M = cu(M) # size: n_test x ns 
    # @info "finished sampling"


    pw_dist_obs = abs.(y_obs .- y_obs')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel = Laplace(0, med_obs)
    K_ytest = pdf.(Q_kernel, y_obs .- y_obs')

    pw_mae = Calibration.pairwise_mae(M')

    pw_edist = pw_mae - 0.5(diag(pw_mae) .+ diag(pw_mae)')
    med_pred = Calibration.median_distance(pw_edist)
    Q_kernel_pred = Laplace(0, med_pred)
    K_qtest = pdf.(Q_kernel_pred, pw_edist)
    # @info "start tullio L_qq"

    @tullio L_qq[i,j] := pdf(Q_kernel, M[i, k] - M[j, l])
    ns = size(M, 2)
    L_qq ./= ns * (ns - 1)
    # @info "start tullio L_qy"
    y_obs_cu = cu(y_obs)
    @tullio L_qy[i,j] := pdf(Q_kernel, M[i, k] - y_obs_cu[j])
    L_qy ./= ns
    prec_skce_kernel = Calibration.PrecomputedSKCEKernel(Array(K_qtest), K_ytest,Array(L_qy), Array(L_qq));
    # @info "start test"
    Calibration.AsymptoticSKCETest(prec_skce_kernel, 1:length(y_obs), 1:length(y_obs))
end

function _SKCE_handle_format(format::String, pred_dir::String, obs_test_file::String, obs_train_file::String)
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
        Calibration.EmpiricalSKCETest(obs_test, obs_train, pred_test)

    elseif format == "WeightedFixSampleEmpirical_train_and_validation"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir)))
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        Calibration.EmpiricalSKCETest(obs_test, [obs_train; obs_calib], pred_test)

    elseif format == "GPB"
        pred_test = Matrix(CSV.read(joinpath(pred_dir, "test_recal_s.csv"), DataFrame))
        pred_test ./= sum(pred_test, dims = 2)
        grid = CSV.read(joinpath(pred_dir, "test_recal_t.csv"), DataFrame) |> Matrix |> vec
        Calibration.EmpiricalSKCETest(obs_test, grid, pred_test)

    elseif format == "KDO.jld2"
        r = load(joinpath(pred_dir, "test_recalibration.jld2"))
        # obs_test_scale = StatsBase.transform(r["standardizer"], obs_test)
        # obs_K = pdf(r["Q_kernel"], obs_test_scale .- obs_test_scale')
        # Calibration.LaplaceSKCETest(r["Q_kernel"], r["Q_kernel_distr"], 
            # r["z"], r["G_test"], obs_test_scale, obs_K)
        pred_test = r["preds_calib_test_B"][:, length(obs_calib)+1:end]'
        Calibration.EmpiricalSKCETest(obs_test, obs_calib, pred_test)
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

        _gauss_skce(obs_test, L, V, P)

    elseif format == "uniformweightempirical"
        pred_fn = joinpath(pred_dir, first(readdir(pred_dir))) 
        preds_test = CSV.read(pred_fn, DataFrame) |> Matrix
        
        # Q_test = [Calibration.Empirical(q) for q in eachrow(preds_test)]
        # Z = cdf.(Q_test, obs_test)

        # _pw_mae = Calibration.pairwise_mae(preds_test')
        # res = Calibration._pw_mae_and_Z_HSICTest(Z, _pw_mae)
        _unifweight_skce(obs_test, preds_test)

    else
        error("not implemented format: ", format)
    end
end

function run_skce_test(t)
    t = _SKCE_handle_format(t.PRED_FORMAT, t.pred_dir, t.obs_test_file, t.obs_train_file)
    t.statistic, pvalue(t)
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
    @info "run SCKE test on model $(pargs["EXPORTED_PREDS_DIR"])"

    CUDA.device!(pargs["gpu"])
    @info CUDA.device()

    EVAL_DIR = @__DIR__
    include(joinpath(EVAL_DIR, "experiment_walker.jl"))

    file_list = data_path_collecter(pargs["EXPORTED_PREDS_DIR"])

    df = DataFrame(
        repo = String[], 
        dataset = String[],
        split = Int[],
        stat = Float64[],
        pval = Float64[])

    pr = Progress(length(file_list); desc = "run SKCE test", showspeed = true)
    _start = time()
    for f in file_list
        _stat, _pval = run_skce_test(f)
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