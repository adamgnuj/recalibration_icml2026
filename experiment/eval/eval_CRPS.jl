using ArgParse, JLD2, ProgressMeter, DataFrames
using CSV, Distributions, Logging, Dates, StatsBase, Calibration
using CUDA
using Tullio, KernelAbstractions


function _tullio_crps(obs_train::AbstractVector, obs_test::AbstractVector, pred_test::AbstractMatrix)
    obs_train = cu(obs_train)
    obs_test = cu(obs_test)
    pred_test = cu(pred_test)

    _cross = sum(pred_test .* abs.(obs_test .- obs_train'), dims = 2) |> vec
    _self = sum(pred_test' .* (abs.(obs_train .- obs_train') * pred_test'), dims = 1) |> vec
    # @tullio _cross[i] := abs(obs_train[j] - obs_test[i]) * pred_test[i, j]
    # @tullio _self[i] := abs(obs_train[j] - obs_train[k]) * pred_test[i, j] * pred_test[i, k]
    crps = _cross - 0.5f0 * _self
    Array{Float64}(crps)
end

function _empirical_unif_weight_crps(M, obs_test)
    @tullio _self[i] := abs(M[i, k] - M[i, l])
    _self ./= size(M, 2) * (size(M, 2) - 1)
    @tullio _cross[i] := abs(M[i, k] - obs_test[i])
    _cross ./= size(M, 2)
    _cross - 0.5*_self
end

function _CRPS_handle_format(format::String, pred_dir::String, obs_test_file::String, obs_train_file::String)
    obs_test = CSV.read(obs_test_file, DataFrame)[:,1] .|> Float64
    obs_train = CSV.read(obs_train_file, DataFrame)[:,1] .|> Float64
    data_dir = first(splitdir(obs_train_file))
    obs_calib = CSV.read(joinpath(data_dir, "y_validation.csv"), DataFrame)[:,1] .|> Float64


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
        crps = _tullio_crps(obs_train, obs_test, pred_test)
        mean(crps)

    elseif format == "WeightedFixSampleEmpirical_train_and_validation"
        pred_fn = joinpath(pred_dir, "preds_test.csv")
        pred_test = Matrix(CSV.read(pred_fn, DataFrame))
        crps = _tullio_crps([obs_train; obs_calib], obs_test, pred_test)
        mean(crps)

    elseif format == "GPB"
        pred_test = Matrix(CSV.read(joinpath(pred_dir, "test_recal_s.csv"), DataFrame))
        pred_test ./= sum(pred_test, dims = 2)
        grid = CSV.read(joinpath(pred_dir, "test_recal_t.csv"), DataFrame) |> Matrix |> vec
        crps = _tullio_crps(grid, obs_test, pred_test)
        mean(crps)


    elseif format == "KDO.jld2"
        r = load(joinpath(pred_dir, "test_recalibration.jld2"))
        # crps = Calibration.mean_crps_laplace_from_gamma(
        #     r["z"], 
        #     StatsBase.transform(r["standardizer"], obs_test),
        #     r["G_test"],
        #     r["Q_kernel"]
        # )
        
        crps = _tullio_crps(obs_calib, obs_test, r["preds_calib_test_B"][:, length(obs_calib)+1:end]')


        # crps .*= only(r["standardizer"].scale) # scale equivariant scoring rule
        mean(crps)
    elseif format == "gaussianmixture"
        fl = readdir(pred_dir)
        if isempty(fl)
            @error "empty preddir" pred_dir
        end
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
        L, V, P = cu(L), cu(V), cu(P)
        P ./= sum(P, dims = 2)
        
        _self = Calibration.batched_gaussian_self_mae(L, V, P)
        _cross = Calibration.batched_gaussian_obs_mae(L, V, P, cu(obs_test))
        mean(_cross - 0.5 * _self)  

    elseif format == "uniformweightempirical"
        ls = readdir(pred_dir)
        if isempty(ls)
            @warn "there is no recal in " pred_dir
        end
        pred_fn = joinpath(pred_dir, first(ls)) 
        preds_test = CSV.read(pred_fn, DataFrame) |> Matrix
        _empirical_unif_weight_crps(cu(preds_test), cu(obs_test)) |> mean
    else
        error("not implemented format: ", format)
    end
end





if abspath(PROGRAM_FILE) == @__FILE__

    if "GPU_DEVICE" in keys(ENV)
        GPU_DEVICE = parse(Int, ENV["GPU_DEVICE"])
        CUDA.device!(GPU_DEVICE)
    else
        GPU_DEVICE = CUDA.device()
    end

    @info "gpu:" Base.active_project() GPU_DEVICE ARGS


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
    @info "eval CRPS on model at" pargs["EXPORTED_PREDS_DIR"]

    EVAL_DIR = @__DIR__
    include(joinpath(EVAL_DIR, "experiment_walker.jl"))

    file_list = data_path_collecter(pargs["EXPORTED_PREDS_DIR"])

    df = DataFrame(
        repo = String[], 
        dataset = String[],
        split = Int[],
        CRPS = Float64[])

    pr = Progress(length(file_list); desc = "eval CRPS", showspeed = true)
    _start = time()
    for f in file_list
        _crps = _CRPS_handle_format(f.PRED_FORMAT, f.pred_dir, f.obs_test_file, f.obs_train_file)
        push!(df, (f.repo, f.dataset, parse(Int, f.split_idx), _crps))
        next!(pr; showvalues = [
            (:repo, f.repo), 
            (:dataset, f.dataset), 
            (:split, f.split_idx), 
            (:CRPS, _crps),
            (:ellapsed, Time(0) + Second(round(Int, time() - _start)))])
    end

    CSV.write(pargs["out_file"], df)
    @info "done, results saved to $(pargs["out_file"])"
end