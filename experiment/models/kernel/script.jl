@info "env:" Base.active_project()

using DataFrames, CSV, Distributions, CUDA
using Calibration
using StatsBase, Statistics, Random
using ProgressMeter, LinearAlgebra
using Optim

if "GPU_DEVICE" in keys(ENV)
    GPU_DEVICE = parse(Int, ENV["GPU_DEVICE"])
    CUDA.device!(GPU_DEVICE)
else
    GPU_DEVICE = CUDA.device()
end

@info "gpu:" Base.active_project() GPU_DEVICE ARGS


# utils:
_safe_sqrt(x::Real) = x >= zero(x) ? sqrt(x) : zero(x)
"""
X is a n_obs × n_features matrix -> get eeuclidean distance mtx n_obs × n_obs
"""
function _distance_mtx(X)
    G = X * X'
    D2 = diag(G) .+ diag(G)' - 2G
    D = _safe_sqrt.(D2)
    D
end

function approx_median_distance(D::CuMatrix; n_samples = 100_000)
    n = size(D, 1)
    max_idx = n * (n - 1) ÷ 2
    n_samples = min(n_samples, max_idx)
    
    # 1. Generate random 1D indices corresponding to the upper triangle
    k_indices = rand(1:max_idx, n_samples)
    
    i_indices = Vector{Int}(undef, n_samples)
    j_indices = Vector{Int}(undef, n_samples)
    
    @inbounds for (idx, k) in enumerate(k_indices)
        # Inverse mapping formula for column-major upper triangle
        j = floor(Int, (1 + sqrt(1 + 8 * (k - 1))) / 2) + 1
        i = k - (j - 1) * (j - 2) ÷ 2
        
        i_indices[idx] = i
        j_indices[idx] = j
    end
    
    # 3. Create Cartesian Indices, push to GPU, and fetch only the samples
    cart_indices = CuArray(CartesianIndex.(i_indices, j_indices))
    samples_gpu = D[cart_indices]
    
    # 4. Pull the small sample back to CPU and compute median
    return median(Array(samples_gpu))
end


function prepare_data(X_train, X_valid,X_test, y_train, y_valid)
    X_standardizer = fit(ZScoreTransform, X_train; dims = 1)
    X_standardizer.scale[X_standardizer.scale .<= 1e-10] .= 1.0 # keep deterministic features
    StatsBase.transform!(X_standardizer, X_train)
    StatsBase.transform!(X_standardizer, X_valid)
    StatsBase.transform!(X_standardizer, X_test)

    train_mask = BitVector([
            ones(size(X_train, 1))..., 
            zeros(size(X_valid, 1))..., 
            zeros(size(X_test, 1))...])
    
    valid_mask = BitVector([
            zeros(size(X_train, 1))..., 
            ones(size(X_valid, 1))..., 
            zeros(size(X_test, 1))...])

    X_all = [
        X_train;
        X_valid;
        X_test
    ] |> cu
    y_all = [
        y_train;
        y_valid
    ] |> cu
    
    X_all, y_all, train_mask, valid_mask
end

"return the lambda opt based on cv using eigendecomp"
function _get_cv_lambda_loss(
        cv_masks::Vector{BitVector}, 
        preds_train_K::AbstractMatrix, 
        obs_train_K::AbstractMatrix)
    
    n_cv = length(cv_masks)
    # eigen decompositions
    # _eig_first = eigen(preds_train_K[cv_masks[1], cv_masks[1]])

    _KK = preds_train_K[cv_masks[1], cv_masks[1]]
    _KK .= (_KK + _KK') / 2
    # _eig_first = eigen(_KK)
    _eig_first = eigen(Symmetric(_KK))

    cv_eigens_preds_train_K = Array{typeof(_eig_first)}(undef, n_cv)
    cv_eigens_preds_train_K[1] = _eig_first
    @showprogress desc = "eig. decomp for cross-validation" for i = 2:n_cv
        # cv_eigens_preds_train_K[i] = eigen(
            # preds_train_K[cv_masks[i], cv_masks[i]])
        _KK = preds_train_K[cv_masks[i], cv_masks[i]]
        _KK .= (_KK + _KK') / 2
        # cv_eigens_preds_train_K[i] = eigen(_KK)
        cv_eigens_preds_train_K[i] = eigen(Symmetric(_KK))
    end

    _lcv(l) = Calibration._mean_cv_error(
        cv_masks, cv_eigens_preds_train_K, preds_train_K, obs_train_K, l)
    
    # l_grid_errors = []
    # @showprogress desc = "opt cv lambda" for l in lambda_grid
    #     err = _lcv(l)
    #     push!(l_grid_errors, err)
    # end
    # i = argmin(l_grid_errors)
    # if i == 1 || i == length(l_grid_errors)
    #     @warn "l_grid_errors edge case:" i length(l_grid_errors)
    # end
    # lambda_grid[i], l_grid_errors[i] # λ_opt, err
    @info "start lambda opt" 
    # opt = optimize(_lcv, 0, 10, Brent(); time_limit = 10.0)
    opt = optimize(l -> _lcv(exp(l)), -10, 2, Brent(); time_limit = 30.0)
    if ~Optim.converged(opt)
        @warn "lambda opt optim reached time limit"
    end
    if opt.minimizer in (opt.initial_upper, opt.initial_lower)
        @warn "lambda opt optim edge case" opt
    end
    # @info "minimizer" opt.minimizer
    @info "minimizer" exp(opt.minimizer)
    exp(opt.minimizer), opt.minimum
end

function kernel_ckme_pred(X_train, X_valid,X_test, y_train, y_valid; seed = 42)
    Random.seed!(seed)
    cv_masks = Calibration._draw_cv_masks(size(X_train, 1))
    # lambda_grid = logrange(1f-6, 1, 10) |> collect
    
    X_all, y_all, train_mask, valid_mask = prepare_data(
        X_train, X_valid,X_test, y_train, y_valid)

    D_y_all = abs.(y_all .- y_all')
    σ_y = approx_median_distance(D_y_all[train_mask, train_mask])
    @. D_y_all = exp(-abs.(D_y_all) / (2σ_y))
    K_y_all = D_y_all # reuse distance buffer

    D_X_all = _distance_mtx(X_all)
    σ_x_medianh = approx_median_distance(D_X_all[train_mask, train_mask])
    @info "median heuristic" σ_x_medianh
    σ_x_grid = fill(σ_x_medianh, 9)
    σ_x_grid .*= logrange(0.001, 10, 9)

    K_x_all = similar(D_X_all) # buffer for kernel mtx

    sigma_hist = []
    for σ_x in σ_x_grid
        @info "kernel bandwith" σ_x
        @. K_x_all = exp(-D_X_all^2 / (2σ_x)) # inplace
        λ_opt, err = _get_cv_lambda_loss(
            cv_masks, K_x_all[train_mask, train_mask], 
            K_y_all[train_mask, train_mask])
        
        push!(sigma_hist, (λ_opt, err))
    end
    j = argmin(last.(sigma_hist))
    if j == 1 || j == length(σ_x_grid)
        @warn "σ_x_grid edge case:" j length(σ_x_grid)
    end
    # make predictions:
    λ_opt, _ = sigma_hist[j]
    σ_x = σ_x_grid[j]
    @info "best parameters:" λ_opt σ_x
    @. K_x_all = exp(-D_X_all^2 / (2σ_x)) # inplace

    # free up GPU ram
    D_X_all = nothing # Remove references
    K_y_all = nothing

    GC.gc(true)  # Force a full garbage collection sweep
    CUDA.reclaim()

    # B_pred = (K_x_all[train_mask, train_mask] 
    #         + λ_opt * size(X_train, 1) * I) \ K_x_all[train_mask, :]
    
    
    # W_pred = Calibration._batched_euclidean_simplex_proj(B_pred) |> Array

    # W_train = W_pred[:, train_mask]
    # W_valid = W_pred[:, valid_mask]
    # W_test = W_pred[:, .!(train_mask .| valid_mask)]

    A = K_x_all[train_mask, train_mask] + λ_opt * size(X_train, 1) * I

    # only produce predictions on the test set to save gpu RAM

    # W_train = (A \ K_x_all[train_mask, train_mask]) |> Calibration._batched_euclidean_simplex_proj |> Array
    # W_valid = (A \ K_x_all[train_mask, valid_mask]) |> Calibration._batched_euclidean_simplex_proj |> Array
    W_test = (A \ K_x_all[train_mask, .!(train_mask .| valid_mask)]) |> Calibration._batched_euclidean_simplex_proj |> Array

    
    # W_train, W_valid, W_test
    W_test, W_test, W_test

    # W_pred, sigma_hist, σ_x
end

function main()
    args_parsed = NamedTuple{(
            :X_train, :X_validation, :X_test, 
            :y_train, :y_validation, 
            :preds_train, :preds_validation, :preds_test
            )}(ARGS)

    y_train = CSV.read(args_parsed.y_train, DataFrame) |> Matrix |> vec .|> Float64
    y_valid = CSV.read(args_parsed.y_validation, DataFrame) |> Matrix |> vec .|> Float64
    X_train = CSV.read(args_parsed.X_train, DataFrame) |> Matrix .|> Float64
    X_valid = CSV.read(args_parsed.X_validation, DataFrame) |> Matrix .|> Float64
    X_test = CSV.read(args_parsed.X_test, DataFrame) |> Matrix .|> Float64

    # clip at 20_000 train size
    if size(X_train, 1) > 20_000
        @warn "use only first 20k training observations (memory limit)"
        X_train = X_train[1:20_000, :]
        y_train = y_train[1:20_000, :]
    end

    W_train, W_valid, W_test = kernel_ckme_pred(
        X_train, X_valid,X_test, y_train, y_valid)

    CSV.write(args_parsed.preds_train, DataFrame(Array(W_train)', :auto))
    CSV.write(args_parsed.preds_validation, DataFrame(Array(W_valid)', :auto))
    CSV.write(args_parsed.preds_test, DataFrame(Array(W_test)', :auto))
    @info "DONE, (test) predictions saved to:" args_parsed.preds_test
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "run ckme basline model's main()"
    wct = @elapsed main()
    @info "WALL_CLOCLK_TIME: $wct"
end