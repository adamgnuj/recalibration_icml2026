using Random, JLD2
using StatsBase, LinearAlgebra, Distributions

include("lambda_cross_validation.jl")




function _draw_cv_masks(n_train, n_cv = 5)
    _ms = vcat([i * ones(Int, ceil(Int, n_train / n_cv)) for i = 1:n_cv]...) |> shuffle!
    cv_masks = [(_ms .!= i)[1:n_train] for i = 1:n_cv]
    cv_masks
end


abstract type PredictionRepresentation end


struct MedianHeuristic
    kernel::Distribution
end

"""
```
struct SampleEmpirical <: PredictionRepresentation
    kernel :: Distribution
end
```

Represent predictions with samples. `train_preds`, `calib_preds` 
and `obs_preds` should be ``ℝ^{m × n}``, where ``m`` is the number of samples in a distribution
and ``n`` is the size of the train/calib/test set. Use the `kernel` for computing the kernelmatrix.

(The actual kernel is defined as `k(u,v) = pdf(kernel, u - v)`)

Since we have an efficient GPU implementation, ``m = 5000`` is a suitable default.
"""
struct SampleEmpirical <: PredictionRepresentation 
    kernel::Distribution
end

"""
`preds = (W, y)` where `y` are in ℝ^n fix samples and `W` ∈ ℝ^{m × n} are weights for each prediction ``i = 1, …, m``.
"""
struct WeightedFixSampleEmpirical <: PredictionRepresentation 
    kernel::Distribution
end

"""
`preds = (loc, var, pi)` where `loc` ∈ ℝ^{m × nc} are the location parameters for the `m` predictions and `nc` components. 
(`var` and `pi` are the variance and the mixture weights)
"""
struct GaussianMixture <: PredictionRepresentation 
    kernel::Distribution
end



mutable struct ReCalibrationConfiguration
    name::String
    directory_path::String
    train_obs::Any
    train_preds::Any
    calib_obs::Any
    calib_preds::Any
    # test_obs::Any
    test_preds::Any
    # lambda_tilde_grid::AbstractVector{<:Float64}
    cv_mask_seed::Int64
    preds_repr::PredictionRepresentation
    obs_repr::MedianHeuristic
    # KDO_conf::KernelDensityOperator.KDOConfiguration
end

Base.show(io::IO, conf::ReCalibrationConfiguration) = print(io, 
        """RecalibrationConf( 
        \tname=$(conf.name) 
        \tdirectory=$(conf.directory_path) 
        \tcv_mask_seed=$(conf.cv_mask_seed) 
        \tpreds_repr=$(conf.preds_repr) 
        \tobs_repr=$(conf.obs_repr) 
        )"""
        )

import KernelFunctions: kernelmatrix

function kernelmatrix(repr::SampleEmpirical, calib_preds, test_preds)
    preds_calib_test = [calib_preds test_preds] |> cu
    preds_calib_test_pw_mae = pairwise_mae(preds_calib_test) # compute energy distance between predictions
    _di(G) = sqrt.(max.(G - 0.5f0 * (diag(G) .+ diag(G)'), 0.0f0)) # distance mtx from edist matrix 
    preds_calib_test_pw_edist = _di(preds_calib_test_pw_mae)
            

    test_mask = BitVector([zeros(size(calib_preds, 2)); ones(size(test_preds, 2))]) # test set idx mask
    
    median_preds = median_distance(preds_calib_test_pw_edist[.!test_mask, .!test_mask])
    Q_kernel_distr = repr.kernel * median_preds

    preds_calib_test_K = pdf(Q_kernel_distr, preds_calib_test_pw_edist)
    n_calib = size(calib_preds, 2)
    n_test = size(test_preds, 2)
    (;Q_kernel_distr, preds_calib_test_K, n_calib, n_test)
end

"function kernelmatrix(repr::WeightedFixSampleEmpirical, calib_preds, test_preds)"
function kernelmatrix(repr::WeightedFixSampleEmpirical, calib_preds, test_preds)
    W_calib, y_train = calib_preds
    W_test, _ = test_preds

    y_train = cu(y_train)
    W = cu([W_calib; W_test])

    y_mae = abs.(y_train .- y_train')

    preds_calib_test_pw_mae = W * y_mae * W'

    _di(G) = sqrt.(max.(G - 0.5f0 * (diag(G) .+ diag(G)'), zero(eltype(G)))) # distance mtx from edist matrix 
    # _di(G) = G - 0.5f0 * (diag(G) .+ diag(G)') # distance mtx from edist matrix 
    preds_calib_test_pw_edist = _di(preds_calib_test_pw_mae)
    preds_calib_test_pw_edist = 0.5(preds_calib_test_pw_edist + preds_calib_test_pw_edist') # fix numerical errors
            

    test_mask = BitVector([zeros(size(W_calib, 1)); ones(size(W_test, 1))]) # test set idx mask
    
    median_preds = median_distance(preds_calib_test_pw_edist[.!test_mask, .!test_mask])
    Q_kernel_distr = repr.kernel * median_preds

    preds_calib_test_K = pdf(Q_kernel_distr, preds_calib_test_pw_edist)
    n_calib = size(W_calib, 1)
    n_test = size(W_test, 1)
    (;Q_kernel_distr, preds_calib_test_K, n_calib, n_test)
end

"function kernelmatrix(repr::WeightedFixSampleEmpirical, calib_preds, test_preds)"
function kernelmatrix(repr::GaussianMixture, calib_preds, test_preds)
    calib_loc, calib_var, calib_pi = calib_preds 
    test_loc, test_var, test_pi = test_preds 

    Q_calib_preds = [MixtureModel(Normal.(collect(l),sqrt.(collect(s))), collect(p)) for (l, s, p) in zip(eachrow(calib_loc), eachrow(calib_var), eachrow(calib_pi))]
    Q_test_preds = [MixtureModel(Normal.(collect(l),sqrt.(collect(s))), collect(p)) for (l, s, p) in zip(eachrow(test_loc), eachrow(test_var), eachrow(test_pi))]


    preds_calib_test_pw_mae = batched_gaussian_pairwise_mae(
        cu([calib_loc; test_loc]),
        cu([calib_var; test_var]),
        cu([calib_pi; test_pi]))



    _di(G) = sqrt.(max.(G - 0.5f0 * (diag(G) .+ diag(G)'), zero(eltype(G)))) # distance mtx from edist matrix 
    # _di(G) = G - 0.5f0 * (diag(G) .+ diag(G)') # distance mtx from edist matrix 
    preds_calib_test_pw_edist = _di(preds_calib_test_pw_mae)
    preds_calib_test_pw_edist = 0.5(preds_calib_test_pw_edist + preds_calib_test_pw_edist') # fix numerical errors
            

    test_mask = BitVector([zeros(length(Q_calib_preds)); ones(length(Q_test_preds))]) # test set idx mask
    
    median_preds = median_distance(preds_calib_test_pw_edist[.!test_mask, .!test_mask])
    Q_kernel_distr = repr.kernel * median_preds

    preds_calib_test_K = pdf(Q_kernel_distr, preds_calib_test_pw_edist)
    n_calib = length(Q_calib_preds)
    n_test = length(Q_test_preds)
    (;Q_kernel_distr, preds_calib_test_K, n_calib, n_test)
end

function kernelmatrix(MH::MedianHeuristic, calib_obs)
    obs_calib = calib_obs |> cu
    
    obs_calib_pw_dist = abs.(obs_calib .- obs_calib')

    median_obs = median_distance(obs_calib_pw_dist) # median heuristic

    Q_kernel = MH.kernel * median_obs
    obs_calib_K = pdf(Q_kernel, obs_calib_pw_dist)

    
    (;Q_kernel, obs_calib_K)
end

"""
```
_nonnegative_KDO_sol(λ̃, K_zz, K_zy_calib, recalpreds_calib_test_B) -> G
```
Use the original linear KDO solution `γ = (K_zz + λ̃ * n * I)^-2 K_zy β`, which works well if ``β ∈ Δₙ``.
> Kernel Conditional Density Operators
"""
function _nonnegative_KDO_sol(λ̃, K_zz, K_zy_calib, recalpreds_calib_test_B)
    n_z = LinearAlgebra.checksquare(K_zz)
    # G = (K_zz + λ̃ * n_z * I)^2 \ (K_zy_calib * recalpreds_calib_test_B)
    _g = (K_zz + λ̃ * n_z * I) \ (K_zy_calib * recalpreds_calib_test_B)
    G = (K_zz + λ̃ * n_z * I) \ _g
    G ./= sum(G, dims = 1)
    G
end


function run_recalibration(conf::ReCalibrationConfiguration)
    @info "---- start recalibration for ----" conf
    _start_time = time()

    train_obs, calib_obs = cu(conf.train_obs), cu(conf.calib_obs)
    Q_kernel_distr, preds_calib_test_K, n_calib, n_test = kernelmatrix(conf.preds_repr, conf.calib_preds, conf.test_preds)
    Q_kernel, obs_calib_K = kernelmatrix(conf.obs_repr, conf.calib_obs)

    tm = BitVector([zeros(n_calib); ones(n_test)]) # test set idx mask

    standardizer = fit(ZScoreTransform, [train_obs; calib_obs]) # standardize observations
    StatsBase.transform!(standardizer, calib_obs)

    Random.seed!(conf.cv_mask_seed)
    cv_masks = _draw_cv_masks(sum(.!tm))

    @info "start cross validation opt. for λ"
    elapsed = @elapsed λ = get_cv_lambda(cv_masks, 
        preds_calib_test_K[.!tm, .!tm], 
        obs_calib_K)
    @info "finished cross validation opt. for λ" λ elapsed

    _rK = preds_calib_test_K[.!tm, .!tm] + λ * length(calib_obs) * I # regularized linear pred in RKHS dual
    recalpreds_calib_test_B = _rK \ preds_calib_test_K[.!tm, :]

    # euclidean projection of CKME weights
    recalpreds_calib_test_B = _batched_euclidean_simplex_proj(recalpreds_calib_test_B)

    

    fn = joinpath(conf.directory_path, conf.name * "_recalibration.jld2")
    @info "save recalibration results to" fn
    jldsave(fn; 
        # λ̃ = λ̃, 
        λ = λ,
        # lt_cv_errors = errors,
        # z = Array{Float64}(z),
        # K_zz = Array{Float64}(K_zz),
        # K_zy_calib = Array{Float64}(K_zy_calib),
        Q_kernel,
        Q_kernel_distr,
        preds_calib_test_B = Array{Float64}(recalpreds_calib_test_B),   
        cv_masks,
        preds_calib_test_K = Array{Float64}(preds_calib_test_K),
        test_mask = tm,
        # G_calib = Array{Float64}(G_calib),
        # G_test = Array{Float64}(G_test),
        standardizer = ZScoreTransform(standardizer.len, standardizer.dims, 
                            Array(standardizer.mean), Array(standardizer.scale))
        )
    @info "done in $(round(Int, time() - _start_time)) secs"
end