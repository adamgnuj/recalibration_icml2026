# lambda tilde cross validation: optimize the regularization for kdo
# the objective value is the KS test statistic

using HypothesisTests, Plots, Distributions

function get_cv_lambda_tilde(
        Q_kernel::Distribution,
        cv_masks::Vector{BitVector}, 
        preds_calib_K::AbstractMatrix, 
        z::AbstractVector,
        obs_calib::AbstractVector,
        K_zz::AbstractMatrix,
        K_zy_calib::AbstractMatrix,
        # recalpreds_calib_B::AbstractMatrix;
        λ::Real,
        λ̃_grid
    )


#     # ------------------ calculate cv β ------------------
    n_calib = size(preds_calib_K, 1)
    n_cv = length(cv_masks)
    B_cv = []

    # calib_obs_K = pdf(Q_kernel, obs_calib .- obs_calib')

    for (i,cv_mask) in enumerate(cv_masks)
        B = (preds_calib_K[cv_mask,cv_mask] + λ * sum(cv_mask) * I) \ preds_calib_K[cv_mask, .!cv_mask]
        B = _batched_euclidean_simplex_proj(B)
        push!(B_cv, B)
    end

    errors = zeros(length(cv_masks), length(λ̃_grid))
    plt = plot(size = (1300, 600), legend = :outerright)
    n_cv_test = sum(.!cv_masks[1])
    # _ks_d = Distributions.quantile_bisect(Distributions.KSDist(n_cv_test), .95)
    # plot!(t -> cdf(Uniform(), t), ls = :dashdot, lab = "KS p = 0.05 (n = $n_cv_test)", 
    #     xlim = (-.05, 1.05), ribbon = _ks_d, fillalpha = .15)
        
    @showprogress desc = "λ̃ c.v." showspeed = true for (i,λ̃) in enumerate(λ̃_grid)
        for (j,(B, cv_mask)) in enumerate(zip(B_cv, cv_masks))
            G = _nonnegative_KDO_sol(λ̃, K_zz, K_zy_calib[:, cv_mask], B)
            Z = _PIT_transform(z, Q_kernel, G, obs_calib[.!cv_mask])

            errors[j,i] = _KS_error(Z) 
            # errors[j,i] = _CRPS_error(Z) 


            # plot!(Array(sort(Z)), (1:length(Z)) / length(Z), 
            #     lab = (j == 1) ? "log_10 λ̃ = $(round(log10(λ̃), digits = 3))" : "", 
            #     c = i+1)      
        end
    end
    # display(plt)

    _chose_lt(λ̃_grid, errors), errors
end


function _PIT_transform(z, Q_kernel, G, obs)
    diag(cdf(Q_kernel, obs .- z') * G)
end

_KS_error(Z) = HypothesisTests.ksstats(Z, Uniform())[2] #(n, δ, δp, δn)

function _CRPS_error(Z)
    Z = min.(max.(Z, zero(eltype(Z))), one(eltype(Z)))
    # _self = mean(abs.(Z .- Z'))
    _self = mae(Z)
    _cross = mean(1/2 .- (1 .- Z) .* Z)
    _cross - 0.5(_self + 1/3) # 𝔼|U - U'| = 1/3
end

function _chose_lt(λ̃_grid, λ̃_cv_errors)
    λ̃_grid = sort(λ̃_grid)
    me = vec(mean(λ̃_cv_errors, dims = 1))
    se = vec(std(λ̃_cv_errors, dims = 1))

    amin = argmin(me)
    σ = se[amin]
    amax_min = findlast(<(me[amin] + .1σ), me)
    @info "λ̃ cv" λ̃_grid[amin] λ̃_grid[amax_min] error_diff = me[amax_min] - me[amin]
    λ̃_grid[amax_min]
end





# # this should be a leave-one-out cross validation (since it has basicly the same computation needs as the n-fold cv.)

# using ProgressMeter

# # export get_cv_lambda

# function get_cv_lambda_tilde(
#         Q_kernel::Distribution,
#         # cv_masks::Vector{BitVector}, 
#         preds_calib_K::AbstractMatrix, 
#         z::AbstractVector,
#         obs_calib::AbstractVector,
#         K_zz::AbstractMatrix,
#         K_zy_calib::AbstractMatrix,
#         # recalpreds_calib_B::AbstractMatrix;
#         λ::Real,
#         λ̃_grid
#     )

#     # ------------------ calculate leave-one-out β ------------------
#     n_calib = size(preds_calib_K, 1)
#     B_loo = CUDA.zeros(n_calib-1, n_calib)
#     Q_loo = CUDA.zeros(Float64, size(K_zz, 1), n_calib)

#     @info "start l.o.o. β invert"
#     @showprogress desc = "l.o.o. β invert" showspeed = true for i = 1:n_calib
#         mask = BitVector(ones(n_calib))
#         mask[i] = 0

#         B_loo[:, i] .= (preds_calib_K[mask,mask] + λ * (n_calib - 1) * I) \ preds_calib_K[mask, i]
#         Q_loo[:, i] .= -K_zz * K_zy_calib[:, mask] * B_loo[:, i] .|> Float64
#     end
#     @info "finished l.o.o. β invert"



    
#     # ------------------ grid search for λ̃ ------------------
#     errors = Float64[]
#     G_lambda_tilde = Dict()
#     for λ̃ in λ̃_grid
#         try
#             @info "start cycle with" λ̃
#             G_lambda_tilde[λ̃] = KernelDensityOperator.optimize_for_G(
#                 λ̃, 
#                 Array{Float64}(K_zz), 
#                 Array{Float64}(K_zy_calib[:, 2:end]), 
#                 Array{Float64}(B_loo),
#                 Q_loo,
#                 )
#             err = mean_crps_laplace_from_gamma(z, obs_calib, G_lambda_tilde[λ̃], Q_kernel) |> mean
#             push!(errors, err)
#             @info "finished cycle with" λ̃ err
#         catch e
#             @warn "error during loo opt for λ̃=$λ̃" e
#             push!(errors, Inf) 
#         end
#     end
#     @info "finished l.o.o. cross validation opt. for λ̃" λ̃_grid errors
#     λ̃_grid[argmin(errors)], G_lambda_tilde
# end

