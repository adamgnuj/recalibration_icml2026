# SKCE calibration test

using KernelFunctions, CalibrationTests, LinearAlgebra

struct PrecomputedSKCEKernel{T} <: Kernel where {T<:AbstractMatrix}
    Kq::T # k(q_i, q_j)
    Ly::T # l(y_i, y_j)
    Lqy::T # 𝔼[l(M_i, y_j)] (where M_i ∼ q_i)
    Lqq::T # 𝔼[l(M_i, M_j)] (where M_i ∼ q_i, M_j ∼ q_j)
end

import CalibrationTests: unsafe_skce_eval
function unsafe_skce_eval(k::PrecomputedSKCEKernel, qi::Int64, yi::Int64, qj::Int64, yj::Int64)
    k.Kq[qi, qj] * (k.Ly[yi, yj] - k.Lqy[qi, yj] - k.Lqy[qj, yi] + k.Lqq[qi, qj])
end


"""
```
LaplaceSKCETest(Q_kernel::Laplace, Q_kernel_distr::Distribution, z::AbstractVector, G::AbstractMatrix, obs::AbstractVector, obs_K::AbstractMatrix) -> AsymptoticSKCETest
```

Performs SKCE auto-calibration for predictions represented as a pdf in the form ``q_i(t) = ∑_j γ_j^{(i)} k(z_j, t)``, where ``z_j`` are fixed support points (`z`),
``k`` is the laplace kernel induced by `Q_kernel`, and ``γ_j^{(i)}`` are weights from `G[:, i]`. The observations are in `obs`,
and `obs_K` is simply the kernel matrix `pdf(Q_kernel, obs .- obs')`. 

`Q_kernel_distr` is a distribution inducing the RKHS (with kernel ``k_d``) over distributions. (``k_d`` has the form ``k_d(q_i, q_j) = κ(ℰ(q_i, q_j))``, where ``ℰ`` denotes the energy distance, and ``κ`` is the pdf of `Q_kernel_distr`.)
"""
function LaplaceSKCETest(Q_kernel::Laplace, Q_kernel_distr::Distribution, z::AbstractVector, G::AbstractMatrix, obs::AbstractVector, obs_K::AbstractMatrix)
    pairwise_mae_L = mae.(Q_kernel .+ z, Q_kernel .+ z')
    pairwise_k_L = _expectation_laplace_L_L.(Q_kernel, z, z')

    paiwise_mae_pred = G' * pairwise_mae_L * G
    paiwise_energy_distance_pred = paiwise_mae_pred - .5(diag(paiwise_mae_pred) .+ diag(paiwise_mae_pred)')
    preds_K = pdf(Q_kernel_distr, paiwise_energy_distance_pred)

    preds_k_MM = G' * pairwise_k_L * G
    pairwise_k_Ly = _expectation_laplace_y_L.(Q_kernel, z, obs')
    preds_obs_k_My = G' * pairwise_k_Ly
    prec_skce_kernel = PrecomputedSKCEKernel(preds_K, obs_K, preds_obs_k_My, preds_k_MM);
    AsymptoticSKCETest(prec_skce_kernel, 1:length(obs), 1:length(obs))
end

"""
```
EmpiricalSKCETest(y_obs::AbstractVector, y_train::AbstractVector, w::AbstractMatrix)
```
Perform the SKCE test using empirical distributions, based on support points
`y_train` and weights `w`, with observations `y_obs`. Also fit Laplace kernels to the energy disatances,
and observations via median heuristic.

`w` is the matrix of wights, where each row corresponds to a test point, 
and each column corresponds to the wights of `y_train`.
"""
function EmpiricalSKCETest(y_obs::AbstractVector, y_train::AbstractVector, w::AbstractMatrix)
    pw_dist_obs = abs.(y_obs .- y_obs')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel = Laplace(0, med_obs)
    K_ytest = pdf.(Q_kernel, y_obs .- y_obs')


    pw_mae = w * abs.(y_train .- y_train') * w'
    pw_edist = pw_mae - 0.5(diag(pw_mae) .+ diag(pw_mae)')
    med_pred = Calibration.median_distance(pw_edist)
    Q_kernel_pred = Laplace(0, med_pred)
    K_qtest = pdf.(Q_kernel_pred, pw_edist)

    # L_qy = w' * pdf.(Q_kernel, y_obs .- y_train') / size(y_train, 1)
    # L_qq = w' * pdf.(Q_kernel_pred, pw_edist) * w

    L_qy = w * pdf.(Q_kernel, y_train .- y_obs')
    L_qq = w * pdf.(Q_kernel, y_train .- y_train') * w';

    prec_skce_kernel = Calibration.PrecomputedSKCEKernel(K_qtest, K_ytest, L_qy, L_qq);
    Calibration.AsymptoticSKCETest(prec_skce_kernel, 1:length(y_obs), 1:length(y_obs))
end
