# SKCE calibration test

using KernelFunctions, LinearAlgebra, Random

"""
Compute ``HKH`` inplace, where ``H = I - 1/n 1 1^T``
"""
function center!(K::AbstractMatrix)
    mK = sum(K, dims = 2) / size(K, 1)
    K .-= mK
    K .-= mK'
    K .+= mean(mK)
end

function _HSIC_test(K::AbstractMatrix, L::AbstractMatrix; m = 1000)
    
    n = LinearAlgebra.checksquare(K)
    center!(K)
    center!(L)
    HSIC = sum(K .* L) / n


    QH_0 = zeros(m)
    p = shuffle(1:n)
    for i = 1:m
        shuffle!(p)
        QH_0[i] = sum(K .* L[p, p]) / n
    end
    p_value = mean(QH_0 .> HSIC)
    p_value, HSIC
end

function LaplaceHSICTest(Q_kernel::Laplace, z::AbstractVector, G::AbstractMatrix, y_obs::AbstractVector)
    pw_dist_obs = abs.(y_obs .- y_obs')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel_obsfit = Laplace(0, med_obs)
    obs_K = pdf.(Q_kernel_obsfit, y_obs .- y_obs')
    
    pairwise_mae_L = mae.(Q_kernel .+ z, Q_kernel .+ z')

    paiwise_mae_pred = G' * pairwise_mae_L * G
    paiwise_energy_distance_pred = paiwise_mae_pred - .5(diag(paiwise_mae_pred) .+ diag(paiwise_mae_pred)')
    med_preds = Calibration.median_distance(paiwise_energy_distance_pred)
    Q_kernel_distrfit = Laplace(0, med_preds)

    preds_K = pdf(Q_kernel_distrfit, paiwise_energy_distance_pred)

    _HSIC_test(cu(preds_K), cu(obs_K))
end


function EmpiricalHSICTest(Z::AbstractVector, y_train::AbstractVector, w::AbstractMatrix)
    pw_dist_obs = abs.(Z .- Z')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel = Laplace(0, med_obs)
    K_ytest = pdf.(Q_kernel, Z .- Z')


    pw_mae = w * abs.(y_train .- y_train') * w'
    pw_edist = pw_mae - 0.5(diag(pw_mae) .+ diag(pw_mae)')
    med_pred = Calibration.median_distance(pw_edist)
    Q_kernel_pred = Laplace(0, med_pred)
    K_qtest = pdf.(Q_kernel_pred, pw_edist)

    _HSIC_test(cu(K_ytest), cu(K_qtest))
end

function _pw_mae_and_Z_HSICTest(Z::AbstractVector, pw_mae::AbstractMatrix)
    pw_dist_obs = abs.(Z .- Z')
    med_obs = Calibration.median_distance(pw_dist_obs)
    Q_kernel = Laplace(0, med_obs)
    K_ytest = pdf.(Q_kernel, Z .- Z')


    # pw_mae = w * abs.(y_train .- y_train') * w'
    pw_edist = pw_mae - 0.5(diag(pw_mae) .+ diag(pw_mae)')
    med_pred = Calibration.median_distance(pw_edist)
    Q_kernel_pred = Laplace(0, med_pred)
    K_qtest = pdf.(Q_kernel_pred, pw_edist)

    _HSIC_test(cu(K_ytest), cu(K_qtest))
end