# mean CRPS from gamma and z


function mean_crps_laplace_from_gamma(
    z::AbstractArray, 
    obs::AbstractArray, 
    G::AbstractMatrix, 
    Q_kernel::Distribution
    )
    
    G = Array{Float64}(G)
    z = Array{Float64}(z)
    obs = Array{Float64}(obs)
    G = Array{Float64}(G)

    pairwise_mae_L = mae.(Q_kernel .+ z, Q_kernel .+ z')
    mae_lapl_obs_all = mae.(Q_kernel .+ z, permutedims(Dirac.(Array(obs))))
    cross_mae_recal = sum(G .* mae_lapl_obs_all, dims = 1) |> vec
    self_mae_recal = sum(G .* (pairwise_mae_L * G), dims = 1) |> vec
    cross_mae_recal - .5self_mae_recal #CRPS
end

# mean nll laplace from gamma and z

function mean_nll_laplace_from_gamma(
    ε::Real,
    z::AbstractArray, 
    obs::AbstractArray, 
    G::AbstractMatrix, 
    Q_kernel::Distribution
    )

    density = sum(pdf(Q_kernel, z .- obs') .* G, dims = 1) |> vec
    density_ε = max.(density, zero(eltype(density))) * (1-ε) .+ ε * 1/(maximum(z) - minimum(z))
    nll = -log.(density_ε)
    (; density, density_ε, nll)
end