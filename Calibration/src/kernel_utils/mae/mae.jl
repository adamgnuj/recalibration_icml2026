# mean absolute error for some distribtuions

export mae

using Distributions

mae(Q::Laplace, D::Dirac) = Q.θ * exp(-abs(Q.μ - D.value) / Q.θ) + abs(Q.μ - D.value)
mae(D::Dirac, Q::Laplace) = mae(Q, D)
function mae(Q1::Laplace{T}, Q2::Laplace{T}) where {T}
    Q1.θ ≈ Q2.θ || error("mae unimplemented for Laplace with different variances \n$((;Q1, Q2))")
    (3*one(T)/2 * Q1.θ + one(T)/2 * abs(Q1.μ - Q2.μ)) * exp(-abs(Q1.μ - Q2.μ) / Q1.θ) + abs(Q1.μ - Q2.μ)
end



const _GaussianMixture = MixtureModel{Univariate, Continuous, Normal{Float64}, 
                                     Categorical{Float64, Vector{Float64}}}

"""
```
mae(P::_GaussianMixture, Q::_GaussianMixture)
```

Compute 𝔼|X - Y| for X ∼ P, Y ∼ Q in 𝒪(1)
"""
function mae(P::_GaussianMixture, Q::_GaussianMixture)
    A(μ, σ) = μ == σ == 0 ? 0 : μ*(2cdf(Normal(), μ / σ) - 1) + 2σ*pdf(Normal(), μ / σ)

    w_p = P.prior.p
    μ_p = getfield.(P.components, :μ)
    σ_p = getfield.(P.components, :σ)
    
    w_q = Q.prior.p
    μ_q = getfield.(Q.components, :μ)
    σ_q = getfield.(Q.components, :σ)
    
    M = μ_p .- μ_q'
    S = sqrt.(σ_p.^2 .+ (σ_q.^2)')
    w_p' * A.(M, S) * w_q
end

mae(P::Normal, Q::_GaussianMixture) = mae(MixtureModel([P]), Q)
mae(P::_GaussianMixture, Q::Normal) = mae(Q, P)
mae(P::Normal, Q::Normal) = mae(MixtureModel([P]), MixtureModel([Q]))

mae(P::Normal, Q::Dirac) = mae(P, Normal(Q.value, 0.0))
mae(P::Dirac, Q::Normal) = mae(Q, P)
mae(P::Dirac, Q::Dirac) = abs(Q.value - P.value)

mae(P::_GaussianMixture, Q::Dirac) = mae(P, Normal(Q.value, 0.0))
mae(P::Dirac, Q::_GaussianMixture) = mae(Q, P)


using Tullio, KernelAbstractions
function batched_gaussian_pairwise_mae(L, V, P)
    Φ(t::Real) = cdf(Normal{typeof(t)}(zero(t), one(t)), t)
    φ(t::Real) = pdf(Normal{typeof(t)}(zero(t), one(t)), t)
    A(μ::Real, σ::Real) = μ*(2Φ(μ / σ) - 1) + 2σ*φ(μ / σ)

    @tullio S[i,j] := P[i, k] * P[j, l] * A(L[i, k] - L[j, l], √(V[i,k] + V[j,l]))
    S[isnan.(S)] .= zero(eltype(S)) # fix for mae(Normal(μ, 0), Normal(μ, 0) ) = 0
    S
end

function batched_gaussian_self_mae(L, V, P)
    Φ(t::Real) = cdf(Normal{typeof(t)}(zero(t), one(t)), t)
    φ(t::Real) = pdf(Normal{typeof(t)}(zero(t), one(t)), t)
    A(μ::Real, σ::Real) = μ*(2Φ(μ / σ) - 1) + 2σ*φ(μ / σ)

    @tullio S[i] := P[i, k] * P[i, l] * A(L[i, k] - L[i, l], √(V[i,k] + V[i,l]))
    S
end

function batched_gaussian_obs_mae(L, V, P, Y)
    Φ(t::Real) = cdf(Normal{typeof(t)}(zero(t), one(t)), t)
    φ(t::Real) = pdf(Normal{typeof(t)}(zero(t), one(t)), t)
    A(μ::Real, σ::Real) = μ*(2Φ(μ / σ) - 1) + 2σ*φ(μ / σ) # 𝔼|𝒩(μ,σ)|

    @tullio S[i] := P[i, k] * A(L[i, k] - Y[i], √V[i,k])
    S
end