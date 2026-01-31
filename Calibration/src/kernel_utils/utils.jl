# utils

include("mae/mae_cuda.jl")
include("mae/mae.jl")
include("mae/expected_laplace_error.jl")

using LinearAlgebra, StatsBase

"""
```
median_distance(D::AbstractMatrix)
```
return the median distance based on distance matrix `D`
"""
median_distance(D::AbstractMatrix) = median(D[triu!(trues(size(D)), 1)])

"""
```julia
_batched_euclidean_simplex_proj(V::CuArray{T, 2, CUDA.DeviceMemory}) where {T <: Real}
```
Euclidean projection of every column of `V` to the probability simplex.

> Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions
"""
function _batched_euclidean_simplex_proj(V::CuArray{T, 2, CUDA.DeviceMemory}) where {T <: Real}
    M = sort(V, dims = 1; rev = true) # μ₁ ≥ … ≥ μₚ
    csM = cumsum(M, dims = 1)
    n = size(V, 1)
    R = M - (1 ./(1:n)) .* (csM .- 1)
    _Inds = (1:n) .* CUDA.ones(Int, size(R, 2))'
    _Inds[R .< 0] .= -1

    _ρ_cind = argmax(_Inds, dims = 1) # returns an Array of CartesianIndex
    _ρ_vind = maximum(_Inds, dims = 1) # returns an Array of Int64 indices

    Θ = 1 ./ _ρ_vind .* (csM[_ρ_cind] .- 1) # θ
    W = max.(V .- Θ, zero(eltype(V)))
end


function _clip_and_renormalize_ckme_weights(B::CuArray{T, 2, CUDA.DeviceMemory}) where {T <: Real}
    B[B .< zero(eltype(B))] .= zero(eltype(B))
    B ./= sum(B, dims = 1)
end

"""
```julia
mae(V::AbstractVector)
```
Compute ``1/n^2 ∑|V_i - V_j|`` in 𝒪(n log(n)).
"""
function mae(V::AbstractVector)
    n = length(V)
    sp = issorted(V) ? (1:n) : sortperm(V)
    2sum(V[sp] .* (2 * (1:n) .- n .- 1)) / n^2
end

import Distributions: cdf, quantile, minimum, maximum, mean, var

struct Empirical <: Distributions.ContinuousUnivariateDistribution
    F::StatsBase.ECDF
end

Empirical(z::AbstractVector{<:Real}) = Empirical(StatsBase.ecdf(z))
Empirical(z::AbstractVector{<:Real}, w::AbstractVector) = Empirical(StatsBase.ecdf(z; weights = w))
cdf(Q::Empirical, z::Real) = Q.F(z)
cdf(Q::Empirical, zz::AbstractVector{<:Real}) = Q.F(zz)
minimum(Q::Empirical) = minimum(Q.F)
maximum(Q::Empirical) = maximum(Q.F)
mean(Q::Empirical) = isempty(Q.F.weights) ? mean(Q.F.sorted_values) : Q.F.weights' * Q.F.sorted_values
var(Q::Empirical) = (isempty(Q.F.weights) ? mean(Q.F.sorted_values.^2) : Q.F.weights' * Q.F.sorted_values.^2) - mean(Q)^2
quantile(Q::Empirical, p::Real) = Distributions.quantile_bisect(Q, p, minimum(Q) - eps(), maximum(Q) + eps())
Base.show(io::IO, Q::Empirical) = print(io, "Empirical(n=$(length(Q.F.sorted_values)), 𝔼=$(round(mean(Q), digits = 2)), √𝕍=$(round(std(Q), digits = 2)))")