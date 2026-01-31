# KernelDensityOperator
module KernelDensityOperator


using SparseArrays, Random, Distributions, LinearAlgebra, Clarabel, ProgressMeter, Printf
using CUDA, CUDA.CUSPARSE

const _T = Float64


"""
```
solver_setup(λ̃::Real, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, β::Vector{Float64}) -> solver
```

Sets up solver for 
``\\min_γ ||\\mathcal{C}_ρ u - μ||^2 + \\tilde{λ} ||u||^2, s.t. K_z γ ≥0``,
where ``u(t) = ∑_i γ_i k(z_i, t)`` is the reconstructed density.

Use [`KernelDensityOperator.update_solver_and_resolve!`](@ref) to reuse solver.
> Schuster et al. (2019) [Kernel Conditional Density Operators](https://arxiv.org/abs/1905.11255)
"""
function solver_setup(λ̃::Real, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, β::Vector{Float64};
    settings::Clarabel.Settings{Float64} = Clarabel.Settings(
            direct_solve_method = :cudss, verbose = false, presolve_enable = false, static_regularization_enable = false)
    )
    
    n_z = LinearAlgebra.checksquare(K_zz)
    T = Float64
    P = K_zz^3
    P .= .5(P + P') / n_z + λ̃ * K_zz
    q = -K_zz' * K_zy * β
    A = -K_zz
    b = zeros(n_z)
     
    solver = Clarabel.Solver{Float64}()
    cones = [Clarabel.NonnegativeConeT(n_z)]
    Clarabel.setup!(solver, sparse(P), q, sparse(A), b, cones, settings) 
    Clarabel.solve!(solver)
    solver
end


"""
```
update_solver_and_resolve!(solver::Clarabel.Solver{Float64}, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, β_new::Vector{Float64}) -> γ
```

Reuse the `solver` just update `β` to `β_new`.  
Use [`KernelDensityOperator.solver_setup`](@ref) for solver setup.
"""
function update_solver_and_resolve!(solver::Clarabel.Solver{Float64}, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, β_new::Vector{Float64})    
    q_new = -K_zz' * K_zy * β_new
    update_solver_and_resolve!(solver, CuArray{Float64, 1, CUDA.DeviceMemory}(q_new))
end

"""
```
update_solver_and_resolve!(solver::Clarabel.Solver{Float64}, q_new::CuArray{Float64, 1, CUDA.DeviceMemory}) -> γ
```
Optimized version to avoid cpu to gpu copy. Use the formula
`q_new = -K_zz' * K_zy * β_new`
to produce the vector `q_new`.
"""
function update_solver_and_resolve!(solver::Clarabel.Solver{Float64}, q_new::CuArray{Float64, 1, CUDA.DeviceMemory})    
    Clarabel.update_q!(solver, q_new)
    Clarabel.solve!(solver)
    solver.solution.status ∈ (Clarabel.SOLVED, Clarabel.ALMOST_SOLVED) || @error "QP solve error" solver.solution.status
    solver.solution.x / sum(solver.solution.x) #renormalize to have a density wrt. the Lebesgue base measure
end


"""
```
optimize_for_G(λ̃::Real, K_zz::AbstractMatrix, K_zy::AbstractMatrix, B::AbstractMatrix; showprogress = true) -> G
```
Convert arbitrary input arrays to `Array{Float64}` type, and calculate `Q`.
"""
function optimize_for_G(λ̃::Real, K_zz::AbstractMatrix, K_zy::AbstractMatrix, B::AbstractMatrix; showprogress = true)
    K_zz = Array{Float64}(K_zz)
    K_zy = Array{Float64}(K_zy)
    B = Array{Float64}(B)

    Q = CuArray{Float64, 2, CUDA.DeviceMemory}(-K_zz' * K_zy * B)
    optimize_for_G(λ̃, K_zz, K_zy, B, Q; showprogress = showprogress)
end

"""
```
optimize_for_G(λ̃::Real, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, B::Matrix{Float64}, Q::CuArray{Float64, 2, CUDA.DeviceMemory}; showprogress = true) -> G
```
Use `Q = CuArray{Float64, 2, CUDA.DeviceMemory}(-K_zz' * K_zy * B)`,
see [`KernelDensityOperator.solver_setup`](@ref) and [`KernelDensityOperator.update_solver_and_resolve!`](@ref) for other details.
"""
function optimize_for_G(λ̃::Real, K_zz::Matrix{Float64}, K_zy::Matrix{Float64}, B::Matrix{Float64}, Q::CuArray{Float64, 2, CUDA.DeviceMemory}; showprogress = true)
    n, n_z = size(B, 2), LinearAlgebra.checksquare(K_zz)
    G = CUDA.zeros(Float64, n_z, n)
    solver = solver_setup(λ̃, K_zz, K_zy, B[:, 1])    
    @showprogress showspeed=true desc = @sprintf("opt γ QP (λ̃=%1.1e)", λ̃) enabled = showprogress for i = 1:n
        G[:, i] .= update_solver_and_resolve!(solver, Q[:, i])
    end
    G
end

abstract type KDOConfiguration end

"""
```
struct MinMaxGrid <:  KDOConfiguration
    n_z :: Int64 
    d :: Real 
end
```
Find the smallest and largest calib obs, scale the range with `d` and
return the `n_z` element equidistant grid.
"""
struct MinMaxGrid <:  KDOConfiguration
    n_z::Int64 # number of grid points
    d::Real # >1 ratio of grid and sample range (max(z) - min(z)) / (max(y) - min(y))
end

function define_grid(
        conf::MinMaxGrid,
        calib_obs::CuArray{T, 1, CUDA.DeviceMemory}, 
        kernel::Distribution, 
        ) where {T<:Number}
    n_z = conf.n_z
    os = (conf.d-1)/2 * (maximum(calib_obs) - minimum(calib_obs))
    lb, ub = minimum(calib_obs) - os, maximum(calib_obs) + os
    z = LinRange(lb, ub, n_z) |> collect |> CuArray{_T}

    _kernel_matrices(z, calib_obs, kernel)
end

function _kernel_matrices(
        z::CuArray{T, 1, CUDA.DeviceMemory},
        calib_obs::CuArray{T, 1, CUDA.DeviceMemory}, 
        kernel::Distribution, 
        ) where {T<:Number}

    n_z = length(z)
    z_all_obs_l_pw_dist = abs.(z .- [z; calib_obs]')
    z_all_obs_K = pdf(kernel, z_all_obs_l_pw_dist)
    K_zz = z_all_obs_K[:, 1:n_z]
    K_zy_calib = z_all_obs_K[:, n_z+1:end]    
    (;z, K_zz, K_zy_calib)
end




end






















































































