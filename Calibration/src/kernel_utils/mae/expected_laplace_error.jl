# mean laplace error

using Distributions

"""
```
_expectation_laplace_L_L(Q_kernel::Laplace, zi::Real, zj::Real)
```
Computes 
```math
𝔼[k(Lⁱ, Lʲ)] = ∬_{ℝ^2} \\left(\\frac{1}{2θ}\\right)^3 e^{-|z_i - x|/θ} e^{-|z_i - y|/θ} e^{-|x - y|/θ} dxdy \\;.
```
"""
function _expectation_laplace_L_L(Q_kernel::Laplace, zi::Real, zj::Real)
    θ = Q_kernel.θ
    (3θ * (θ + abs(zi - zj)) + (zi - zj)^2) * exp(-abs(zi - zj) / θ) / 16θ^3
end

"""
```
_expectation_laplace_y_L(Q_kernel::Laplace, zi::Real, y::Real)
```
Computes 
```math
𝔼[k(Lⁱ, y)] = ∫_ℝ \\frac{1}{2θ} e^{-|z_i - t|/θ} \\frac{1}{2θ} e^{-|y - t|/θ} dt \\; .
```
"""
function _expectation_laplace_y_L(Q_kernel::Laplace, zi::Real, y::Real)
    θ = Q_kernel.θ
    d = abs(zi - y)/θ
    (1 + d) * exp(-d) / 4θ
end