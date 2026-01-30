# run with julia --project=../../ script.jl 
@info "env:" Base.active_project()

using DataFrames, CSV, Distributions
using Calibration


"""
```
_PIT_recal_weighted_empirical(y_train::AbstractVector, y_cal::AbstractVector, W_cal::AbstractMatrix, W_test::AbstractMatrix)
```
PIT recalibration of a weighted empirical CDFs.
- `W_cal ∈ ℝ^{n × m}` where `n` is the calibration set size, `m` is the train set size.
- `W_test ∈ ℝ^{N × m}` where `N` is the test set size.

out: W_recal_test using PIT calibraion
"""
function _PIT_recal_weighted_empirical(y_train::AbstractVector, y_cal::AbstractVector, W_cal::AbstractMatrix, W_test::AbstractMatrix)

    preds_Q_cal = [Calibration.Empirical(y_train, w) for w in eachrow(W_cal)]
    Z_cal = cdf.(preds_Q_cal, y_cal)
    Q̂_Z = Calibration.Empirical(Z_cal)
    
    # compose
    sp = sortperm(y_train)
    sp_inv = sortperm(sp)
    W_scs = cumsum(W_test[:, sp], dims = 2) # these are the CDF values at the jumps
    W_sr = cdf(Q̂_Z, W_scs) # recalibrate
    # compute individual weights from cdf values
    W = W_sr[:, 1:end] - [zeros(eltype(W_sr), size(W_sr, 1)) W_sr[:, 1:end-1]]
    W_recal_test = W[:, sp_inv] # invert sorting
    W_recal_test ./= sum(W_recal_test, dims = 2) # fix numerical errors
end


# recalibration with inverse cdf sampling

import Distributions: cdf, quantile
struct _PIT_recalibrated <: Distributions.ContinuousUnivariateDistribution
    Qc::Distributions.ContinuousUnivariateDistribution
    Qb::Distributions.ContinuousUnivariateDistribution
end
cdf(Q::_PIT_recalibrated, t::Real) = cdf(Q.Qc, cdf(Q.Qb, t))
function quantile(Q::_PIT_recalibrated, p::Real)
    eps = 1e-9
    m = min(quantile(Q.Qb, eps), quantile(Q.Qc, eps))
    M = max(quantile(Q.Qb, 1-eps), quantile(Q.Qc, 1-eps))
    d = M - m
    Distributions.quantile_bisect(Q, p, m -5d, M + 5d)
end



begin
    args_parsed = NamedTuple{(
        :X_train, :X_validation, :X_test, 
        :y_train, :y_validation, 
        :preds_train, :preds_validation, :preds_test,
        :preds_format, :recal_preds_test_out)}(ARGS)


    y_train = CSV.read(args_parsed.y_train, DataFrame) |> Matrix |> vec .|> Float64
    y_valid = CSV.read(args_parsed.y_validation, DataFrame) |> Matrix |> vec .|> Float64


    PREDS_FORMAT = first(readlines(args_parsed.preds_format))
    @info "parsed:" PREDS_FORMAT

    if PREDS_FORMAT == "WeightedFixSampleEmpirical"
        preds_W_train = CSV.read(args_parsed.preds_train, DataFrame) |> Matrix
        preds_W_valid = CSV.read(args_parsed.preds_validation, DataFrame) |> Matrix
        preds_W_test = CSV.read(args_parsed.preds_test, DataFrame) |> Matrix

        preds_repr_train = (preds_W_train, y_train)
        preds_repr_valid = (preds_W_valid, y_train)
        preds_repr_test = (preds_W_test, y_train)

        recal_preds_W_test = _PIT_recal_weighted_empirical(
            y_train, y_valid, preds_W_valid, preds_W_test)

        CSV.write(joinpath(args_parsed.recal_preds_test_out, "recal_test.csv"), DataFrame(recal_preds_W_test, :auto))
        @info "PIT recal preds saved to" args_parsed.recal_preds_test_out
    
    elseif PREDS_FORMAT == "gaussianmixture"
        # preds_train = CSV.read(args_parsed.preds_train, DataFrame) |> Matrix
        preds_valid = CSV.read(args_parsed.preds_validation, DataFrame) |> Matrix
        preds_test = CSV.read(args_parsed.preds_test, DataFrame) |> Matrix


        nc = size(preds_test, 2) ÷ 3
        preds_repr_test = (preds_test[:, 1:nc], preds_test[:, nc+1:2nc], preds_test[:, 2nc+1:end])
        L, V, P = preds_repr_test
        P ./= sum(P, dims = 2)
        Q_test = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]

        nc = size(preds_valid, 2) ÷ 3
        preds_repr_valid = (preds_valid[:, 1:nc], preds_valid[:, nc+1:2nc], preds_valid[:, 2nc+1:end])
        L, V, P = preds_repr_valid
        P ./= sum(P, dims = 2)
        Q_valid = [MixtureModel([Normal(m, sqrt(v)) for (m,v) in zip(mm, vv)], collect(p)) 
                    for (mm, vv, p) in zip(eachrow(L), eachrow(V), eachrow(P))]
        
        Z_valid = cdf.(Q_valid, y_valid)
        Q_Z_valid = Calibration.Empirical(Z_valid)

        ns = 1000
        Q_recal_test_sample = [rand(_PIT_recalibrated(Q_Z_valid, q), ns) for q in Q_test]
        df_samples = DataFrame(hcat(Q_recal_test_sample...)', :auto)
        CSV.write(joinpath(args_parsed.recal_preds_test_out, "recal_test.csv"), df_samples)
        @info "PIT recal preds saved to" args_parsed.recal_preds_test_out

    else
        error("Not implemented PREDS_FORMAT: ", PREDS_FORMAT)
    end
end