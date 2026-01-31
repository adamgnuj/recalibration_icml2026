# gpu cross validatoin for kme

using Optim, ProgressMeter

# export get_cv_lambda

function _error_per_cv(λ, vecs, vals, cv_mask, preds_train_K, obs_train_K)
    _B = vecs * inv(Diagonal(vals) + λ * length(vals) * I) * vecs' * preds_train_K[cv_mask, .!cv_mask] #original
    _B = _batched_euclidean_simplex_proj(_B)
    # _B .= max.(_B, zero(eltype(_B))) # euclidean proj to \mathcal{H}^{+}_1
    # _B ./= sum(_B, dims = 1)
    # _B = (preds_train_K[cv_mask, cv_mask] + λ * length(vals) * I) \ preds_train_K[cv_mask, .!cv_mask] # not using eig.decomp.
  
    obs_self = tr(obs_train_K[.!cv_mask, .!cv_mask])
    cross = tr(_B' * obs_train_K[cv_mask, .!cv_mask])
    pred_self = tr(_B' * obs_train_K[cv_mask, cv_mask] * _B)
    (obs_self + pred_self - 2cross) / sum(.!cv_mask)
end

function _mean_cv_error(cv_masks, cv_eigens_preds_train_K, preds_train_K, obs_train_K, λ)
    n_cv = length(cv_masks)
    res = 0.0
    for i = 1:n_cv
        vals, vecs = cv_eigens_preds_train_K[i]
        cv_mask = cv_masks[i]
        res += _error_per_cv(Float32(λ), vecs, vals, cv_mask, preds_train_K, obs_train_K)
    end
    res / n_cv
end


"""
```
get_cv_lambda(cv_masks::Vector{BitVector}, preds_train_K::AbstractMatrix, obs_train_K::AbstractMatrix) -> λ
```

Optimise for best ``λ`` used for regularization in the conditional kernel mean embedding step. 

The `cv_masks` vectore stores the ``n``-fold cross-validation masks for the indices of the training data. 
(I.e. for ``5``-fold cv. `length(cv_masks) == 5` and `cv_masks[i]` is a `size(preds_train_K, 1)` length bitvector, 
with `true` entry at indices used for prediction at cross validation split `i`.)
`preds_train_K` is the kernel matrix for the predictions, `obs_train_K` is the kernel matrix for the observations.
"""
function get_cv_lambda(cv_masks::Vector{BitVector}, preds_train_K::AbstractMatrix, obs_train_K::AbstractMatrix)
    n_cv = length(cv_masks)
    # eigen decompositions
    _eig_first = eigen(preds_train_K[cv_masks[1], cv_masks[1]])
    cv_eigens_preds_train_K = Array{typeof(_eig_first)}(undef, n_cv)
    cv_eigens_preds_train_K[1] = _eig_first
    @showprogress desc = "eig. decomp for cross-validation" for i = 2:n_cv
        cv_eigens_preds_train_K[i] = eigen(preds_train_K[cv_masks[i], cv_masks[i]])
    end

    opt_res = optimize(l -> _mean_cv_error(cv_masks, cv_eigens_preds_train_K, preds_train_K, obs_train_K, l), 0.0, 10_000)
    opt_res.minimizer
end