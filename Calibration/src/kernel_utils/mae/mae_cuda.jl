#mae CUDA
using CUDA

export pairwise_mae


function _reconstruct_indices(l)
    i = 1 + Int32(ceil((sqrt(1 + 8l) - 1) / 2)) # row in lower tri
    j = l - (i-1)*(i-2)÷2 # column
    i,j
end

function _self_sae(M, i)
    acc = zero(eltype(M))
    m = size(M, 1)
    for s = 1:m
        acc += M[s, i] * (2s - m - 1)
    end
    2acc
end

"""
```
_M_binsearch(M, val, i, p_min, p_max) -> p
```
Binary search for `p` in `[p_min, p_max]` such that `M[p, i] <= val < M[p+1, i]`.
"""
function _M_binsearch(M, val, i, p_min, p_max)
    if M[p_max, i] <= val # this ensures 𝒪(1) for "disjoint" samples
        return p_max
    elseif M[p_min, i] > val
        return p_min -1 
    end
    while true
        p = Int32(floor((p_min + p_max) / 2))
        if p == p_min
            return p
        end
        if M[p, i] <= val
            p_min = p
        else
            p_max = p
        end
    end
end

function _linear_merge_intermediate_sae(M, i, j, r, s)
    m = size(M, 1)
    acc = zero(eltype(M))
    while (r <= m) && (s <= m)
        if (M[r, i] < M[s, j])
            acc += M[r, i] * (2s - 2 - m)
            r += 1
        else
            acc += M[s, j] * (2r - 2 - m)
            s += 1
        end
    end
    acc, r, s
end

function _first_step_binsearch_sae(M, M_cumsum, i, j)
    m = size(M, 1)
    acc = zero(eltype(M))
    r,s = 1,1
    if M[r, i] < M[s, j]
        p = _M_binsearch(M, M[s, j], i, r, m)
        acc -= M_cumsum[p, i] * (m - s + 1) 
        # acc += M_cumsum[end, j] * (p - r + 1)
        r = p+1
    else # M[s, j] < M[r, i] same logic:
        p = _M_binsearch(M, M[r, i], j, s, m)
        acc -= M_cumsum[p, j] * (m - r + 1) 
        # acc += M_cumsum[end, i] * (p - s + 1)
        s = p+1
    end
    acc, r, s
end

function _finish_cums_sae(M, M_cumsum, i, j, r, s)
    m = size(M, 1)
    acc = zero(eltype(M))
    if r == m+1
        acc += (M_cumsum[m, j] - (s == 1 ? 0 : M_cumsum[s-1, j])) * m
    else
        acc += (M_cumsum[m, i] - (r == 1 ? 0 : M_cumsum[r-1, i])) * m
    end
    acc
end

function _merge_bin_lin_cums_sae(M, M_cumsum, i, j)
    acc_1, r, s = _first_step_binsearch_sae(M, M_cumsum, i, j)
    acc_2, r, s = _linear_merge_intermediate_sae(M, i, j, r, s)
    acc_3 = _finish_cums_sae(M, M_cumsum, i, j, r, s)
    acc_1 + acc_2 + acc_3
end


function kernel_mae!(M::CuDeviceMatrix{Float32, 1}, M_cumsum::CuDeviceMatrix{Float32, 1}, out::CuDeviceMatrix{Float32, 1})    
    m, n = size(M)
    T = n*(n-1)÷2
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x #linear index [1, ..., T] on lower triangular of the kmtx
    
    if l <= T # offdiagonal entries
        i,j = _reconstruct_indices(l)
        S = _merge_bin_lin_cums_sae(M, M_cumsum, i, j)
        S /= m^2
        out[i,j] = S
        out[j,i] = S
        return
    elseif l <= T + n #diagonal entries at the end
        i = l - T
        # out[i,i] = _self_sae(M, i) / (m * (m-1))
        out[i,i] = _self_sae(M, i) / m^2
        return
    else
        return
    end
end


"""
```
pairwise_mae(M::AbstractArray) -> S
```
Compute pairwise mean absolute error for samples in `M`.

`` M ∈ ℝ^{m × n}`` and ``S[i,j] = 1/m^2 ∑_{s,r = 1}^m |M[s, i] - M[r, j]|``.

Executed on GPU, with computational complexity 𝒪(nmlog(m) + n²m) and memory complexity 𝒪(mn²).
"""
function pairwise_mae(M::AbstractArray)
    n = size(M, 2)
    
    M = M .|> Float32 |> cu
    sort!(M, dims = 1)
    M_cumsum = cumsum(M, dims = 1)
    out = CUDA.zeros(n,n)
    
    T = n*(n-1)÷2
    steps = T + n
    
    ts = min(256, n)
    bs = cld(steps, ts)

    CUDA.@sync @cuda threads=ts blocks=bs kernel_mae!(M, M_cumsum, out)
    out
end