# These kernels run on GPUs, some of which (e.g. Metal) cannot compile Float64
# arithmetic, so index computations must stay in integer arithmetic throughout:
# i ÷ 2 == trunc(i/2) and (N + 1) ÷ 2 == ceil(N/2) for the i, N ≥ 0 used here.

"""
$(TYPEDSIGNATURES)

Permute `i` such that, for example, `i ∈ 1:N` becomes

    [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 8, 2, 7, 3, 6, 4, 5]

    [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 9, 2, 8, 3, 7, 4, 6, 5]

for `N=8` and `N=9` respectively.

See equation (20) of [Makhoul80](@citet).
"""
@inline permute_index(i, N)::Int = ifelse(isodd(i),
                                          i ÷ 2 + 1,
                                          N - (i - 1) ÷ 2)

#####
##### The GPU cosine transforms work in the buffer, which holds the transformed dimension
##### permuted and, for the second dimension, transposed into the first
#####

@inline transformed_index(i, j, k, ::Val{1}) = i
@inline transformed_index(i, j, k, ::Val{2}) = j
@inline transformed_index(i, j, k, ::Val{3}) = k

@inline buffer_indices(i, j, k, n, ::Val{1}) = (n, j, k)
@inline buffer_indices(i, j, k, n, ::Val{2}) = (n, i, k)
@inline buffer_indices(i, j, k, n, ::Val{3}) = (i, j, n)

transform_buffer(buffer, grid, dim) = dim == 2 ? reshape(buffer, size(grid, 2), size(grid, 1), size(grid, 3)) : buffer

@kernel function _permute_indices!(B, A, dim, N)
    i, j, k = @index(Global, NTuple)
    n = permute_index(transformed_index(i, j, k, dim), N)
    @inbounds B[buffer_indices(i, j, k, n, dim)...] = A[i, j, k]
end

@kernel function _unpermute_indices!(A, B, dim, N)
    i, j, k = @index(Global, NTuple)
    n = permute_index(transformed_index(i, j, k, dim), N)
    @inbounds A[i, j, k] = real(B[buffer_indices(i, j, k, n, dim)...])
end

@kernel function _twiddle_forward!(A, B, ω, dim)
    i, j, k = @index(Global, NTuple)
    n = transformed_index(i, j, k, dim)
    @inbounds A[i, j, k] = 2 * real(ω[n] * B[buffer_indices(i, j, k, n, dim)...])
end

@kernel function _twiddle_backward!(B, A, ω, dim)
    i, j, k = @index(Global, NTuple)
    n = transformed_index(i, j, k, dim)
    @inbounds B[buffer_indices(i, j, k, n, dim)...] = ω[n] * A[i, j, k]
end
