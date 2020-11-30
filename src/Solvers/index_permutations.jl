# For why we use Base.unsafe_trunc instead of floor below see:
# https://github.com/CliMA/Oceananigans.jl/issues/828
# https://github.com/CliMA/Oceananigans.jl/pull/997

"""
    _permute_index(i, N)

Permute `i` such that, for example, `i ∈ 1:N` becomes

    [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 8, 2, 7, 3, 6, 4, 5]

    [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 9, 2, 8, 3, 7, 4, 6, 5]

for `N=8` and `N=9` respectively.
"""
@inline _permute_index(i, N)::Int = ifelse(isodd(i),
                                           Base.unsafe_trunc(Int, i/2) + 1,
                                           N - Base.unsafe_trunc(Int, (i-1)/2))

"""
    _unpermute_index(i, N)

Permute `i` in the opposite manner as `_permute_index`, such that,
for example, `i ∈ 1:N` becomes

   [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 3, 5, 7, 8, 6, 4, 2]

   [1, 2, 3, 4, 5, 6, 7, 8, 9] -> [1, 3, 5, 7, 9, 8, 6, 4, 2]

for `N=8` and `N=9` respectively.
"""
@inline _unpermute_index(i, N) = ifelse(i <= ceil(N/2), 2i-1, 2(N-i+1))

# Fallback (do not permute)
@inline permute_index(arch, topo, i, N) = i
@inline unpermute_index(arch, topo, i, N) = i

# Only need to permute along bounded dimensions on GPUs.
@inline permute_index(::GPU, ::Bounded, i, N) = _permute_index(i, N)
@inline unpermute_index(::GPU, ::Bounded, i, N) = _unpermute_index(i, N)
