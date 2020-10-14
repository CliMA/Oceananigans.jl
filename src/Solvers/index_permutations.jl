# We never need to permute indices on the CPU.
@inline   permute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k

# For why we use Base.unsafe_trunc instead of floor below see:
# https://github.com/CliMA/Oceananigans.jl/issues/828
# https://github.com/CliMA/Oceananigans.jl/pull/997

"""
    _permute_index(i, N)

Permute `i` such that, for example, `i ∈ 1:N` becomes

    [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 8, 2, 7, 3, 6, 4, 5]

if `N=8`.
"""
@inline _permute_index(i, N)::Int = ifelse(isodd(i),
                                           Base.unsafe_trunc(Int, i/2) + 1,
                                           N - Base.unsafe_trunc(Int, (i-1)/2))

"""
    _unpermute_index(i, N)

Permute `i` in the opposite manner as `_permute_index`, such that,
for example, `i ∈ 1:N` becomes

   [1, 2, 3, 4, 5, 6, 7, 8] -> [1, 3, 5, 7, 8, 6, 4, 3]

if `N=8`.
"""
@inline _unpermute_index(i, N) = ifelse(i <= N/2, 2i-1, 2(N-i+1))

@inline   permute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k

@inline   permute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j,   _permute_index(k, Nz)
@inline unpermute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, _unpermute_index(k, Nz)

@inline   permute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i,   _permute_index(j, Ny),   _permute_index(k, Nz)
@inline unpermute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i, _unpermute_index(j, Ny), _unpermute_index(k, Nz)
