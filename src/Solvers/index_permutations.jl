# We never need to permute indices on the CPU.
@inline   permute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k

"""
    _permute_index(i, N)

Perform the permutation [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]
on index `i` out of `N`.
"""
@inline _permute_index(i, N)::Int = ifelse(isodd(i), floor(i/2) + 1, N - floor((i-1)/2))

"""
    _permute_index(i, N)

Undo the permutation [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]
on index `i` out of `N`.
"""
@inline _unpermute_index(i, N) = ifelse(i <= N/2, 2i-1, 2(N-i+1))

@inline   permute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k

@inline   permute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j,   _permute_index(k, Nz)
@inline unpermute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, _unpermute_index(k, Nz)

@inline   permute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i,   _permute_index(j, Ny),   _permute_index(k, Nz)
@inline unpermute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i, _unpermute_index(j, Ny), _unpermute_index(k, Nz)
