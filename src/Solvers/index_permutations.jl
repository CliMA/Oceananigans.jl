# We never need to permute indices on the CPU.
@inline   permute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(solver_type, ::CPU, i, j, k, Nx, Ny, Nz) = i, j, k

"""
    _permute_index(i, N)

Perform the permutation [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]
on index `i` out of `N`.
"""
@inline function _permute_index(i, N)
    if (i & 1) == 1  # Same as isodd(i)
        return floor(Int, i/2) + 1
    else
        return N - floor(Int, (i-1)/2)
    end
end

"""
    _permute_index(i, N)

Undo the permutation [a, b, c, d, e, f, g, h] -> [a, c, e, g, h, f, d, b]
on index `i` out of `N`.
"""
@inline function _unpermute_index(i, N)
    if i <= N/2
        return 2i-1
    else
        return 2(N-i+1)
    end
end

@inline   permute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k
@inline unpermute_index(::TriplyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, k

@inline   permute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j,   _permute_index(k, Nz)
@inline unpermute_index(::HorizontallyPeriodic, ::GPU, i, j, k, Nx, Ny, Nz) = i, j, _unpermute_index(k, Nz)

@inline   permute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i,   _permute_index(j, Ny),   _permute_index(k, Nz)
@inline unpermute_index(::Channel, ::GPU, i, j, k, Nx, Ny, Nz) = i, _unpermute_index(j, Ny), _unpermute_index(k, Nz)
