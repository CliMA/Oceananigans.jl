@inline ϊx_caa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i+1, j, k])
@inline ϊx_faa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i-1, j, k] + f[i,   j, k])

@inline ϊy_aca(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j,   k] + f[i, j+1, k])
@inline ϊy_afa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j-1, k] + f[i, j,   k])

@inline ϊz_aac(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j,   k] + f[i, j, k+1])
@inline ϊz_aaf(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j, k-1] + f[i, j,   k])

