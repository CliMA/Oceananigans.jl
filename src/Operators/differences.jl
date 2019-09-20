@inline δx_caa(i, j, k, grid, f) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_faa(i, j, k, grid, f) = @inbounds f[i,   j, k] - f[i-1, j, k]

@inline δy_aca(i, j, k, grid, f) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_afa(i, j, k, grid, f) = @inbounds f[i, j,   k] - f[i, j-1, k]

@inline δz_aac(i, j, k, grid, f) = @inbounds f[i, j, k+1] - f[i, j,   k]
@inline δz_aaf(i, j, k, grid, f) = @inbounds f[i, j,   k] - f[i, j, k-1]

