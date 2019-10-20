####
#### Difference operators
####

@inline δx_caa(i, j, k, grid, u) = @inbounds u[i+1, j, k] - u[i,   j, k]
@inline δx_faa(i, j, k, grid, c) = @inbounds c[i,   j, k] - c[i-1, j, k]

@inline δy_aca(i, j, k, grid, v) = @inbounds v[i, j+1, k] - v[i, j,   k]
@inline δy_afa(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j-1, k]

@inline δz_aac(i, j, k, grid, w) = @inbounds w[i, j, k+1] - w[i, j,   k]
@inline δz_aaf(i, j, k, grid, c) = @inbounds c[i, j,   k] - c[i, j, k-1]


# Difference operators of the form δ(A*f) where A is an area and f is an array.
@inline δxA_caa(i, j, k, grid, u) = @inbounds Ax(i+1, j, k, grid) * u[i+1, j, k] - Ax(i,   j, k, grid) * u[i,   j, k]
@inline δxA_faa(i, j, k, grid, c) = @inbounds Ax(i,   j, k, grid) * c[i,   j, k] - Ax(i-1, j, k, grid) * c[i-1, j, k]

@inline δyA_aca(i, j, k, grid, v) = @inbounds Ay(i, j+1, k, grid) * v[i, j+1, k] - Ay(i, j,   k, grid) * v[i, j,   k]
@inline δyA_afa(i, j, k, grid, c) = @inbounds Ay(i,   j, k, grid) * c[i, j,   k] - Ay(i, j-1, k, grid) * c[i, j-1, k]

@inline δzA_aac(i, j, k, grid, w) = @inbounds Az(i, j, k+1, grid) * w[i, j, k+1] - Az(i, j, k,   grid) * w[i, j,   k]
@inline δzA_aaf(i, j, k, grid, c) = @inbounds Az(i, j,   k, grid) * c[i, j,   k] - Az(i, j, k-1, grid) * c[i, j, k-1] 

