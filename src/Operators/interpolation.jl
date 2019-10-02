@inline ιx_caa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i, j, k+1])
@inline ιx_faa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j, k-1] + f[i,   j, k])

@inline ιy_aca(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j,   k] + f[i, j+1, k])
@inline ιy_afa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j-1, k] + f[i, j,   k])

@inline ιz_aac(i, j, k, grid::RegularCartesianGrid{T}, f) where T = @inbounds T(0.5) * (f[i, j,   k] + f[i+1, j, k])
@inline ιz_aaf(i, j, k, grid::RegularCartesianGrid{T}, f) where T = @inbounds T(0.5) * (f[i-1, j, k] + f[i, j,   k])

@inline ιz_aac(i, j, k, grid::VerticallyStretchedCartesianGrid, f) =
    @inbounds ((grid.zC[i] - grid.zF[i]) * f[i, j, k] + (grid.zF[i+1] - grid.zC[i]) * f[i+1, j, k]) / grid.ΔzF[i]
@inline ιz_aaf(i, j, k, grid::VerticallyStretchedCartesianGrid, f) =
    @inbounds ((grid.zF[i] - grid.zC[i-1]) * f[i-1, j, k] + (grid.zC[i] - grid.zF[i]) * f[i, j, k]) / grid.ΔzC[i-1]

