@inline ιx_caa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[k, j,   i] + f[k, j, i+1])
@inline ιx_faa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[k, j, i-1] + f[k, j,   i])

@inline ιy_aca(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[k, j,   i] + f[k, j+1, i])
@inline ιy_afa(i, j, k, grid::AbstractGrid{T}, f) where T = @inbounds T(0.5) * (f[k, j-1, i] + f[k, j,   i])

@inline ιz_aac(i, j, k, grid::RegularCartesianGrid{T}, f) where T = @inbounds T(0.5) * (f[k,   j, i] + f[k+1, j, i])
@inline ιz_aaf(i, j, k, grid::RegularCartesianGrid{T}, f) where T = @inbounds T(0.5) * (f[k-1, j, i] + f[k,   j, i])

@inline ιz_aac(i, j, k, grid::VerticallyStretchedCartesianGrid, f) =
    @inbounds ((grid.zC[k] - grid.zF[k]) * f[k, j, i] + (grid.zF[k+1] - grid.zC[k]) * f[k+1, j, i]) / grid.ΔzF[k]
@inline ιz_aaf(i, j, k, grid::VerticallyStretchedCartesianGrid, f) =
    @inbounds ((grid.zF[k] - grid.zC[k-1]) * f[k-1, j, i] + (grid.zC[k] - grid.zF[k]) * f[k, j, i]) / grid.ΔzC[k-1]

