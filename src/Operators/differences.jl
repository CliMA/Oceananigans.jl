@inline δx_caa(i, j, k, grid, f) = @inbounds f[k, j, i+1] - f[k, j,   i]
@inline δx_faa(i, j, k, grid, f) = @inbounds f[k, j,   i] - f[k, j, i-1]

@inline δy_aca(i, j, k, grid, f) = @inbounds f[k, j+1, i] - f[k, j,   i]
@inline δy_afa(i, j, k, grid, f) = @inbounds f[k, j,   i] - f[k, j-1, i]

@inline δz_aac(i, j, k, grid, f) = @inbounds f[k+1, j, i] - f[k,   j, i]
@inline δz_aaf(i, j, k, grid, f) = @inbounds f[k,   j, i] - f[k-1, j, i]

@inline ∂x_caa(i, j, k, grid, f) = δx_caa(i, j, k, grid, f) / grid.Δx
@inline ∂x_faa(i, j, k, grid, f) = δx_faa(i, j, k, grid, f) / grid.Δx

@inline ∂y_aca(i, j, k, grid, f) = δy_caa(i, j, k, grid, f) / grid.Δy
@inline ∂y_afa(i, j, k, grid, f) = δy_faa(i, j, k, grid, f) / grid.Δy

@inline ∂z_aac(i, j, k, grid::RegularCartesianGrid, f) = δz_aac(i, j, k, grid, f) / grid.Δz
@inline ∂z_aaf(i, j, k, grid::RegularCartesianGrid, f) = δz_aaf(i, j, k, grid, f) / grid.Δz

@inline ∂z_aac(i, j, k, grid::VerticallyStretchedCartesianGrid, f) = δz_aac(i, j, k, grid, f) / grid.ΔzF[k]
@inline ∂z_aaf(i, j, k, grid::VerticallyStretchedCartesianGrid, f) = δz_aaf(i, j, k, grid, f) / grid.ΔzC[k]

