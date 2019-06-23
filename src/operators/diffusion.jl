@inline diffusive_flux_x(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat) =
    κ * AxF(i, j, k, grid) * δx_faa(i, j, k, Q)

@inline diffusive_flux_y(i, j, k, grid::Grid, Q::AbstractArray, κ:AbstractFloat) =
    κ * AyF(i, j, k, grid) * δy_afa(i, j, k, Q)

@inline diffusive_flux_z(i, j, k, grid::Grid, Q::AbstractArray, κ:AbstractFloat) =
    κ * Az(i, j, k ,grid) * δz_aaf(i, j, k, Q)

@inline δx_diffusive_flux(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat) =
    diffusive_flux_x(i+1, j, k, grid, Q, κ) - diffusive_flux_x(i, j, k, grid, Q, κ)

@inline δy_diffusive_flux(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat) =
    diffusive_flux_y(i, j+1, k, grid, Q, κ) - diffusive_flux_y(i, j, k, grid, Q, κ)

@inline function δz_diffusive_Flux(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    if k == grid.Nz
        return diffusive_flux_z(i, j, k, grid, Q, κ)
    else
        return diffusive_flux_z(i, j, k, grid, Q, κ) - diffusive_flux_z(i, j, k+1, grid, Q, κ)
    end
end

"""
    ∇κ∇Q(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)

Calculates diffusion for a tracer Q via

    1/V * [δx_caa(κ * Ax * δx_faa(Q)) + δy_aca(κ * Ay * δy_afa(Q)) + δz_aac(κ * Az * δz_aaf(Q))]

which will end up at the location `ccc`.
"""
@inline function ∇κ∇Q(i, j, k, grid::Grid, Q::AbstractArray, κ::AbstractFloat)
    1/V(i, j, k, grid) * (δx_diffusive_flux(i, j, k, grid, Q, κ) +
                          δy_diffusive_flux(i, j, k, grid, Q, κ) +
                          δz_diffusive_Flux(i, j, k, grid, Q, κ))
end
