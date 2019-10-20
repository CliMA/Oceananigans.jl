# Calculate diffusive fluxes.
""" Calculate κ * AxF * δx_faa(Q) -> fcc. """
@inline diffusive_flux_x(i, j, k, grid::Grid, Q::AbstractArray, κ_fcc::AbstractFloat) =
    κ_fcc * AxF(i, j, k, grid) * δx_faa(i, j, k, Q)

""" Calculate κ * AyF * δy_afa(Q) -> cfc. """
@inline diffusive_flux_y(i, j, k, grid::Grid, Q::AbstractArray, κ_cfc:AbstractFloat) =
    κ_cfc * AyF(i, j, k, grid) * δy_afa(i, j, k, Q)

""" Calculate κ * Az * δz_aaf(Q) -> ccf. """
@inline diffusive_flux_z(i, j, k, grid::Grid, Q::AbstractArray, κ_ccf:AbstractFloat) =
    κ_ccf * Az(i, j, k ,grid) * δz_aaf(i, j, k, Q)

""" Calculate δx_caa[κ * AxF * δx_faa(Q)] -> ccc. """
@inline δx_diffusive_flux(i, j, k, grid::Grid, Q::AbstractArray, κ_fcc::AbstractFloat) =
    diffusive_flux_x(i+1, j, k, grid, Q, κ_fcc) - diffusive_flux_x(i, j, k, grid, Q, κ_fcc)

""" Calculate δy_aca[κ * AyF * δy_afa(Q)] -> ccc. """
@inline δy_diffusive_flux(i, j, k, grid::Grid, Q::AbstractArray, κ_cfc::AbstractFloat) =
    diffusive_flux_y(i, j+1, k, grid, Q, κ_cfc) - diffusive_flux_y(i, j, k, grid, Q, κ_cfc)

""" Calculate δz_aac[κ * Az * δz_aaf(Q)] -> ccc. """
@inline function δz_diffusive_Flux(i, j, k, grid::Grid, Q::AbstractArray, κ_ccf::AbstractFloat)
    if k == grid.Nz
        return diffusive_flux_z(i, j, k, grid, Q, κ_ccf)
    else
        return diffusive_flux_z(i, j, k, grid, Q, κ_ccf) - diffusive_flux_z(i, j, k+1, grid, Q, κ_ccf)
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
