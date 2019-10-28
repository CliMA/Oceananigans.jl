####
#### Diffusive fluxes
####

@inline diffusive_flux_x(i, j, k, grid, c, κ_fcc) = κ_fcc * AxF(i, j, k, grid) * δx_faa(i, j, k, grid, c)
@inline diffusive_flux_y(i, j, k, grid, c, κ_cfc) = κ_cfc * AyF(i, j, k, grid) * δy_afa(i, j, k, grid, c)
@inline diffusive_flux_z(i, j, k, grid, c, κ_ccf) = κ_ccf *  Az(i, j, k, grid) * δz_aaf(i, j, k, grid, c)

####
#### Laplacian diffusion operator
####

"""
    ∇κ∇c(i, j, k, grid, c, κ)

Calculates diffusion for a tracer c via

    1/V * [δx_caa(κ * Ax * δx_faa(c)) + δy_aca(κ * Ay * δy_afa(c)) + δz_aac(κ * Az * δz_aaf(c))]

which will end up at the location `ccc`.
"""
@inline function ∇κ∇c(i, j, k, grid, κ, c)
    1/VC(i, j, k, grid) * (δx_caa(i, j, k, grid, diffusive_flux_x, c, κ) +
                           δy_aca(i, j, k, grid, diffusive_flux_y, c, κ) +
                           δz_aac(i, j, k, grid, diffusive_flux_z, c, κ))
end
