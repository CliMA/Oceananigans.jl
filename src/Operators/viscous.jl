####
#### Viscous fluxes
####

@inline viscous_flux_ux(i, j, k, grid, u, ν_ccc) = ν_ccc * ℑFx_caa(i, j, k, grid) * δx_caa(i, j, k, grid, u)
@inline viscous_flux_uy(i, j, k, grid, u, ν_ffc) = ν_ffc * ℑFy_afa(i, j, k, grid) * δy_afa(i, j, k, grid, u)
@inline viscous_flux_uz(i, j, k, grid, u, ν_fcf) = ν_fcf * ℑFz_aaf(i, j, k, grid) * δz_aaf(i, j, k, grid, u)

@inline viscous_flux_vx(i, j, k, grid, v, ν_ffc) = ν_ffc * ℑFx_faa(i, j, k, grid) * δx_faa(i, j, k, grid, v)
@inline viscous_flux_vy(i, j, k, grid, v, ν_ccc) = ν_ccc * ℑFy_aca(i, j, k, grid) * δy_aca(i, j, k, grid, v)
@inline viscous_flux_vz(i, j, k, grid, v, ν_cff) = ν_cff * ℑFz_aaf(i, j, k, grid) * δz_aaf(i, j, k, grid, v)

@inline viscous_flux_wx(i, j, k, grid, w, ν_fcf) = ν_fcf * ℑFz_faa(i, j, k, grid) * δx_faa(i, j, k, grid, w)
@inline viscous_flux_wy(i, j, k, grid, w, ν_cff) = ν_cff * ℑFy_afa(i, j, k, grid) * δy_afa(i, j, k, grid, w)
@inline viscous_flux_wz(i, j, k, grid, w, ν_ccc) = ν_ccc * ℑFz_aac(i, j, k, grid) * δz_aac(i, j, k, grid, w)

####
#### Viscous dissipation operators
####

"""
    ∇ν∇u(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the u-velocity via

    1/V * [δx_faa(ν * ℑx_caa(Ax) * δx_caa(u)) + δy_aca(ν * ℑy_afa(Ay) * δy_afa(u)) + δz_aac(ν * ℑz_aaf(Az) * δz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function ∇ν∇u(i, j, k, grid, u, ν)
    1/Vᵘ(i, j, k, grid) * (δx_faa(i, j, k, grid, viscous_flux_ux, u, ν) +
                           δy_aca(i, j, k, grid, viscous_flux_uy, u, ν) +
                           δz_aac(i, j, k, grid, viscous_flux_uz, u, ν))
end

"""
    ∇ν∇v(i, j, k, grid::Grid, v::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δx_caa(ν * ℑx_faa(Ax) * δx_faa(v)) + δy_afa(ν * ℑy_aca(Ay) * δy_aca(v)) + δz_aac(ν * ℑz_aaf(Az) * δz_aaf(v))]

which will end up at the location `cfc`.
"""
@inline function ∇ν∇v(i, j, k, grid, v, ν)
    1/Vᵛ(i, j, k, grid) * (δx_caa(i, j, k, grid, viscous_flux_vx, v, ν) +
                           δy_afa(i, j, k, grid, viscous_flux_vy, v, ν) +
                           δz_aac(i, j, k, grid, viscous_flux_vz, v, ν))
end

"""
    ∇ν∇w(i, j, k, grid::Grid, w::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δx_caa(ν * ℑx_faa(Ax) * δx_faa(w)) + δy_aca(ν * ℑy_afa(Ay) * δy_afa(w)) + δz_aaf(ν * ℑz_aac(Az) * δz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function ∇ν∇w(i, j, k, grid, w, ν)
    1/Vʷ(i, j, k, grid) * (δx_caa(i, j, k, grid, viscous_flux_wx, w, ν) +
                           δy_aca(i, j, k, grid, viscous_flux_wy, w, ν) +
                           δz_aaf(i, j, k, grid, viscous_flux_wz, w, ν))
end

