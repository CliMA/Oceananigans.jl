# Calculate viscous fluxes for u-velocity field.
""" Calculate ν * (Ax)ˣ * δx_caa(u) -> ccc. """
@inline viscous_flux_ux(i, j, k, grid::Grid, u::AbstractArray, ν_ccc::AbstractFloat) =
    ν_ccc * ϊAxF_caa(i, j, k, grid) * δx_caa(i, j, k, grid, u)

""" Calculate ν * (Ay)ʸ * δy_afa(u) -> ffc. """
@inline viscous_flux_uy(i, j, k, grid::Grid, u::AbstractArray, ν_ffc::AbstractFloat) =
    ν_ffc * ϊAyF_afa(i, j, k, grid) * δy_afa(i, j, k, grid, u)

# Calculate the components of the divergence of the u-viscous flux.
""" Calculate ν * (Az)ᶻ * δz_aaf(u) -> fcf. """
@inline viscous_flux_uz(i, j, k, grid::Grid, u::AbstractArray, ν_fcf::AbstractFloat) =
    ν_fcf * ϊAz_aaf(i, j, k, grid) * δz_aaf(i, j, k, grid, u)

""" Calculate δx_faa[ν * (Ax)ˣ * δx_caa(u)] -> fcc. """
@inline δx_viscous_flux_u(i, j, k, grid::Grid, u::AbstractArray, ν_ccc::AbstractFloat) =
    viscous_flux_ux(i, j, k, grid, u, ν_ccc) - viscous_flux_ux(i-1, j, k, grid, u, ν_ccc)

""" Calculate δy_aca[ν * (Ay)ʸ * δy_afa(u)] -> fcc. """
@inline δy_viscous_flux_u(i, j, k, grid::Grid, u::AbstractArray, ν_ffc::AbstractFloat) =
    viscous_flux_uy(i, j+1, k, grid, u, ν_ffc) - viscous_flux_uy(i, j, k, grid, u, ν_ffc)

""" Calculate δz_aac[ν * (Az)ᶻ * δz_aaf(u)] -> fcc. """
@inline function δz_viscous_flux_u(i, j, k, grid::Grid, u::AbstractArray, ν_fcf::AbstractFloat)
    if k == grid.Nz
        return viscous_flux_uz(i, j, k, grid, u, ν_fcf)
    else
        return viscous_flux_uz(i, j, k, grid, u, ν_fcf) - viscous_flux_uz(i, j, k+1, grid, u, ν_fcf)
    end
end

"""
    ∇ν∇u(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the u-velocity via

    1/V * [δx_faa(ν * ϊx_caa(Ax) * δx_caa(u)) + δy_aca(ν * ϊy_afa(Ay) * δy_afa(u)) + δz_aac(ν * ϊz_aaf(Az) * δz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function ∇ν∇u(i, j, k, grid::Grid, u::AbstractArray, ν::AbstractFloat)
    1/Vᵘ(i, j, k, grid) * (δx_viscous_flux_u(i, j, k, grid, u, ν) +
                           δy_viscous_flux_u(i, j, k, grid, u, ν) +
                           δz_viscous_flux_u(i, j, k, grid, u, ν))
end

# Calculate viscous fluxes for v-velocity field.
""" Calculate ν * (Ax)ˣ * δx_faa(v) -> ffc. """
@inline viscous_flux_vx(i, j, k, grid::Grid, v::AbstractArray, ν_ffc::AbstractFloat) =
    ν_ffc * ϊAxF_faa(i, j, k, grid) * δx_faa(i, j, k, grid, v)

""" Calculate ν * (Ay)ʸ * δy_aca(v) -> ccc. """
@inline viscous_flux_vy(i, j, k, grid::Grid, v::AbstractArray, ν_ccc::AbstractFloat) =
    ν_ccc * ϊAyF_aca(i, j, k, grid) * δy_aca(i, j, k, grid, v)

""" Calculate ν * (Az)ᶻ * δz_aaf(v) -> cff. """
@inline viscous_flux_vz(i, j, k, grid::Grid, v::AbstractArray, ν_cff::AbstractFloat) =
    ν_cff * ϊAz_aaf(i, j, k, grid) * δz_aaf(i, j, k, grid, v)

# Calculate the components of the divergence of the v-viscous flux.
""" Calculate δx_caa[ν * (Ax)ˣ * δx_faa(v)] -> cfc. """
@inline δx_viscous_flux_v(i, j, k, grid::Grid, v::AbstractArray, ν_ffc::AbstractFloat) =
    viscous_flux_vx(i+1, j, k, grid, v, ν_ffc) - viscous_flux_vx(i, j, k, grid, v, ν_ffc)

""" Calculate δy_afa[ν * (Ay)ʸ * δy_aca(v)] -> cfc. """
@inline δy_viscous_flux_v(i, j, k, grid::Grid, v::AbstractArray, ν_ccc::AbstractFloat) =
    viscous_flux_vy(i, j, k, grid, v, ν_ccc) - viscous_flux_vy(i, j-1, k, grid, v, ν_ccc)

""" Calculate δz_aac[ν * (Az)ᶻ * δz_aaf(v)] -> cfc. """
@inline function δz_viscous_flux_v(i, j, k, grid::Grid, v::AbstractArray, ν_cff::AbstractFloat)
    if k == grid.Nz
        return viscous_flux_vz(i, j, k, grid, v, ν_cff)
    else
        return viscous_flux_vz(i, j, k, grid, v, ν_cff) - viscous_flux_vz(i, j, k+1, grid, v, ν_cff)
    end
end

"""
    ∇ν∇v(i, j, k, grid::Grid, v::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the v-velocity via

    1/Vᵛ * [δx_caa(ν * ϊx_faa(Ax) * δx_faa(v)) + δy_afa(ν * ϊy_aca(Ay) * δy_aca(v)) + δz_aac(ν * ϊz_aaf(Az) * δz_aaf(v))]

which will end up at the location `cfc`.
"""
@inline function ∇ν∇v(i, j, k, grid::Grid, v::AbstractArray, ν::AbstractFloat)
    1/Vᵛ(i, j, k, grid) * (δx_viscous_flux_v(i, j, k, grid, v, ν) +
                           δy_viscous_flux_v(i, j, k, grid, v, ν) +
                           δz_viscous_flux_v(i, j, k, grid, v, ν))
end

# Calculate viscous fluxes for v-velocity field.
""" Calculate ν * (Ax)ˣ * δx_faa(w) -> fcf. """
@inline viscous_flux_wx(i, j, k, grid::Grid, w::AbstractArray, ν_fcf::AbstractFloat) =
    ν_fcf * ϊAxF_faa(i, j, k, grid) * δx_faa(i, j, k, grid, w)

""" Calculate ν * (Ay)ʸ * δy_afa(w) -> cff. """
@inline viscous_flux_wy(i, j, k, grid::Grid, w::AbstractArray, ν_cff::AbstractFloat) =
    ν_cff * ϊAyF_afa(i, j, k, grid) * δy_afa(i, j, k, grid, w)

""" Calculate ν * (Az)ᶻ * δz_aac(w) -> ccc. """
@inline viscous_flux_wz(i, j, k, grid::Grid, w::AbstractArray, ν_ccc::AbstractFloat) =
    ν_ccc * ϊAz_aac(i, j, k, grid) * δz_aac(i, j, k, grid, w)

# Calculate the components of the divergence of the w-viscous flux.
""" Calculate δx_caa[ν * (Ax)ˣ * δx_faa(w)] -> ccf. """
@inline δx_viscous_flux_w(i, j, k, grid::Grid, w::AbstractArray, ν_fcf::AbstractFloat) =
    viscous_flux_wx(i+1, j, k, grid, w, ν_fcf) - viscous_flux_wx(i, j, k, grid, w, ν_fcf)

""" Calculate δy_aca[ν * (Ay)ʸ * δy_afa(w)] -> ccf. """
@inline δy_viscous_flux_w(i, j, k, grid::Grid, w::AbstractArray, ν_cff::AbstractFloat) =
    viscous_flux_wy(i, j+1, k, grid, w, ν_cff) - viscous_flux_wy(i, j, k, grid, w, ν_cff)

""" Calculates δz_aaf[ν * (Az)ᶻ * δz_aac(w)) -> ccf.  """
@inline function δz_viscous_flux_w(i, j, k, grid::Grid{T}, u::AbstractArray, ν_ccc::AbstractFloat) where T
    if k == 1
        return -zero(T)
    else
        return viscous_flux_wz(i, j, k-1, grid, w, ν_ccc) - viscous_flux_wz(i, j, k, grid, w, ν_ccc)
    end
end

"""
    ∇ν∇w(i, j, k, grid::Grid, w::AbstractArray, ν::AbstractFloat)

Calculates viscous dissipation for the w-velocity via

    1/Vʷ * [δx_caa(ν * ϊx_faa(Ax) * δx_faa(w)) + δy_aca(ν * ϊy_afa(Ay) * δy_afa(w)) + δz_aaf(ν * ϊz_aac(Az) * δz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function ∇ν∇w(i, j, k, grid::Grid, w::AbstractArray, ν::AbstractFloat)
    1/Vʷ(i, j, k, grid) * (δx_viscous_flux_w(i, j, k, grid, w, ν) +
                           δy_viscous_flux_w(i, j, k, grid, v, ν) +
                           δz_viscous_flux_w(i, j, k, grid, v, ν))
end
