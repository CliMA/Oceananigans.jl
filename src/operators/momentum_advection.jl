# Calculate momentum fluxes for the u-velocity field.
""" Calculate (AxF * u)ˣ * u̅ˣ -> ccc. """
@inline mom_flux_uu(i, j, k, grid::Grid, u::AbstractArray) =
    ϊxAxF_caa(i, j, k, grid, u) * ϊx_caa(i, j, k, grid, u)

""" Calculate (AyF * v)ˣ * u̅ʸ -> ffc. """
@inline mom_flux_uv(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray) =
    ϊxAyF_faa(i, j, k, grid, v) * ϊy_afa(i, j, k, grid, u)

""" Calculate (Az * w)ˣ * u̅ᶻ -> fcf. """
@inline mom_flux_uw(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray) =
    ϊxAz_faa(i, j, k, grid, w) * ϊz_aaf(i, j, k, grid, u)

# Calculate the components of the divergence of the u-momentum flux.
""" Calculate δx_faa[(AxF * u)ˣ * u̅ˣ] -> fcc. """
@inline δx_mom_flux_u(i, j, k, grid::Grid, u::AbstractArray) =
    mom_flux_uu(i, j, k, grid, u) - mom_flux_uu(i-1, j, k, grid, u)

""" Calculate δy_aca[(AyF * v)ˣ * u̅ʸ] -> fcc. """
@inline δy_mom_flux_u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray) =
    mom_flux_uv(i, j+1, k, grid, u, v) - mom_flux_uv(i, j, k, grid, u, v)

""" Calculate δz_aac[(Az * w)ˣ * u̅ᶻ] -> fcc. """
@inline function δz_mom_flux_u(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray)
    if k == grid.Nz
        return mom_flux_uw(i, j, k, grid, u, w)
    else
        return mom_flux_uw(i, j, k+1, grid, u, w) - mom_flux_uw(i, j, k, grid, u, w)
    end
end

# Calculate momentum fluxes for the v-velocity field.
""" Calculate (AxF * u)ʸ * v̅ˣ -> ffc. """
@inline mom_flux_vu(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray) =
    ϊyAxF_afa(i, j, k, grid, u) * ϊx_faa(i,   j, k, grid, v)

""" Calculate (AyF * v)ʸ * v̅ʸ -> ccc. """
@inline mom_flux_vv(i, j, k, grid::Grid, v::AbstractArray) =
    ϊyAyF_aca(i, j, k, grid, v) * ϊy_aca(i, j, k, grid, v)

""" Calculate (Az * w)ʸ * v̅ᶻ -> cff. """
@inline mom_flux_vw(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray) =
    ϊyAz_afa(i, j, k, grid, w) * ϊz_aaf(i, j, k, grid, v)

# Calculate the components of the divergence of the v-momentum flux.
""" Calculate δx_caa[(AxF * u)ʸ * v̅ˣ] -> cfc. """
@inline δx_mom_flux_v(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray) =
    mom_flux_vu(i+1, j, k, grid, u, v) - mom_flux_vu(i, j, k, grid, u, v)

""" Calculate δy_afa[(AyF * v)ʸ * v̅ʸ] -> cfc. """
@inline δy_mom_flux_v(i, j, k, grid::Grid, v::AbstractArray) =
    mom_flux_vv(i, j, k, grid, v) - mom_flux_vv(i, j-1, k, grid, v)

""" Calculate δz_aac[(Az * w)ʸ * v̅ᶻ] -> cfc. """
@inline function δz_mom_flux_v(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray)
    if k == grid.Nz
        return mom_flux_vw(i, j, k, grid, v, w)
    else
        return mom_flux_vw(i, j, k, grid, v, w) - mom_flux_vw(i, j, k+1, grid, v, w)
    end
end

# Calculate momentum fluxes for the w-velocity field.
""" Calculate (AxF * u)ᶻ * w̅ˣ -> fcf. """
@inline mom_flux_wu(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray) =
    ϊzAxF_aaf(i, j, k, grid, u) * ϊx_faa(i, j, k, grid, w)

""" Calculate δx_caa[(AxF * u)ᶻ * w̅ˣ] -> ccf. """
@inline δx_mom_flux_w(i, j, k, grid::Grid, u::AbstractArray, w::AbstractArray) =
    mom_flux_wu(i+1, j, k, grid, u, w) - mom_flux_wu(i, j, k, grid, u, w)

""" Calculate (AyF * v)ᶻ * w̅ʸ -> cff. """
@inline mom_flux_wv(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray) =
    ϊzAyF_afa(i, j, k, grid, v) * ϊy_afa(i, j, k, grid, w)

""" Calculate δy_aca[(AyF * v)ᶻ * w̅ʸ] -> ccf. """
@inline δy_mom_flux_v(i, j, k, grid::Grid, v::AbstractArray, w::AbstractArray) =
    mom_flux_wv(i, j+1, k, grid, v, w) - mom_flux_wv(i, j, k, grid, v, w)

""" Calculate (Az * w)ᶻ * w̅ᶻ -> ccc. """
@inline mom_flux_ww(i, j, k, grid::Grid{T}, w::AbstractArray) =
    ϊzAz_aac(i, j, k, grid, w) * ϊz_aac(i, j, k, grid, w)

""" Calculate δz_aaf[(Az * w)ᶻ * w̅ᶻ] -> ccf. """
@inline function δz_mom_flux_w(i, j, k, grid::Grid{T}, w::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        return mom_flux_ww(i, j, k-1, grid, w) - mom_flux_ww(i, j, k, grid, w)
    end
end

"""
    u∇u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)

Calculates the advection of momentum in the x-direction v·∇u with a velocity
field v = (u, v, w),

    1/Vᵘ * [δx_faa(ϊx_caa(Ax * u) * ϊx_caa(u)) + δy_fca(ϊx_faa(Ay * v) * ϊy_afa(u)) + δz_fac(ϊx_faa(Az * w) * ϊz_aaf(u))]

which will end up at the location `fcc`.
"""
@inline function u∇u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    1/Vᵘ(i, j, k, grid) * (δx_mom_flux_u(i, j, k, grid, u) +
                           δy_mom_flux_u(i, j, k, grid, u, v) +
                           δz_mom_flux_u(i, j, k, grid, u, w))
end

"""
    u∇v(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)

Calculates the advection of momentum in the y-direction v·∇v with a velocity
field v = (u, v, w) via

    (v̅ʸ)⁻¹ * [δx_cfa(ϊy_afa(Ax * u) * ϊx_faa(v)) + δy_afa(ϊy_aca(Ay * v) * ϊy_aca(v)) + δz_afc(ϊx_faa(Az * w) * ϊz_aaf(w))]

which will end up at the location `cfc`.
"""
@inline function u∇v(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    1/Vᵛ(i, j, k, grid) * (δx_mom_flux_v(i, j, k, grid, u, v) +
                           δy_mom_flux_v(i, j, k, grid, v) +
                           δz_mom_flux_v(i, j, k, grid, v, w))
end

"""
    u∇w(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)

Calculates the advection of momentum in the z-direction v·∇w with a velocity
field v = (u, v, w),

    1/Vʷ * [δx_caf(ϊz_aaf(Ax * u) * ϊx_faa(w)) + δy_acf(ϊz_aaf(Ay * v) * ϊy_afa(w)) + δz_aaf(ϊz_aac(Az * w) * ϊz_aac(w))]

which will end up at the location `ccf`.
"""
@inline function u∇w(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray)
    1/Vʷ(i, j, k, grid) * (δx_mom_flux_w(i, j, k, grid, u, w) +
                           δy_mom_flux_w(i, j, k, grid, v, w) +
                           δz_mom_flux_w(i, j, k, grid, w))
end
