# Calculate fluxes of a vector field (fx, fy, fz) through each face where the
# components (fx, fy, fz) are defined normal to the faces (e.g. a velocity field).
# The flux in this case is given by Ax*fx, Ay*fy, and Az*fz.
@inline flux_x(i, j, k, grid::Grid, fx::AbstractArray) = @inbounds AxF(i, j, k, grid) * fx[i, j, k]
@inline flux_y(i, j, k, grid::Grid, fy::AbstractArray) = @inbounds AyF(i, j, k, grid) * fy[i, j, k]
@inline flux_z(i, j, k, grid::Grid, fz::AbstractArray) = @inbounds  Az(i, j, k, grid) * fz[i, j, k]

# Calculate components of the divergence of a flux (fx, fy, fz).
@inline δx_caa_flux(i, j, k, grid::Grid, fx::AbstractArray) =
    flux_x(i+1, j, k, grid, fx) - flux_x(i, j, k, grid, fx)

@inline δy_aca_flux(i, j, k, grid::Grid, fy::AbstractArray) =
    flux_y(i, j+1, k, grid, fx) - flux_y(i, j, k, grid, fy)

@inline function δz_aac_flux(i, j, k, grid::Grid, fz::AbstractArray)
    if k == grid.Nz
        return flux_z(i, j, k, grid, fz)
    else
        return flux_z(i, j, k, grid, fz) - flux_z(i, j, k+1, grid, fz)
    end
end

# Calculate the flux of a tracer quantity Q (e.g. temperature) through the faces
# of a cell. In this case, the fluxes are given by u*Ax*T̅ˣ, v*Ay*T̅ʸ, and
# w*Az*T̅ᶻ.
@inline tracer_flux_x(i, j, k, grid::Grid, u::AbstractArray, Q::AbstractArray) =
    @inbounds AxF(i, j, k, grid) * u[i, j, k] * ϊx_faa(i, j, k, grid, Q)

@inline tracer_flux_y(i, j, k, grid::Grid, v::AbstractArray, Q::AbstractArray) =
    @inbounds AyF(i, j, k, grid) * v[i, j, k] * ϊy_afa(i, j, k, grid, Q)

@inline tracer_flux_z(i, j, k, grid::Grid, w::AbstractArray, Q::AbstractArray) =
    @inbounds Ax(i, j, k, grid)  * w[i, j, k] * ϊz_aaf(i, j, k, grid, Q)


# Calculate the components of the divergence of the flux of a tracer quantity Q
# over a cell.
@inline δx_caa_tracer_flux(i, j, k, grid::Grid, u::AbstractArray, Q::AbstractArray) =
    tracer_flux_x(i+1, j, k, grid, u, Q) - tracer_flux_x(i, j, k, grid, u, Q)

@inline δy_aca_tracer_flux(i, j, k, grid::Grid, v::AbstractArray, Q::AbstractArray) =
    tracer_flux_y(i, j+1, k, grid, v, Q) - tracer_flux_y(i, j, k, grid, v, Q)

@inline function δz_aac_tracer_flux(i, j, k, grid::Grid, w::AbstractArray, Q::AbstractArray)
    if k == grid.Nz
        return tracer_flux_z(i, j, k, grid, w, Q)
    else
        return tracer_flux_z(i, j, k, grid, w, Q) - tracer_flux_z(i, j, k+1, grid, w, Q)
    end
end

"""
    divh_u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray)

Calculates the horizontal divergence ∇ₕ·(u, v) of the velocity (u, v) via

    1/V * [δx_caa(Ax * u) + δy_aca(Ay * v)]

which will end up at the location `cca`.
"""
@inline function divh_u(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray)
    1/V(i, j, k, grid) * (δx_caa_flux(i, j, k, grid, u) + δy_aca_flux(i, j, k, grid, v))
end

"""
    div_ccc(i, j, k, grid::Grid, fx::AbstractArray, fy::AbstractArray, fz::AbstractArray)

Calculates the divergence ∇·f of a vector field f = (fx, fy, fz),

    1/V * [δx_caa(Ax * fx) + δx_aca(Ay * fy) + δz_aac(Az * fz)],

which will end up at the cell centers `ccc`.
"""
@inline function div_ccc(i, j, k, grid::Grid, fx::AbstractArray, fy::AbstractArray, fz::AbstractArray)
    1/V(i, j, k, grid) * (δx_caa_flux(i, j, k, grid, fx) +
                          δy_aca_flux(i, j, k, grid, fy) +
                          δz_aac_flux(i, j, k, grid, fz))
end

"""
    div_flux(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray, Q::AbstractArray)

Calculates the divergence of the flux of a tracer quantity Q being advected by
a velocity field v = (u, v, w), ∇·(vQ),

    1/V * [δx_caa(Ax * u * ϊx_faa(Q)) + δy_aca(Ay * v * ϊy_afa(Q)) + δz_aac(Az * w * ϊz_aaf(Q))]

which will end up at the location `ccc`.
"""
@inline function div_flux(i, j, k, grid::Grid, u::AbstractArray, v::AbstractArray, w::AbstractArray, Q::AbstractArray)
    if k == 1
        return @inbounds 1/V(i, j, k, grid) * (δx_caa_tracer_flux(i, j, k, grid, u, Q) +
                                               δy_aca_tracer_flux(i, j, k, grid, v, Q) -
                                               tracer_flux_z(i, j, 2, w, Q))
    else
        return 1/V(i, j, k, grid) * (δx_caa_tracer_flux(i, j, k, grid, u, Q) +
                                     δy_aca_tracer_flux(i, j, k, grid, v, Q) +
                                     δz_aac_tracer_flux(i, j, k, grid, w, Q))
    end
end
