using Oceananigans.Operators: V⁻¹ᶜᶜᶜ

@inline _advective_tracer_flux_x(i, j, k, grid, scheme, U, c) = advective_tracer_flux_x(i, j, k, grid, scheme, U, c)
@inline _advective_tracer_flux_y(i, j, k, grid, scheme, V, c) = advective_tracer_flux_y(i, j, k, grid, scheme, V, c)
@inline _advective_tracer_flux_z(i, j, k, grid, scheme, W, c) = advective_tracer_flux_z(i, j, k, grid, scheme, W, c)

@inline _transport_flux_value(q::Number, i, j, k) = q
@inline _transport_flux_value(q, i, j, k) = @inbounds q[i, j, k]

#####
##### Fallback tracer fluxes!
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, args...) = zero(grid)

#####
##### Non-orthogonal horizontal tracer fluxes
#####

@inline _nonorthogonal_advective_tracer_flux_x(i, j, k, grid, scheme::FluxFormAdvection, U, c) =
    _nonorthogonal_advective_tracer_flux_x(i, j, k, grid, scheme.x, U, c)

@inline _nonorthogonal_advective_tracer_flux_y(i, j, k, grid, scheme::FluxFormAdvection, U, c) =
    _nonorthogonal_advective_tracer_flux_y(i, j, k, grid, scheme.y, U, c)

@inline _nonorthogonal_advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, c) =
    advective_tracer_flux_x(i, j, k, grid, scheme, U, c)

@inline _nonorthogonal_advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, scheme::CenteredScheme, U, c) =
    advective_tracer_flux_y(i, j, k, grid, scheme, U, c)

@inline _nonorthogonal_advective_tracer_flux_x(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, c) =
    advective_tracer_flux_x(i, j, k, grid, scheme, U, c)

@inline _nonorthogonal_advective_tracer_flux_y(i, j, k, grid::SphericalShellGrid, scheme::UpwindScheme, U, c) =
    advective_tracer_flux_y(i, j, k, grid, scheme, U, c)

@inline nonorthogonal_vertical_tracer_flux_divergence(i, j, k, grid, advection, W, c) =
    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, W, c)

@inline nonorthogonal_vertical_tracer_flux_divergence(i, j, k, grid, advection, ::ZeroField, c) =
    zero(grid)

#####
##### Tracer advection operator
#####

"""
    div_uc(i, j, k, grid, advection, U, c)

Calculate the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field, ``𝛁⋅(𝐯 c)``,

```
1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]
```
which ends up at the location `ccc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    u = u_velocity(U)
    v = v_velocity(U)
    w = w_velocity(U)

    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, w, c))
end

@inline function div_Uc(i, j, k, grid::SphericalShellGrid, advection, U, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U, c) +
                                    nonorthogonal_vertical_tracer_flux_divergence(i, j, k, grid, advection, w_velocity(U), c))
end

# Fallbacks for zero velocities, zero tracer and `nothing` advection
@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, advection, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, advection, ::ZeroU, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid::SphericalShellGrid, ::Nothing, ::ZeroU, ::ZeroField) = zero(grid)
