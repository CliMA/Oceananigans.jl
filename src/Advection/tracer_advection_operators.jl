
@inline _advective_tracer_flux_x(args...) = advective_tracer_flux_x(args...)
@inline _advective_tracer_flux_y(args...) = advective_tracer_flux_y(args...)
@inline _advective_tracer_flux_z(args...) = advective_tracer_flux_z(args...)

#####
##### Fallback tracer fluxes!
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, args...) = zero(grid)

# Fallback for `nothing` advection and `ZeroField` tracers and velocities
@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, ::ZeroField, ::ZeroField) = zero(grid)

@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, V, ::ZeroField) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, W, ::ZeroField) = zero(grid)
@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, ::ZeroField, c) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, ::ZeroField, c) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, ::ZeroField, c) = zero(grid)

for scheme in (:UpwindBiased, :Centered, :WENO, :FluxFormAdvection)
    @eval begin
        # Fallback for `ZeroField` tracers and velocities
        @inline _advective_tracer_flux_x(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_tracer_flux_y(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)
        @inline _advective_tracer_flux_z(i, j, k, grid, ::$Scheme, ::ZeroField, ::ZeroField) = zero(grid)

        # Fallback for `ZeroField` tracers
        @inline _advective_tracer_flux_x(i, j, k, grid, ::$Scheme, U, ::ZeroField) = zero(grid)
        @inline _advective_tracer_flux_y(i, j, k, grid, ::$Scheme, V, ::ZeroField) = zero(grid)
        @inline _advective_tracer_flux_z(i, j, k, grid, ::$Scheme, W, ::ZeroField) = zero(grid)

        # Fallback for `ZeroField` velocities
        @inline _advective_tracer_flux_x(i, j, k, grid, ::$Scheme, ::ZeroField, c) = zero(grid)
        @inline _advective_tracer_flux_y(i, j, k, grid, ::$Scheme, ::ZeroField, c) = zero(grid)
        @inline _advective_tracer_flux_z(i, j, k, grid, ::$Scheme, ::ZeroField, c) = zero(grid)
    end
end

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
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, _advective_tracer_flux_x, advection, U.u, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, _advective_tracer_flux_y, advection, U.v, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, _advective_tracer_flux_z, advection, U.w, c))
end

# Fallbacks for zero velocities, zero tracer and `nothing` advection
@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
