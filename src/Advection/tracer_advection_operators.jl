using Oceananigans.Operators: V⁻¹ᶜᶜᶜ

@inline _advective_tracer_flux_x(i, j, k, grid, scheme, U, c) = advective_tracer_flux_x(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), U, c)
@inline _advective_tracer_flux_y(i, j, k, grid, scheme, V, c) = advective_tracer_flux_y(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), V, c)
@inline _advective_tracer_flux_z(i, j, k, grid, scheme, W, c) = advective_tracer_flux_z(i, j, k, grid, scheme, TimeSteppers.time_discretization(scheme), W, c)

#####
##### Fallback tracer fluxes!
#####

# Fallback for `nothing` advection
@inline _advective_tracer_flux_x(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_y(i, j, k, grid, ::Nothing, args...) = zero(grid)
@inline _advective_tracer_flux_z(i, j, k, grid, ::Nothing, args...) = zero(grid)

#####
##### Tracer advection operator
#####

"""
$(TYPEDSIGNATURES)

Calculate the divergence of the flux of a tracer quantity ``c`` being advected by
a velocity field, ``𝛁⋅(𝐯 c)``,

```
1/V * [δxᶜᵃᵃ(Ax * u * ℑxᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * v * ℑyᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * w * ℑzᵃᵃᶠ(c))]
```
which ends up at the location `ccc`.
"""
@inline function div_Uc(i, j, k, grid, advection, U, c)
    ℓ  = (Center(), Center(), Center())
    δx = advective_flux_difference(i, j, k, grid, Val(1), ℓ, advection, U.u, c)
    δy = advective_flux_difference(i, j, k, grid, Val(2), ℓ, advection, U.v, c)
    δz = advective_flux_difference(i, j, k, grid, Val(3), ℓ, advection, U.w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (δx + δy + δz)
end

#####
##### Flux difference across a cell, from one stencil of 2N + 1 values shared by both faces
#####

# the velocity that sets the upwind bias: the volume flux, or the velocity itself across a tracer cell
@inline upwind_velocity(i, j, k, grid, dir, ℓ, scheme, U) = transport(i, j, k, grid, dir, ℓ, scheme, U)
@inline upwind_velocity(i, j, k, grid, dir, ::Tuple{Center, Center, Center}, scheme, U) = @inbounds U[i, j, k]

# the volume flux through the faces along `dir` of a tracer cell
@inline transport(i, j, k, grid, ::Val{1}, ::Tuple{Center, Center, Center}, scheme, U) = Ax_qᶠᶜᶜ(i, j, k, grid, U)
@inline transport(i, j, k, grid, ::Val{2}, ::Tuple{Center, Center, Center}, scheme, V) = Ay_qᶜᶠᶜ(i, j, k, grid, V)
@inline transport(i, j, k, grid, ::Val{3}, ::Tuple{Center, Center, Center}, scheme, W) = Az_qᶜᶜᶠ(i, j, k, grid, W)

# the factor `s` of the explicit flux `s * Ũ * ψ` through faces at `ℓ`, below one only where the time discretization treats
# the flux implicitly
@inline explicit_velocity_scale(i, j, k, grid, ℓ, scheme, td, U) = one(grid)

# the reconstructions at the two faces along `dir` of the cell at `ℓ`, from one window of their 2N + 1 values
@inline function face_values(i, j, k, grid, dir, ℓ, scheme::AbstractAdvectionScheme{N}, ũ⁻, ũ⁺, ψ) where N
    ℓᶠ = node_location(dir, ℓ)
    (S⁻, _, _), (S⁺, _, _) = load_stencil(i, j, k, grid, dir, ℓᶠ, Val(N), Val(2N + 1), ψ)
    C⁻ = stretched_coefficients(i, j, k, dir, ℓᶠ, scheme, bias(ũ⁻))
    C⁺ = stretched_coefficients(stencil_node(i, j, k, dir, 1)..., dir, ℓᶠ, scheme, bias(ũ⁺))
    return stretched_interpolate(scheme, bias(ũ⁻), S⁻, C⁻), stretched_interpolate(scheme, bias(ũ⁺), S⁺, C⁺)
end

@inline function face_values(i, j, k, grid, dir, ℓ::Tuple{Center, Center, Center}, scheme::AbstractAdvectionScheme{N}, ũ⁻, ũ⁺, ψ) where N
    ℓᶠ = node_location(dir, ℓ)
    stencil⁻, stencil⁺ = load_stencil(i, j, k, grid, dir, ℓᶠ, Val(N), Val(2N + 1), CubicContinuation(), ψ)
    C⁻ = stretched_coefficients(i, j, k, dir, ℓᶠ, scheme, bias(ũ⁻))
    C⁺ = stretched_coefficients(stencil_node(i, j, k, dir, 1)..., dir, ℓᶠ, scheme, bias(ũ⁺))
    return wall_interpolate(scheme, bias(ũ⁻), stencil⁻, C⁻), wall_interpolate(scheme, bias(ũ⁺), stencil⁺, C⁺)
end

# The difference of the advective fluxes of `ψ` through the two faces along `dir` of its cell at `ℓ`, centered along `dir`
@inline function advective_flux_difference(i, j, k, grid, dir, ℓ, scheme::Union{CenteredScheme{N}, UpwindScheme{N}}, U, ψ) where N
    ℓᶠ = node_location(dir, ℓ)

    i⁺, j⁺, k⁺ = stencil_node(i, j, k, dir, 1)
    ũ⁻ = upwind_velocity(i,  j,  k,  grid, dir, ℓ, scheme, U)
    ũ⁺ = upwind_velocity(i⁺, j⁺, k⁺, grid, dir, ℓ, scheme, U)
    ψ⁻, ψ⁺ = face_values(i, j, k, grid, dir, ℓ, scheme, ũ⁻, ũ⁺, ψ)

    td = TimeSteppers.time_discretization(scheme)
    s⁻ = explicit_velocity_scale(i,  j,  k,  grid, ℓᶠ, scheme, td, U)
    s⁺ = explicit_velocity_scale(i⁺, j⁺, k⁺, grid, ℓᶠ, scheme, td, U)

    F⁻ = s⁻ * (transport(i,  j,  k,  grid, dir, ℓ, scheme, U) * ψ⁻)
    F⁺ = s⁺ * (transport(i⁺, j⁺, k⁺, grid, dir, ℓ, scheme, U) * ψ⁺)

    return F⁺ - F⁻
end

@inline advective_flux_difference(i, j, k, grid, dir, ℓ, ::Nothing, U, ψ) = zero(grid)

# Fallbacks for zero velocities, zero tracer and `nothing` advection
@inline div_Uc(i, j, k, grid, advection, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, advection, ::ZeroU, ::ZeroField) = zero(grid)

@inline div_Uc(i, j, k, grid, ::Nothing, U, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, c) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, U, ::ZeroField) = zero(grid)
@inline div_Uc(i, j, k, grid, ::Nothing, ::ZeroU, ::ZeroField) = zero(grid)
