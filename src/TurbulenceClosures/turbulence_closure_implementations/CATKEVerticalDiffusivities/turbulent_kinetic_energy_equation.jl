#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

@inline ϕ²(i, j, k, grid, ϕ) = ϕ(i, j, k, grid)^2

# Temporary way to get the vertical diffusivity for the TKE equation terms...
# Assumes that the vertical diffusivity is dominated by the CATKE contribution.
@inline shear_production(i, j, k, grid, closure, velocities, diffusivities) = zero(eltype(grid))
@inline buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities) = zero(eltype(grid))

# Unlike the above, this fallback for dissipation is generically correct (we only want to compute dissipation once)
@inline dissipation(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs) = zero(eltype(grid))

@inline function shear_production(i, j, k, grid, closure::FlavorOfCATKE, velocities, diffusivities)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    νᶻ = diffusivities.Kᵘ
    νᶻ_ijk = @inbounds νᶻ[i, j, k]
    return νᶻ_ijk * (∂z_u² + ∂z_v²)
end

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivities)
    κᶻ = diffusivities.Kᶜ
    κᶻ_ijk = @inbounds κᶻ[i, j, k]
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - κᶻ_ijk * N²
end

const VITD = VerticallyImplicitTimeDiscretization

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(eltype(grid))

@inline function implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, velocities, tracers, buoyancy, clock, tracer_bcs)
    e = tracers.e
    FT = eltype(grid)
    @inbounds e⁺ = abs(e[i, j, k])

    ℓ = TKE_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴰ = closure.Cᴰ

    # Note:
    #   * because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   * then      L e = - ϵ
    #                   = - Cᴰ e³² / ℓ
    #
    #   * and thus    L = - Cᴰ √e / ℓ
    
    return - Cᴰ * sqrt(e⁺) / ℓ
end

# Fallbacks for explicit time discretization
@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, args...) =
    @inbounds - tracers.e[i, j, k] * implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE, args...)

@inline implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE, args...) = zero(eltype(grid))

#####
##### For closure tuples...
#####

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1])

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any, <:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1]) +
    shear_production(i, j, k, grid, closures[2], velocities, diffusivities[2])

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1]) +
    shear_production(i, j, k, grid, closures[2], velocities, diffusivities[2]) +
    shear_production(i, j, k, grid, closures[3], velocities, diffusivities[3])

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, diffusivities[1])

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    buoyancy_flux(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2])

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    buoyancy_flux(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2]) +
    buoyancy_flux(i, j, k, grid, closures[3], velocities, tracers, buoyancy, diffusivities[3])

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any}, velocities, tracers, buoyancy, diffusivities) =
    dissipation(i, j, k, grid, closures[1], velocities, diffusivities[1])

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    dissipation(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    dissipation(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2])

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    dissipation(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    dissipation(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2]) +
    dissipation(i, j, k, grid, closures[3], velocities, tracers, buoyancy, diffusivities[3])

