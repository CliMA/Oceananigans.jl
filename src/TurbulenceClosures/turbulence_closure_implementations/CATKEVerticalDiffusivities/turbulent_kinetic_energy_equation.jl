#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

@inline ϕ²(i, j, k, grid, ϕ) = ϕ(i, j, k, grid)^2

@inline function shear_production(i, j, k, grid, closure, velocities, diffusivities)
    ∂z_u² = ℑxzᶜᵃᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.u)
    ∂z_v² = ℑyzᵃᶜᶜ(i, j, k, grid, ϕ², ∂zᵃᵃᶠ, velocities.v)
    @inbounds Ku = diffusivities.Kᵘ[i, j, k]
    return Ku * (∂z_u² + ∂z_v²)
end

@inline function buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities)
    @inbounds Kc = diffusivities.Kᶜ[i, j, k]
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - Kc * N²
end

@inline function dissipation(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    e = tracers.e
    FT = eltype(grid)
    @inbounds eⁱʲᵏ = e[i, j, k]
    ẽ³² = sqrt(abs(eⁱʲᵏ^3))

    ℓ = TKE_mixing_lengthᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs)
    Cᴰ = closure.Cᴰ

    return Cᴰ * ẽ³² / ℓ
end

