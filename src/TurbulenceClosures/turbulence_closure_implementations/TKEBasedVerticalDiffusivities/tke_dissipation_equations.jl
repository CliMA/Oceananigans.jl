import Oceananigans.TurbulenceClosures: closure_source_term

"""
    struct DissipationEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct TKEDissipationEquations{FT}
    Cᴾϵ  :: FT = 1.44  # Dissipation equation coefficient for shear production
    Cᴮϵ  :: FT = -0.65 # Dissipation equation coefficient for buoyancy flux
    Cᵋϵ  :: FT = 1.92  # Dissipation equation coefficient for dissipation
    σϵ   :: FT = 1.2   # Dissipation Schmidt number
    σk   :: FT = 1.0   # TKE Schmidt number
end

#####
##### TKE equation
#####

@inline closure_source_term(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, ::Val{:k}, velocities, tracers, buoyancy) =
    shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy) + 
       buoyancy_flux(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy) -
         dissipation(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)

@inline function shear_production(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, velocities, tracers, buoyancy)
    u = velocities.u
    v = velocities.v
    κᵘ = diffusivity_fields.κᵘ
    return shear_production(i, j, k, grid, κᵘ, u, v)
end

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, velocities, tracers, buoyancy) =
    explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κᶜ)

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD}, diffusivity_fields, velocities, tracers, buoyancy)
    wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κᶜ)
    kⁱʲᵏ = @inbounds tracers.k[i, j, k]

    closure_ij = getclosure(i, j, closure)
    kᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

    dissipative_buoyancy_flux = (sign(wb) * sign(kⁱʲᵏ) < 0) & (kⁱʲᵏ > kᵐⁱⁿ)

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    return ifelse(dissipative_buoyancy_flux, zero(grid), wb)
end

@inline dissipation(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD}, args...) = zero(grid)

#####
##### Dissipation equation
#####

@inline function closure_source_term(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, ::Val{:ϵ}, velocities, tracers, buoyancy)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    k = @inbounds tracers.k[i, j, k]
    kᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy
    k̃ = min(kᵐⁱⁿ, k)

    Pᵏ = shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    Dᵋ = dissipation_destruction(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields)
    wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κᶜ)

    closure_ij = getclosure(i, j, closure)
    Cᴾϵ = closure_ij.dissipation_equation.Cᴾϵ 
    Cᴮϵ = closure_ij.dissipation_equation.Cᴮϵ 

    # Explicit/implicit buoyancy flux term in dissipation equation
    Bᵋ = dissipation_buoyancy_source(closure_ij, grid, wb, k̃)

    # Production of dissipation / reduction of mixing length by shear production of TKE
    Pᵋ = Cᴾϵ * Pᵏ * ϵ / k̃ 

    return Pᵋ + Bᵋ - Dᵋ
end

@inline dissipation_buoyancy_source(closure::FlavorOfKEpsilon, grid, wb, k) = closure.Cᴮϵ * wb / k

@inline function dissipation_buoyancy_source(closure::FlavorOfKEpsilon{<:VITD}, grid, wb, k)
    Bᵋ = closure.Cᴮϵ * wb / k
    return max(zero(grid), Bᵋ)
end

@inline function dissipation_destruction(i, j, k, grid, closure::FlavorOfKEpsilon, velocities, tracers, buoyancy, diffusivity_fields)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    closure_ij = getclosure(i, j, closure)
    Cᵋϵ = closure_ij.dissipation_equation.Cᵋϵ     
    return Cᵋϵ * ϵ^2
end

@inline dissipation_destruction(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD}, args...) = zero(grid)


