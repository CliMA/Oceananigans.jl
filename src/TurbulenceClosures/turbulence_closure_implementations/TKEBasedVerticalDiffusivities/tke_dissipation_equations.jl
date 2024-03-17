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

@inline function closure_source_term(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, ::Val{:e},
                                     velocities, tracers, buoyancy)

    P = shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    wb = buoyancy_flux(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    ϵ = @inbounds tracers.ϵ[i, j, k]
    ϵ = clip(ϵ)

    return P + wb - ϵ
end

@inline function shear_production(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields,
                                  velocities, tracers, buoyancy)
    u = velocities.u
    v = velocities.v
    κu = diffusivity_fields.κu
    return shear_production(i, j, k, grid, κu, u, v)
end

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, velocities, tracers, buoyancy) =
    explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD}, diffusivity_fields,
                               velocities, tracers, buoyancy)

    wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]

    closure_ij = getclosure(i, j, closure)
    eᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

    dissipative_buoyancy_flux = (sign(wb) * sign(eⁱʲᵏ) < 0) & (eⁱʲᵏ > eᵐⁱⁿ)

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    return ifelse(dissipative_buoyancy_flux, zero(grid), wb)
end

@inline dissipation(i, j, k, grid, closure::FlavorOfKEpsilon{<:VITD}, args...) = zero(grid)

#####
##### Dissipation equation
#####

@inline function closure_source_term(i, j, k, grid, closure::FlavorOfKEpsilon, diffusivity_fields, ::Val{:ϵ},
                                     velocities, tracers, buoyancy)

    ϵⁱʲᵏ = @inbounds tracers.ϵ[i, j, k]
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]
    closure_ij = getclosure(i, j, closure)

    eᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy
    eˡⁱᵐ = min(eᵐⁱⁿ, eⁱʲᵏ)

    # Production of dissipation / reduction of mixing length by shear production of TKE
    Pᵉ = shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    Cᴾϵ = closure_ij.tke_dissipation_equations.Cᴾϵ 
    Pᵋ = Cᴾϵ * Pᵉ * ϵⁱʲᵏ / eˡⁱᵐ

    # Production / destruction of dissipation by buoyancy fluxes
    wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)
    ωᴮ = dissipation_buoyancy_transformation_rate(closure_ij, wb, eⁱʲᵏ)

    # Explicit/implicit buoyancy flux term in dissipation equation
    ωᴰ = dissipation_destruction_rate(closure_ij, ϵⁱʲᵏ, eⁱʲᵏ)

    #return Pᵋ + ϵⁱʲᵏ * ωᴮ - ϵⁱʲᵏ * ωᴰ
    #return Pᵋ + ϵⁱʲᵏ * ωᴮ - ϵⁱʲᵏ * ωᴰ
    #return - ϵⁱʲᵏ * ωᴰ
    return 0 #Pᵋ
end

@inline function explicit_dissipation_buoyancy_transformation_rate(closure, wb, e)
    Cᴮϵ = closure.tke_dissipation_equations.Cᴮϵ
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    eˡⁱᵐ = min(eᵐⁱⁿ, e)
    return Cᴮϵ * wb / eˡⁱᵐ
end

# Dissipation buoyancy source
@inline dissipation_buoyancy_transformation_rate(closure::FlavorOfKEpsilon, wb, e) =
    explicit_dissipation_buoyancy_transformation_rate(closure::FlavorOfKEpsilon, wb, e)

@inline function dissipation_buoyancy_transformation_rate(closure::FlavorOfKEpsilon{<:VITD}, wb, e)
    ωᴮ = explicit_dissipation_buoyancy_transformation_rate(closure, wb, e)
    return max(zero(ωᴮ), ωᴮ)
end

@inline function explicit_dissipation_destruction_rate(closure, ϵ, e)
    Cᵋϵ = closure.tke_dissipation_equations.Cᵋϵ
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    eˡⁱᵐ = min(eᵐⁱⁿ, e)
    return 1e-6 * Cᵋϵ *  ϵ / eˡⁱᵐ
end

# Dissipation destruction
@inline dissipation_destruction_rate(closure::FlavorOfKEpsilon, ϵ, e) = explicit_dissipation_destruction_rate(closure, ϵ, e)
@inline dissipation_destruction_rate(closure::FlavorOfKEpsilon{<:VITD}, ϵ, e) = zero(ϵ)

