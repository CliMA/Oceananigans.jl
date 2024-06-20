"""
    struct CATKEEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct CATKEEquation{FT}
    CʰⁱD  :: FT = 0.357 # Dissipation length scale shear coefficient for high Ri
    CˡᵒD  :: FT = 0.926 # Dissipation length scale shear coefficient for low Ri
    CᵘⁿD  :: FT = 1.437 # Dissipation length scale shear coefficient for high Ri
    CᶜD   :: FT = 2.556 # Dissipation length scale convecting layer coefficient
    CᵉD   :: FT = 0.0   # Dissipation length scale penetration layer coefficient
    Cᵂu★  :: FT = 0.405 # Surface shear-driven TKE flux coefficient
    CᵂwΔ  :: FT = 0.873 # Surface convective TKE flux coefficient
    Cᵂϵ   :: FT = 1.0   # Dissipative near-bottom TKE flux coefficient
end

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

#=
@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivities) =
    explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities)

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, velocities, tracers, buoyancy, diffusivities)
    wb = explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities)

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    return max(zero(grid), wb)
end
=#

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(grid)

@inline function dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers,
                                             buoyancy, surface_buoyancy_flux)

    # Convective dissipation length
    Cᶜ = closure.turbulent_kinetic_energy_equation.CᶜD
    Cᵉ = closure.turbulent_kinetic_energy_equation.CᵉD
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    Jᵇ = surface_buoyancy_flux
    ℓʰ = convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, Jᵇ)

    # "Stable" dissipation length
    Cˡᵒ = closure.turbulent_kinetic_energy_equation.CˡᵒD
    Cʰⁱ = closure.turbulent_kinetic_energy_equation.CʰⁱD
    Cᵘⁿ = closure.turbulent_kinetic_energy_equation.CᵘⁿD
    σᴰ = stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cᵘⁿ, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    ℓ★ = stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)
    ℓ★ = ℓ★ / σᴰ

    # Dissipation length
    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓᴰ = max(ℓ★, ℓʰ)

    H = total_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓᴰ)
end

@inline function dissipation_rate(i, j, k, grid, closure::FlavorOfCATKE,
                                  velocities, tracers, buoyancy, diffusivities)

    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities.Jᵇ)
    e = tracers.e
    FT = eltype(grid)
    eᵢ = @inbounds e[i, j, k]
    
    # Note:
    #   Because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   then      L e = - ϵ
    #                 = - Cᴰ e³² / ℓ
    #
    #   and thus    L = - Cᴰ √e / ℓ .

    ω_numerical = 1 / closure.negative_tke_damping_time_scale
    ω_physical = sqrt(abs(eᵢ)) / ℓᴰ

    return ifelse(eᵢ < 0, ω_numerical, ω_physical)
end

# Fallbacks for explicit time discretization
@inline function dissipation(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, args...)
    eᵢ = @inbounds tracers.e[i, j, k]
    ω = dissipation_rate(i, j, k, grid, closure, velocities, tracers, args...)
    return ω * eᵢ
end

#####
##### TKE top boundary condition
#####

@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfCATKE, buoyancy)
    closure = getclosure(i, j, closure)

    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    tke_parameters = closure.turbulent_kinetic_energy_equation

    return _top_tke_flux(i, j, grid, clock, fields, tke_parameters, closure,
                         buoyancy, top_tracer_bcs, top_velocity_bcs)
end

@inline function _top_tke_flux(i, j, grid, clock, fields,
                               tke::CATKEEquation, closure::CATKEVD,
                               buoyancy, top_tracer_bcs, top_velocity_bcs)

    wΔ³ = top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᵂu★ = tke.Cᵂu★
    CᵂwΔ = tke.CᵂwΔ

    return - Cᵂu★ * u★^3 - CᵂwΔ * wΔ³
end

#####
##### Utilities for model constructors
#####

""" Add TKE boundary conditions specific to `CATKEVerticalDiffusivity`. """
function add_closure_specific_boundary_conditions(closure::FlavorOfCATKE,
                                                  user_bcs,
                                                  grid,
                                                  tracer_names,
                                                  buoyancy)

    top_tracer_bcs = top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    top_velocity_bcs = top_velocity_boundary_conditions(grid, user_bcs)
    parameters = TKETopBoundaryConditionParameters(top_tracer_bcs, top_velocity_bcs)
    top_tke_bc = FluxBoundaryCondition(top_tke_flux, discrete_form=true, parameters=parameters)

    if :e ∈ keys(user_bcs)
        e_bcs = user_bcs[:e]
        
        tke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center),
                                          top = top_tke_bc,
                                          bottom = e_bcs.bottom,
                                          north = e_bcs.north,
                                          south = e_bcs.south,
                                          east = e_bcs.east,
                                          west = e_bcs.west)
    else
        tke_bcs = FieldBoundaryConditions(grid, (Center, Center, Center), top=top_tke_bc)
    end

    new_boundary_conditions = merge(user_bcs, (; e = tke_bcs))

    return new_boundary_conditions
end

Base.summary(::CATKEEquation) = "TKEBasedVerticalDiffusivities.CATKEEquation"
Base.show(io::IO, tke::CATKEEquation) =
    print(io, "TKEBasedVerticalDiffusivities.CATKEEquation parameters:", '\n',
              "├── CʰⁱD: ", tke.CʰⁱD, '\n',
              "├── CˡᵒD: ", tke.CˡᵒD, '\n',
              "├── CᵘⁿD: ", tke.CᵘⁿD, '\n',
              "├── CᶜD:  ", tke.CᶜD,  '\n',
              "├── CᵉD:  ", tke.CᵉD,  '\n',
              "├── Cᵂu★: ", tke.Cᵂu★, '\n',
              "└── CᵂwΔ: ", tke.CᵂwΔ)

