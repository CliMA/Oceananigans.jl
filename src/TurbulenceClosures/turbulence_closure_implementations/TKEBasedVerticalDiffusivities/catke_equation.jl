import Oceananigans.TurbulenceClosures: closure_source_term

@inline function closure_source_term(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, ::Val{:e},
                                     velocities, tracers, buoyancy)

    P  = shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    wb =    buoyancy_flux(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)
    ϵ  =      dissipation(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)

    return P + wb - ϵ
end

"""
    struct CATKEEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct CATKEEquation{FT}
    CˡᵒD :: FT = 2.46  # Dissipation length scale shear coefficient for low Ri
    CʰⁱD :: FT = 0.983 # Dissipation length scale shear coefficient for high Ri
    CᶜD  :: FT = 2.75  # Dissipation length scale convecting layer coefficient
    CᵉD  :: FT = 0.0   # Dissipation length scale penetration layer coefficient
    Cᵂu★ :: FT = 0.059 # Surface shear-driven TKE flux coefficient
    CᵂwΔ :: FT = 0.572 # Surface convective TKE flux coefficient
    Cᵂϵ  :: FT = 1.0   # Dissipative near-bottom TKE flux coefficient
end

@inline function shear_production(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, velocities, tracers, buoyancy)
    u = velocities.u
    v = velocities.v
    κu = diffusivity_fields.κu
    return shear_production(i, j, k, grid, κu, u, v)
end

#=
# Non-conservative reconstruction of buoyancy flux:
@inline function explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields)
    closure = getclosure(i, j, closure)
    κc = κcᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields.Jᵇ)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - κc * N²
end
=#

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, velocities, tracers, buoyancy) =
    explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)

const VITD = VerticallyImplicitTimeDiscretization

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, diffusivity_fields, velocities, tracers, buoyancy)
    wb = explicit_buoyancy_flux(i, j, k, grid, tracers, buoyancy, diffusivity_fields.κc)

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    wb = max(wb, zero(grid)) 

    return wb
end

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(grid)

@inline function dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers,
                                             buoyancy, Jᵇ, hc)

    # Convective dissipation length
    Cᶜ = closure.turbulent_kinetic_energy_equation.CᶜD
    Cᵉ = closure.turbulent_kinetic_energy_equation.CᵉD
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓh = convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, Jᵇ, hc)

    # "Stable" dissipation length
    Cˡᵒ = closure.turbulent_kinetic_energy_equation.CˡᵒD
    Cʰⁱ = closure.turbulent_kinetic_energy_equation.CʰⁱD
    ℓ★ = stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    σ = stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    return max(ℓ★, ℓh) / σ
end

@inline function dissipation_rate(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivity_fields)
    Jᵇ = diffusivity_fields.Jᵇ
    hc = diffusivity_fields.hc
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ, hc)
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]
    
    # Note:
    #   Because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   then      L e = - ϵ
    #                 = - Cᴰ e³² / ℓ
    #
    #   and thus    L = - Cᴰ √e / ℓ .

    ωn = 1 / closure.negative_turbulent_kinetic_energy_damping_time_scale
    ωp = sqrt(abs(eⁱʲᵏ)) / ℓᴰ

    return ifelse(eⁱʲᵏ < 0, ωn, ωp)
end

# Fallbacks for explicit time discretization
@inline function dissipation(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, velocities, tracers, buoyancy)
    eᵢ = @inbounds tracers.e[i, j, k]
    ω = dissipation_rate(i, j, k, grid, closure, velocities, tracers, args...)
    return ω * eᵢ
end

#####
##### TKE top boundary condition
#####

""" Compute the flux of TKE through the surface / top boundary. """
@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfCATKE, buoyancy)
    closure = getclosure(i, j, closure)

    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    tke_parameters = closure.turbulent_kinetic_energy_equation

    return _top_tke_flux(i, j, grid, clock, fields, tke_parameters, closure,
                         buoyancy, top_tracer_bcs, top_velocity_bcs)
end

""" Compute the flux of TKE through the surface / top boundary. """
@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure, buoyancy) = zero(grid)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any, <:Any}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any, <:Any, <:Any}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[3], buoyancy)

@inline function _top_tke_flux(i, j, grid, clock, fields,
                               tke::CATKEEquation, closure::CATKEVD,
                               buoyancy, top_tracer_bcs, top_velocity_bcs)

    wΔ³ = top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᵂu★ = tke.Cᵂu★
    CᵂwΔ = tke.CᵂwΔ

    return - Cᵂu★ * u★^3 - CᵂwΔ * wΔ³
end

""" Computes the friction velocity u★ based on fluxes of u and v. """
@inline function friction_velocity(i, j, grid, clock, fields, velocity_bcs)
    Jᵘ = getbc(velocity_bcs.u, i, j, grid, clock, fields) 
    Jᵛ = getbc(velocity_bcs.v, i, j, grid, clock, fields) 
    return sqrt(sqrt(Jᵘ^2 + Jᵛ^2))
end

""" Computes the convective velocity w★. """
@inline function top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, tracer_bcs)
    Jᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, fields)
    Δz = Δzᶜᶜᶜ(i, j, grid.Nz, grid)
    return clip(Jᵇ) * Δz   
end

struct TKETopBoundaryConditionParameters{C, U}
    top_tracer_boundary_conditions :: C
    top_velocity_boundary_conditions :: U
end

@inline Adapt.adapt_structure(to, p::TKETopBoundaryConditionParameters) =
    TKETopBoundaryConditionParameters(adapt(to, p.top_tracer_boundary_conditions),
                                      adapt(to, p.top_velocity_boundary_conditions))

@inline on_architecture(to, p::TKETopBoundaryConditionParameters) =
    TKETopBoundaryConditionParameters(on_architecture(to, p.top_tracer_boundary_conditions),
                                      on_architecture(to, p.top_velocity_boundary_conditions))


using Oceananigans.BoundaryConditions: Flux
const TKEBoundaryFunction = DiscreteBoundaryFunction{<:TKETopBoundaryConditionParameters}
const TKEBoundaryCondition = BoundaryCondition{<:Flux, <:TKEBoundaryFunction}

@inline getbc(bc::TKEBoundaryCondition, i::Integer, j::Integer, grid::AbstractGrid, clock, fields, clo, buoyancy) =
    bc.condition.func(i, j, grid, clock, fields, bc.condition.parameters, clo, buoyancy)

@inline getbc(bc::TKEBoundaryCondition, i::Integer, j::Integer, k::Integer, grid::AbstractGrid, clock, fields, clo, buoyancy) =
    bc.condition.func(i, j, k, grid, clock, fields, bc.condition.parameters, clo, buoyancy)

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

Base.summary(::CATKEEquation) = "CATKEEquation"
Base.show(io::IO, tke::CATKEEquation) =
    print(io, "CATKEEquation parameters: \n" *
              "    CˡᵒD: $(tke.CˡᵒD),  \n" *
              "    CʰⁱD: $(tke.CʰⁱD),  \n" *
              "    CᶜD:  $(tke.CᶜD),  \n" *
              "    CᵉD:  $(tke.CᵉD),  \n" *
              "    Cᵂu★: $(tke.Cᵂu★), \n" *
              "    CᵂwΔ: $(tke.CᵂwΔ)")

