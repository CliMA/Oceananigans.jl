import Oceananigans.TurbulenceClosures: closure_source_term

@inline closure_source_term(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, ::Val{:e}, velocities, tracers, buoyancy) =
    shear_production(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy) + 
       buoyancy_flux(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy) -
         dissipation(i, j, k, grid, closure, diffusivity_fields, velocities, tracers, buoyancy)

"""
    struct TurbulentKineticEnergyEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct TurbulentKineticEnergyEquation{FT}
    CˡᵒD  :: FT = 2.46  # Dissipation length scale shear coefficient for low Ri
    CʰⁱD  :: FT = 0.983 # Dissipation length scale shear coefficient for high Ri
    CᶜD   :: FT = 2.75  # Dissipation length scale convecting layer coefficient
    CᵉD   :: FT = 0.0   # Dissipation length scale penetration layer coefficient
    Cᵂu★  :: FT = 0.059 # Surface shear-driven TKE flux coefficient
    CᵂwΔ  :: FT = 0.572 # Surface convective TKE flux coefficient
    Cᵂϵ   :: FT = 1.0   # Dissipative near-bottom TKE flux coefficient
end

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

# Note special attention paid to averaging the vertical grid spacing correctly
@inline Δz_κᵘ_∂z_u²ᶠᶜᶠ(i, j, k, grid, κᵘ, u) = ℑxᶠᵃᵃ(i, j, k, grid, κᵘ) * Δzᶠᶜᶠ(i, j, k, grid) * ∂zᶠᶜᶠ(i, j, k, grid, u)^2
@inline Δz_κᵘ_∂z_v²ᶜᶠᶠ(i, j, k, grid, κᵘ, v) = ℑyᵃᶠᵃ(i, j, k, grid, κᵘ) * Δzᶜᶠᶠ(i, j, k, grid) * ∂zᶜᶠᶠ(i, j, k, grid, v)^2

@inline κᵘ_∂z_u²ᶠᶜᶜ(i, j, k, grid, κᵘ, u) = ℑzᵃᵃᶜ(i, j, k, grid, Δz_κᵘ_∂z_u²ᶠᶜᶠ, κᵘ, u) / Δzᶠᶜᶜ(i, j, k, grid) 
@inline κᵘ_∂z_v²ᶜᶠᶜ(i, j, k, grid, κᵘ, v) = ℑzᵃᵃᶜ(i, j, k, grid, Δz_κᵘ_∂z_v²ᶜᶠᶠ, κᵘ, v) / Δzᶜᶠᶜ(i, j, k, grid) 

@inline function shear_production(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, velocities, tracers, buoyancy)
    u = velocities.u
    v = velocities.v
    κᵘ = diffusivity_fields.κᵘ

    # Reconstruct the shear production term in an "approximately conservative" manner
    # (ie respecting the spatial discretization and using a stencil commensurate with the
    # loss of mean kinetic energy due to shear production --- but _not_ respecting the 
    # the temporal discretization. Note that also respecting the temporal discretization, would
    # require storing the velocity field at n and n+1):

    return ℑxᶜᵃᵃ(i, j, k, grid, κᵘ_∂z_u²ᶠᶜᶜ, κᵘ, u) +
           ℑyᵃᶜᵃ(i, j, k, grid, κᵘ_∂z_v²ᶜᶠᶜ, κᵘ, v)

    #=
    # Non-conservative reconstructions of shear production:
    closure = getclosure(i, j, closure)
    κᵘ = κuᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields.Qᵇ)
    S² = shearᶜᶜᶜ(i, j, k, grid, u, v)
    return κᵘ * S²
    =#
end

# To reconstruct buoyancy flux "conservatively" (ie approximately correpsonding to production/destruction
# of mean potential energy):
@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, diffusivity_fields)
    κᶜ = @inbounds diffusivity_fields.κᶜ[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κᶜ * N²
end
 
@inline explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivity_fields)

#=
# Non-conservative reconstruction of buoyancy flux:
@inline function explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields)
    closure = getclosure(i, j, closure)
    κᶜ = κcᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields.Qᵇ)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - κᶜ * N²
end
=#

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, diffusivity_fields, velocities, tracers, buoyancy) =
    explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields)

const VITD = VerticallyImplicitTimeDiscretization

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, diffusivity_fields, velocities, tracers, buoyancy)
    wb = explicit_buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields)
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]

    closure_ij = getclosure(i, j, closure)
    eᵐⁱⁿ = closure_ij.minimum_turbulent_kinetic_energy

    dissipative_buoyancy_flux = (sign(wb) * sign(eⁱʲᵏ) < 0) & (eⁱʲᵏ > eᵐⁱⁿ)

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    return ifelse(dissipative_buoyancy_flux, zero(grid), wb)
end

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(grid)

@inline function dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers,
                                             buoyancy, surface_buoyancy_flux)

    # Convective dissipation length
    Cᶜ = closure.turbulent_kinetic_energy_equation.CᶜD
    Cᵉ = closure.turbulent_kinetic_energy_equation.CᵉD
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    Qᵇ = surface_buoyancy_flux
    ℓʰ = convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, Qᵇ)

    # "Stable" dissipation length
    Cˡᵒ = closure.turbulent_kinetic_energy_equation.CˡᵒD
    Cʰⁱ = closure.turbulent_kinetic_energy_equation.CʰⁱD
    σᴰ = stability_functionᶜᶜᶜ(i, j, k, grid, closure, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)
    ℓ★ = stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)
    #ℓ★ = ℓ★ / σᴰ
    ℓ★ = ℓ★ * σᴰ

    # Dissipation length
    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓᴰ = max(ℓ★, ℓʰ)

    H = total_depthᶜᶜᵃ(i, j, grid)

    return min(H, ℓᴰ)
end

@inline function dissipation_rate(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivity_fields)
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivity_fields.Qᵇ)
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

    ω_numerical = 1 / closure.negative_turbulent_kinetic_energy_damping_time_scale
    ω_physical = sqrt(abs(eᵢ)) / ℓᴰ

    return ifelse(eᵢ < 0, ω_numerical, ω_physical)
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
                               tke::TurbulentKineticEnergyEquation, closure::CATKEVD,
                               buoyancy, top_tracer_bcs, top_velocity_bcs)

    wΔ³ = top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᵂu★ = tke.Cᵂu★
    CᵂwΔ = tke.CᵂwΔ

    return - Cᵂu★ * u★^3 - CᵂwΔ * wΔ³
end

""" Computes the friction velocity u★ based on fluxes of u and v. """
@inline function friction_velocity(i, j, grid, clock, fields, velocity_bcs)
    FT = eltype(grid)
    Qᵘ = getbc(velocity_bcs.u, i, j, grid, clock, fields) 
    Qᵛ = getbc(velocity_bcs.v, i, j, grid, clock, fields) 
    return sqrt(sqrt(Qᵘ^2 + Qᵛ^2))
end

""" Computes the convective velocity w★. """
@inline function top_convective_turbulent_velocity_cubed(i, j, grid, clock, fields, buoyancy, tracer_bcs)
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, fields)
    Δz = Δzᶜᶜᶜ(i, j, grid.Nz, grid)
    return clip(Qᵇ) * Δz   
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

#####
##### Utilities for model constructors
#####

""" Infer tracer boundary conditions from user_bcs and tracer_names. """
function top_tracer_boundary_conditions(grid, tracer_names, user_bcs)
    default_tracer_bcs = NamedTuple(c => FieldBoundaryConditions(grid, (Center, Center, Center)) for c in tracer_names)
    bcs = merge(default_tracer_bcs, user_bcs)
    return NamedTuple(c => bcs[c].top for c in tracer_names)
end

""" Infer velocity boundary conditions from `user_bcs` and `tracer_names`. """
function top_velocity_boundary_conditions(grid, user_bcs)
    default_top_bc = default_prognostic_bc(topology(grid, 3)(), Center(), DefaultBoundaryCondition())

    user_bc_names = keys(user_bcs)
    u_top_bc = :u ∈ user_bc_names ? user_bcs.u.top : default_top_bc
    v_top_bc = :v ∈ user_bc_names ? user_bcs.v.top : default_top_bc

    return (u=u_top_bc, v=v_top_bc)
end

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

Base.summary(::TurbulentKineticEnergyEquation) = "CATKEVerticalDiffusivities.TurbulentKineticEnergyEquation"
Base.show(io::IO, tke::TurbulentKineticEnergyEquation) =
    print(io, "CATKEVerticalDiffusivities.TurbulentKineticEnergyEquation parameters: \n" *
              "    CˡᵒD: $(tke.CˡᵒD),  \n" *
              "    CʰⁱD: $(tke.CʰⁱD),  \n" *
              "    CᶜD:  $(tke.CᶜD),  \n" *
              "    CᵉD:  $(tke.CᵉD),  \n" *
              "    Cᵂu★: $(tke.Cᵂu★), \n" *
              "    CᵂwΔ: $(tke.CᵂwΔ)")

