"""
    struct TurbulentKineticEnergyEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct TurbulentKineticEnergyEquation{FT}
    C⁻D   :: FT = 1.0
    C⁺D   :: FT = 1.0
    CᶜD   :: FT = 0.0
    CᵉD   :: FT = 0.0
    Cᵂu★  :: FT = 1.0
    CᵂwΔ  :: FT = 1.0
end

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

@inline ϕ²(i, j, k, grid, ϕ) = ϕ(i, j, k, grid)^2

# Temporary way to get the vertical diffusivity for the TKE equation terms...
# Assumes that the vertical diffusivity is dominated by the CATKE contribution.
# TODO: include shear production and buoyancy flux from AbstractScalarDiffusivity
@inline shear_production(i, j, k, grid, closure, velocities, diffusivities) = zero(grid)
@inline buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities) = zero(grid)

# Unlike the above, this fallback for dissipation is generically correct (we only want to compute dissipation once)
@inline dissipation(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, tracer_bcs) = zero(grid)

@inline function shear_productionᶜᶜᶠ(i, j, k, grid, velocities, diffusivities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return @inbounds diffusivities.Kᵘ[i, j, k] * (∂z_u² + ∂z_v²)
end

@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, diffusivities)
    κᶻ = @inbounds diffusivities.Kᶜ[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κᶻ * N²
end

@inline shear_production(i, j, k, grid, closure::FlavorOfCATKE, velocities, diffusivities) =
    ℑzᵃᵃᶜ(i, j, k, grid, shear_productionᶜᶜᶠ, velocities, diffusivities)

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivities) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)

const VITD = VerticallyImplicitTimeDiscretization

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, velocities, tracers, buoyancy, diffusivities)
    wb = ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
    return ifelse(sign(wb) * sign(eⁱʲᵏ) < 0, zero(grid), wb)
end

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(grid)

@inline function implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD},
                                                  velocities, tracers, buoyancy, clock, tracer_bcs)
    e = tracers.e
    FT = eltype(grid)

    # Convective dissipation length
    Cᶜ = closure.turbulent_kinetic_energy_equation.CᶜD
    Cᵉ = closure.turbulent_kinetic_energy_equation.CᵉD
    Cˢᶜ = closure.mixing_length.Cˢᶜ
    ℓʰ = ℑzᵃᵃᶜ(i, j, k, grid, convective_mixing_lengthᶜᶜᶠ, Cᶜ, Cᵉ, Cˢᶜ, velocities, tracers, buoyancy, clock, tracer_bcs)

    # "Stable" dissipation length
    C⁻D = closure.turbulent_kinetic_energy_equation.C⁻D
    C⁺D = closure.turbulent_kinetic_energy_equation.C⁺D
    Riᶜ = closure.mixing_length.CRiᶜ
    Riʷ = closure.mixing_length.CRiʷ
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    σ = scale(Ri, C⁻D, C⁺D, Riᶜ, Riʷ)

    Cᵇ = closure.mixing_length.Cᵇ
    Cˢ = closure.mixing_length.Cˢ
    ℓ★ = σ * ℑzᵃᵃᶜ(i, j, k, grid, stable_mixing_lengthᶜᶜᶠ, Cᵇ, Cˢ, tracers.e, velocities, tracers, buoyancy)

    # Dissipation length
    ℓᴰ = ℓ★ + ℓʰ

    eᵢ = @inbounds e[i, j, k]

    # Note:
    #   Because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   then      L e = - ϵ
    #                 = - Cᴰ e³² / ℓ
    #
    #   and thus    L = - Cᴰ √e / ℓ .

    return - sqrt(abs(eᵢ)) / ℓᴰ
end

# Fallbacks for explicit time discretization
@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, args...) =
    @inbounds - tracers.e[i, j, k] * implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, args...)

@inline implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE, args...) = zero(grid)

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

#####
##### TKE top boundary condition
#####

""" Compute the flux of TKE through the surface / top boundary. """
@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfCATKE, buoyancy)
    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    closure = getclosure(i, j, closure)
    tke_parameters = closure.turbulent_kinetic_energy_equation

    return _top_tke_flux(i, j, grid, tke_parameters, closure,
                         buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)
end

""" Compute the flux of TKE through the surface / top boundary. """
@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure, buoyancy) = zero(grid)
@inline inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure, buoyancy) = zero(grid)

#=
@inline inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure::Tuple{}, buoyancy) = zero(grid)
@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure::Tuple{}, buoyancy) = zero(grid)

@inline inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:FlavorOfCATKE}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:FlavorOfCATKE}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any, <:Any}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2], buoyancy)

@inline inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any, <:Any}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2:end], buoyancy)

@inline inner_top_tke_flux(i, j, grid, clock, fields, parameters, closure::Tuple, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) +
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2:end], buoyancy) +
=#

@inline function _top_tke_flux(i, j, grid, tke::TurbulentKineticEnergyEquation, closure::CATKEVD,
                               buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)

    wΔ³ = top_convective_turbulent_velocity³(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
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
@inline function top_convective_turbulent_velocity³(i, j, grid, clock, fields, buoyancy, tracer_bcs)
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

using Oceananigans.BoundaryConditions: Flux
const TKEBoundaryFunction = DiscreteBoundaryFunction{<:TKETopBoundaryConditionParameters}
const TKEBoundaryCondition = BoundaryCondition{<:Flux, <:TKEBoundaryFunction}

@inline getbc(bc::TKEBoundaryCondition, i::Integer, j::Integer, grid::AbstractGrid, clock, model_fields, closure, buoyancy) =
    bc.condition.func(i, j, grid, clock, model_fields, bc.condition.parameters, closure, buoyancy)

@inline getbc(bc::TKEBoundaryCondition, i::Integer, j::Integer, k::Integer, grid::AbstractGrid, clock, model_fields, closure, buoyancy) =
    bc.condition.func(i, j, k, grid, clock, model_fields, bc.condition.parameters, closure, buoyancy)

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

Base.show(io::IO, tke::TurbulentKineticEnergyEquation) =
    print(io, "CATKEVerticalDiffusivities.TurbulentKineticEnergyEquation parameters: \n" *
              "    C⁻D  = $(tke.C⁻D),  \n" *
              "    C⁺D  = $(tke.C⁺D),  \n" *
              "    CᶜD  = $(tke.CᶜD),  \n" *
              "    CᵉD  = $(tke.CᵉD),  \n" *
              "    Cᵂu★ = $(tke.Cᵂu★), \n" *
              "    CᵂwΔ = $(tke.CᵂwΔ)")
