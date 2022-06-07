"""
    struct TurbulentKineticEnergyEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.

Turbulent kinetic energy dissipation
====================================

Surface flux model
==================

```math
Qᵉ = - Cᴰ * (Cᵂu★ * u★³ + CᵂwΔ * w★³)
```

where `Qᵉ` is the surface flux of TKE, `Cᴰ` is a free parameter called the "dissipation parameter",
`u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)` is the friction velocity and `w★ = (Qᵇ * Δz)^(1/3)` is the
turbulent velocity scale associated with the surface vertical grid spacing `Δz` and the
surface buoyancy flux `Qᵇ`.
"""
Base.@kwdef struct TurbulentKineticEnergyEquation{FT}
    Cᴰ⁻   :: FT = 1.70
    Cᴰʳ   :: FT = 6.34
    CᴰRiᶜ :: FT = 1.09
    CᴰRiʷ :: FT = 1.57
    Cᵂu★  :: FT = 9.90
    CᵂwΔ  :: FT = 8.26
end

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
    νᶻ = ℑzᵃᵃᶜ(i, j, k, grid, diffusivities.Kᵘ)
    return νᶻ * (∂z_u² + ∂z_v²)
end

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivities)
    κᶻ = ℑzᵃᵃᶜ(i, j, k, grid, diffusivities.Kᶜ)
    N² = ℑzᵃᵃᶜ(i, j, k, grid, ∂z_b, buoyancy, tracers)
    return - κᶻ * N²
end

const VITD = VerticallyImplicitTimeDiscretization

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(eltype(grid))

@inline function implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, velocities, tracers, buoyancy, clock, tracer_bcs)
    e = tracers.e
    FT = eltype(grid)

    # Tracer mixing length
    ℓ = ℑzᵃᵃᶜ(i, j, k, grid, tracer_mixing_lengthᶜᶜᶠ, closure, velocities, tracers, buoyancy, clock, tracer_bcs)

    # Ri-dependent dissipation coefficient
    Cᴰ⁻ = closure.turbulent_kinetic_energy_equation.Cᴰ⁻
    Cᴰʳ = closure.turbulent_kinetic_energy_equation.Cᴰʳ
    Riᶜ = closure.turbulent_kinetic_energy_equation.CᴰRiᶜ
    Riʷ = closure.turbulent_kinetic_energy_equation.CᴰRiʷ
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    Cᴰ = scale(Ri, Cᴰ⁻, Cᴰʳ, Riᶜ, Riʷ)

    eᵢ = @inbounds e[i, j, k]

    # Note:
    #   Because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   then      L e = - ϵ
    #                 = - Cᴰ e³² / ℓ
    #
    #   and thus    L = - Cᴰ √e / ℓ .
    #

    # Second note: negative values of e are unphysical.
    # Therefore we introduce a simple parameterization for destroying
    # negative e (below "k" denotes an inverse length scale):
    #k⁺ = Cᴰ / ℓ
    #k⁻ = 10 / Δzᶜᶜᶠ(i, j, k, grid)
    #k = ifelse(eᵢ > 0, k⁺, k⁻)

    return - Cᴰ * sqrt(abs(eᵢ)) / ℓ
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

#####
##### TKE top boundary condition
#####

""" Compute the flux of TKE through the surface / top boundary. """
@inline function _top_tke_flux(i, j, grid, clock, fields, parameters, closure::FlavorOfCATKE, buoyancy)
    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions
    closure = getclosure(i, j, closure)

    return top_tke_flux(i, j, grid, closure.turbulent_kinetic_energy_equation, closure,
                         buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)
end

""" Compute the flux of TKE through the surface / top boundary. """
@inline  top_tke_flux(i, j, grid, clock, fields, parameters, closure, buoyancy) = zero(grid)
@inline _top_tke_flux(args...) = top_tke_flux(args...)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:FlavorOfCATKE}, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple{<:Any, <:Any}, buoyancy) =
    _top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    _top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2], buoyancy)

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    _top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2:end], buoyancy)

@inline _top_tke_flux(i, j, grid, clock, fields, parameters, closure::Tuple, buoyancy) =
    _top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) +
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2:end], buoyancy) +

@inline function top_tke_flux(i, j, grid, tke::TurbulentKineticEnergyEquation, closure::FlavorOfCATKE,
                              buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)

    wΔ³ = top_convective_turbulent_velocity³(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᴰ = tke.Cᴰ
    Cᵂu★ = tke.Cᵂu★
    CᵂwΔ = tke.CᵂwΔ

    return - Cᴰ * (Cᵂu★ * u★^3 + CᵂwΔ * wΔ³)
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
    FT = eltype(grid)
    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, fields)
    Δz = Δzᶜᶜᶜ(i, j, grid.Nz, grid)
    return max(zero(FT), Qᵇ) * Δz   
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

    new_boundary_conditions = merge(user_bcs, (e = tke_bcs,))

    return new_boundary_conditions
end

Base.show(io::IO, tke::TurbulentKineticEnergyEquation) =
    print(io, "TurbulentKineticEnergyEquation: \n" *
              "          Cᴰ⁻: $(tke.Cᴰ⁻), \n" *
              "          Cᴰʳ: $(tke.Cᴰʳ), \n" *
              "         Cᵂu★: $(tke.Cᵂu★), \n" *
              "         CᵂwΔ: $(tke.CᵂwΔ)")
