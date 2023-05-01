"""
    struct TurbulentKineticEnergyEquation{FT}

Parameters for the evolution of oceanic turbulent kinetic energy at the O(1 m) scales associated with
isotropic turbulence and diapycnal mixing.
"""
Base.@kwdef struct TurbulentKineticEnergyEquation{FT}
    C⁻D   :: FT = 2.3
    C⁺D   :: FT = 6.7
    CᶜD   :: FT = 0.88
    CᵉD   :: FT = 0.0
    Cᵂu★  :: FT = 1.1
    CᵂwΔ  :: FT = 4.0
    Cᵂϵ   :: FT = 1.0
end

#####
##### Terms in the turbulent kinetic energy equation, all at cell centers
#####

@inline ν_∂z_u²(i, j, k, grid, ν, u) = ℑxᶠᵃᵃ(i, j, k, grid, ν) * ∂zᶠᶜᶠ(i, j, k, grid, u)^2
@inline ν_∂z_v²(i, j, k, grid, ν, v) = ℑyᵃᶠᵃ(i, j, k, grid, ν) * ∂zᶜᶠᶠ(i, j, k, grid, v)^2

@inline function shear_production(i, j, k, grid, closure::FlavorOfCATKE, velocities, diffusivities)
    κᵘ = diffusivities.κᵘ
    u = velocities.u
    v = velocities.v

    # Separate reconstruction of the u- and v- contributions is essential for numerical stability
    return ℑxzᶜᵃᶜ(i, j, k, grid, ν_∂z_u², κᵘ, u) + ℑyzᵃᶜᶜ(i, j, k, grid, ν_∂z_v², κᵘ, v)
end

@inline function buoyancy_fluxᶜᶜᶠ(i, j, k, grid, tracers, buoyancy, diffusivities)
    κᶻ = @inbounds diffusivities.κᶜ[i, j, k]
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return - κᶻ * N²
end

@inline buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, diffusivities) =
    ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)

const VITD = VerticallyImplicitTimeDiscretization

@inline function buoyancy_flux(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, velocities, tracers, buoyancy, diffusivities)
    wb = ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)
    eⁱʲᵏ = @inbounds tracers.e[i, j, k]

    dissipative_buoyancy_flux = sign(wb) * sign(eⁱʲᵏ) < 0

    # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
    # If buoyancy flux is a _sink_ of TKE, we treat it implicitly, and return zero here for
    # the explicit buoyancy flux.
    return ifelse(dissipative_buoyancy_flux, zero(grid), wb)
end

@inline dissipation(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, args...) = zero(grid)

@inline function implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE,
                                                  velocities, tracers, buoyancy, clock, tracer_bcs)
    e = tracers.e
    FT = eltype(grid)

    # Convective dissipation length
    Cᶜ = closure.turbulent_kinetic_energy_equation.CᶜD
    Cᵉ = closure.turbulent_kinetic_energy_equation.CᵉD
    Cˢᶜ = closure.mixing_length.Cˢᶜ
    ℓʰ = convective_length_scaleᶜᶜᶜ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᶜ, velocities, tracers, buoyancy, clock, tracer_bcs)

    # "Stable" dissipation length
    C⁻D = closure.turbulent_kinetic_energy_equation.C⁻D
    C⁺D = closure.turbulent_kinetic_energy_equation.C⁺D
    Riᶜ = closure.mixing_length.CRiᶜ
    Riʷ = closure.mixing_length.CRiʷ
    Ri = Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy)
    σ = scale(Ri, C⁻D, C⁺D, Riᶜ, Riʷ)
    ℓ★ = σ * stable_length_scaleᶜᶜᶜ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)

    # Dissipation length
    H = total_depthᶜᶜᵃ(i, j, grid)
    ℓᴰ = min(H, ℓ★ + ℓʰ)

    eᵢ = @inbounds e[i, j, k]
    
    # Note:
    #   Because   ∂t e + ⋯ = ⋯ + L e = ⋯ - ϵ,
    #
    #   then      L e = - ϵ
    #                 = - Cᴰ e³² / ℓ
    #
    #   and thus    L = - Cᴰ √e / ℓ .

    τ = closure.negative_turbulent_kinetic_energy_damping_time_scale

    return ifelse(eᵢ < 0, -1/τ, -sqrt(abs(eᵢ)) / ℓᴰ)
end

# Fallbacks for explicit time discretization
@inline function dissipation(i, j, k, grid, closure::FlavorOfCATKE, velocities, tracers, args...)
    eᵢ = @inbounds tracers.e[i, j, k]
    L = implicit_dissipation_coefficient(i, j, k, grid, closure, velocities, tracers, args...)
    return L * eᵢ
end

@inline implicit_dissipation_coefficient(i, j, k, grid, closure::FlavorOfCATKE, args...) = zero(grid)

#####
##### For closure tuples...
#####

# TODO: include shear production and buoyancy flux from AbstractScalarDiffusivity

@inline shear_production(i, j, k, grid, closure, velocities, diffusivities) = zero(grid)

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1])

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any, <:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1]) +
    shear_production(i, j, k, grid, closures[2], velocities, diffusivities[2])

@inline shear_production(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, velocities, diffusivities) =
    shear_production(i, j, k, grid, closures[1], velocities, diffusivities[1]) +
    shear_production(i, j, k, grid, closures[2], velocities, diffusivities[2]) +
    shear_production(i, j, k, grid, closures[3], velocities, diffusivities[3])

@inline buoyancy_flux(i, j, k, grid, closure, velocities, tracers, buoyancy, diffusivities) = zero(grid)

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, diffusivities[1])

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    buoyancy_flux(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2])

@inline buoyancy_flux(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, velocities, tracers, buoyancy, diffusivities) =
    buoyancy_flux(i, j, k, grid, closures[1], velocities, tracers, buoyancy, diffusivities[1]) +
    buoyancy_flux(i, j, k, grid, closures[2], velocities, tracers, buoyancy, diffusivities[2]) +
    buoyancy_flux(i, j, k, grid, closures[3], velocities, tracers, buoyancy, diffusivities[3])

# Unlike the above, this fallback for dissipation is generically correct (we only want to compute dissipation once)
@inline dissipation(i, j, k, grid, closure, args...) = zero(grid)

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any}, args...) = dissipation(i, j, k, grid, closures[1], args...)

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any}, args...) = 
    dissipation(i, j, k, grid, closures[1], args...) +
    dissipation(i, j, k, grid, closures[2], args...)

@inline dissipation(i, j, k, grid, closures::Tuple{<:Any, <:Any, <:Any}, args...) = 
    dissipation(i, j, k, grid, closures[1], args...) +
    dissipation(i, j, k, grid, closures[2], args...) +
    dissipation(i, j, k, grid, closures[3], args...)

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
              "    C⁻D  = $(tke.C⁻D),  \n" *
              "    C⁺D  = $(tke.C⁺D),  \n" *
              "    CᶜD  = $(tke.CᶜD),  \n" *
              "    CᵉD  = $(tke.CᵉD),  \n" *
              "    Cᵂu★ = $(tke.Cᵂu★), \n" *
              "    CᵂwΔ = $(tke.CᵂwΔ)")
