"""
    struct SurfaceTKEFlux{FT}

A model for the flux of TKE across the numerical ocean surface with
free parameters of type `FT`, parameterized in
terms of the kinematic surface stress and buoyancy flux:

```math
Qᵉ = - Cᴰ * (Cᵂu★ * u★³ + CᵂwΔ * w★³)
```

where `Qᵉ` is the surface flux of TKE, `Cᴰ = CATKEVerticalDiffusivity.Cᴰ`,
`u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)` is the friction velocity and `w★ = (Qᵇ * Δz)^(1/3)` is the
turbulent velocity scale associated with the surface vertical grid spacing `Δz` and the
surface buoyancy flux `Qᵇ`.
             
The 2 free parameters in `SurfaceTKEFlux` have been _experimentally_ calibrated
against large eddy simulations of ocean surface boundary layer turbulence in idealized
scenarios involving monotonic boundary layer deepening into variable stratification
due to constant surface momentum fluxes and/or destabilizing surface buoyancy flux.
See https://github.com/CliMA/LESbrary.jl for more information about the large eddy simulations.
The calibration was performed using a combination of Markov Chain Monte Carlo (MCMC)-based simulated
annealing and noisy Ensemble Kalman Inversion methods.
"""
Base.@kwdef struct SurfaceTKEFlux{FT}
    Cᵂu★ :: FT = 3.62
    CᵂwΔ :: FT = 1.31
end

#####
##### TKE top boundary condition
#####

""" Compute the flux of TKE through the surface / top boundary. """
@inline function top_tke_flux(i, j, grid, clock, fields, parameters, closure::Union{CATKEVD, CATKEVDArray}, buoyancy)
    top_tracer_bcs = parameters.top_tracer_boundary_conditions
    top_velocity_bcs = parameters.top_velocity_boundary_conditions

    return _top_tke_flux(i, j, grid, closure.surface_TKE_flux, closure,
                         buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)
end

""" Compute the flux of TKE through the surface / top boundary. """
@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure, buoyancy) = 0

@inline top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple::Tuple, buoyancy) =
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[1], buoyancy) + 
    top_tke_flux(i, j, grid, clock, fields, parameters, closure_tuple[2:end], buoyancy)

@inline function _top_tke_flux(i, j, grid, surface_TKE_flux::SurfaceTKEFlux, closure,
                              buoyancy, fields, top_tracer_bcs, top_velocity_bcs, clock)

    wΔ³ = top_convective_turbulent_velocity³(i, j, grid, clock, fields, buoyancy, top_tracer_bcs)
    u★ = friction_velocity(i, j, grid, clock, fields, top_velocity_bcs)

    Cᴰ = closure.Cᴰ
    Cᵂu★ = surface_TKE_flux.Cᵂu★
    CᵂwΔ = surface_TKE_flux.CᵂwΔ

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

@inline getbc(bc::TKEBoundaryCondition, i, j, grid, clock, model_fields, closure, buoyancy) =
    bc.condition.func(i, j, grid, clock, model_fields, bc.condition.parameters, closure, buoyancy)

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

    default_top_bc = default_prognostic_field_boundary_condition(topology(grid, 3)(), Center())

    user_bc_names = keys(user_bcs)
    u_top_bc = :u ∈ user_bc_names ? user_bcs.u.top : default_top_bc
    v_top_bc = :v ∈ user_bc_names ? user_bcs.v.top : default_top_bc

    return (u=u_top_bc, v=v_top_bc)
end

""" Add TKE boundary conditions specific to `CATKEVerticalDiffusivity`. """
function add_closure_specific_boundary_conditions(closure::Union{CATKEVD, CATKEVDArray},
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

Base.show(io::IO, SurfaceTKEFlux::SurfaceTKEFlux) =
    print(io, "SurfaceTKEFlux: \n" *
              "         Cᵂu★ = $(SurfaceTKEFlux.Cᵂu★), \n" *
              "         CᵂwΔ = $(SurfaceTKEFlux.CᵂwΔ)")
