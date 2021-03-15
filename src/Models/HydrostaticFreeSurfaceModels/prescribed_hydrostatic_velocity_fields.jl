#####
##### PrescribedVelocityFields
#####

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: FunctionField

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models.IncompressibleModels: extract_boundary_conditions

using Adapt

struct PrescribedVelocityFields{U, V, W, P}
    u :: U
    v :: V
    w :: W
    parameters :: P
end

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

zerofunc(x, y, z) = 0

"""
    PrescribedVelocityFields(; u=zerofunc, v=zerofunc, w=zerofunc, parameters=nothing)

Builds `PrescribedVelocityFields` with prescribed functions `u`, `v`, and `w`.

If `isnothing(parameters)`, then `u, v, w` are called with the signature

    `u(x, y, z, t) = # something interesting`

If `!isnothing(parameters)`, then `u, v, w` are called with the signature

    `u(x, y, z, t, parameters) = # something parameterized and interesting`

In the constructor for `HydrostaticFreeSurfaceModel`, the functions `u, v, w` are wrapped
in `FunctionField` and associated with the model's `grid` and `clock`.
"""
PrescribedVelocityFields(; u=zerofunc, v=zerofunc, w=zerofunc, parameters=nothing) =
    PrescribedVelocityFields(u, v, w, parameters)

function HydrostaticFreeSurfaceVelocityFields(velocities::PrescribedVelocityFields,
                                              arch, grid, clock, bcs)

    u = FunctionField{Face, Center, Center}(velocities.u, grid; clock=clock, parameters=velocities.parameters)
    v = FunctionField{Center, Face, Center}(velocities.v, grid; clock=clock, parameters=velocities.parameters)
    w = FunctionField{Center, Center, Face}(velocities.w, grid; clock=clock, parameters=velocities.parameters)

    return PrescribedVelocityFields(u, v, w, velocities.parameters)
end

@inline fill_halo_regions!(::PrescribedVelocityFields, args...) = nothing
@inline fill_halo_regions!(::FunctionField, args...) = nothing

ab2_step_free_surface!(::Nothing, args...) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, args...) = nothing

validate_velocity_boundary_conditions(::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

FreeSurfaceDisplacementField(::PrescribedVelocityFields, arch, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, arch, grid) = nothing, nothing
FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, arch, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, free_surface, tracers) = tracers
calculate_hydrostatic_momentum_tendencies!(tendencies, ::PrescribedVelocityFields, args...) = []

apply_flux_bcs!(::Nothing, c, arch, events, barrier, clock, model_fields) = nothing

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(Adapt.adapt(to, velocities.u),
                             Adapt.adapt(to, velocities.v),
                             Adapt.adapt(to, velocities.w),
                             nothing)
