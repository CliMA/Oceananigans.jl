#####
##### PrescribedVelocityFields
#####

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: AbstractField, FunctionField
using Oceananigans.TimeSteppers: tick!
using Oceananigans.LagrangianParticleTracking: update_particle_properties!

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models.NonhydrostaticModels: extract_boundary_conditions
import Oceananigans.Utils: datatuple
import Oceananigans.TimeSteppers: time_step!

using Adapt

struct PrescribedVelocityFields{U, V, W, P}
    u :: U
    v :: V
    w :: W
    parameters :: P
end

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

zerofunc(args...) = 0

"""
    PrescribedVelocityFields(; u=zerofunc, v=zerofunc, w=zerofunc, parameters=nothing)

Builds `PrescribedVelocityFields` with prescribed functions `u`, `v`, and `w`.

If `isnothing(parameters)`, then `u, v, w` are called with the signature

```
u(x, y, z, t) = # something interesting
```

If `!isnothing(parameters)`, then `u, v, w` are called with the signature

```
u(x, y, z, t, parameters) = # something parameterized and interesting
```

In the constructor for `HydrostaticFreeSurfaceModel`, the functions `u, v, w` are wrapped
in `FunctionField` and associated with the model's `grid` and `clock`.
"""
PrescribedVelocityFields(; u=zerofunc, v=zerofunc, w=zerofunc, parameters=nothing) =
    PrescribedVelocityFields(u, v, w, parameters)

PrescribedField(X, Y, Z, f::Function,      grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)
PrescribedField(X, Y, Z, f::AbstractField, grid; kwargs...) = f

function PrescribedField(X, Y, Z, f::Field, grid; kwargs...)
    fill_halo_regions!(f)
    return f
end

function HydrostaticFreeSurfaceVelocityFields(velocities::PrescribedVelocityFields, grid, clock, bcs)

    u = PrescribedField(Face, Center, Center, velocities.u, grid; clock=clock, parameters=velocities.parameters)
    v = PrescribedField(Center, Face, Center, velocities.v, grid; clock=clock, parameters=velocities.parameters)
    w = PrescribedField(Center, Center, Face, velocities.w, grid; clock=clock, parameters=velocities.parameters)

    return PrescribedVelocityFields(u, v, w, velocities.parameters)
end

function HydrostaticFreeSurfaceTendencyFields(::PrescribedVelocityFields, free_surface, grid, tracer_names)
    tracers = TracerFields(tracer_names, grid)
    return merge((u = nothing, v = nothing, η = nothing), tracers)
end

@inline fill_halo_regions!(::PrescribedVelocityFields, args...) = nothing
@inline fill_halo_regions!(::FunctionField, args...) = nothing

@inline datatuple(obj::PrescribedVelocityFields) = (; u = datatuple(obj.u), v = datatuple(obj.v), w = datatuple(obj.w))

ab2_step_velocities!(::PrescribedVelocityFields, args...) = [NoneEvent()]
ab2_step_free_surface!(::Nothing, args...) = NoneEvent()
compute_w_from_continuity!(::PrescribedVelocityFields, args...) = nothing

validate_velocity_boundary_conditions(::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

FreeSurfaceDisplacementField(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing
FreeSurface(free_surface::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
FreeSurface(free_surface::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = tracers
calculate_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields; kwargs...) = []

apply_flux_bcs!(::Nothing, c, arch, events, barrier, clock, model_fields) = nothing

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(Adapt.adapt(to, velocities.u),
                             Adapt.adapt(to, velocities.v),
                             Adapt.adapt(to, velocities.w),
                             nothing)

# If the model only tracks particles... do nothing but that!!!
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, G, T, V, B, R, F, P, U, C} where
                 {TS, E, A, S, G, T, V, B, R, F, P<:LagrangianParticles, U<:PrescribedVelocityFields, C<:NamedTuple{(), Tuple{}}}                 

function time_step!(model::OnlyParticleTrackingModel, Δt; euler=false) 
    model.timestepper.previous_Δt = Δt
    tick!(model.clock, Δt)
    update_particle_properties!(model, Δt)
end