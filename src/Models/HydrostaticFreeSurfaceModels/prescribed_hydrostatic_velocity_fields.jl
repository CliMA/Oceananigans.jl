#####
##### PrescribedVelocityFields
#####

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: AbstractField, FunctionField, flatten_tuple
using Oceananigans.TimeSteppers: tick!, step_lagrangian_particles!

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models: extract_boundary_conditions
import Oceananigans.Utils: datatuple, sum_of_velocities
import Oceananigans.TimeSteppers: time_step!

using Adapt

struct PrescribedVelocityFields{U, V, W, P}
    u :: U
    v :: V
    w :: W
    parameters :: P
end

@inline Base.getindex(U::PrescribedVelocityFields, i) = getindex((u=U.u, v=U.v, w=U.w), i)

"""
    PrescribedVelocityFields(; u = ZeroField(),
                               v = ZeroField(),
                               w = ZeroField(),
                               parameters = nothing)

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
function PrescribedVelocityFields(; u = ZeroField(),
                                    v = ZeroField(),
                                    w = ZeroField(),
                                    parameters = nothing)

    return PrescribedVelocityFields(u, v, w, parameters)
end

wrap_prescribed_field(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)
wrap_prescribed_field(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

function hydrostatic_velocity_fields(velocities::PrescribedVelocityFields, grid, clock, bcs)

    parameters = velocities.parameters
    u = wrap_prescribed_field(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = wrap_prescribed_field(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = wrap_prescribed_field(Center, Center, Face, velocities.w, grid; clock, parameters)

    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(w)
    prescribed_velocities = (; u, v, w)
    @apply_regionally replace_horizontal_vector_halos!(prescribed_velocities, grid)

    return PrescribedVelocityFields(u, v, w, parameters)
end

hydrostatic_tendency_fields(::PrescribedVelocityFields, free_surface, grid, tracer_names, bcs) = 
    merge((u=nothing, v=nothing), TracerFields(tracer_names, grid))

free_surface_names(free_surface, ::PrescribedVelocityFields, grid) = tuple()
free_surface_names(::SplitExplicitFreeSurface, ::PrescribedVelocityFields, grid) = tuple()

@inline fill_halo_regions!(::PrescribedVelocityFields, args...) = nothing
@inline fill_halo_regions!(::FunctionField, args...) = nothing

@inline datatuple(obj::PrescribedVelocityFields) = (; u = datatuple(obj.u), v = datatuple(obj.v), w = datatuple(obj.w))
@inline velocities(obj::PrescribedVelocityFields) = (u = obj.u, v = obj.v, w = obj.w)

# Extend sum_of_velocities for `PrescribedVelocityFields`
@inline sum_of_velocities(U1::PrescribedVelocityFields, U2) = sum_of_velocities(velocities(U1), U2)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields) = sum_of_velocities(U1, velocities(U2))

@inline sum_of_velocities(U1::PrescribedVelocityFields, U2, U3) = sum_of_velocities(velocities(U1), U2, U3)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields, U3) = sum_of_velocities(U1, velocities(U2), U3)
@inline sum_of_velocities(U1, U2, U3::PrescribedVelocityFields) = sum_of_velocities(U1, U2, velocities(U3))

ab2_step_velocities!(::PrescribedVelocityFields, args...) = nothing
rk3_substep_velocities!(::PrescribedVelocityFields, args...) = nothing
step_free_surface!(::Nothing, model, timestepper, Δt) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, args...; kwargs...) = nothing

validate_velocity_boundary_conditions(grid, ::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

free_surface_displacement_field(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing

materialize_free_surface(::Nothing,                      ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = tracers
compute_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields, kernel_parameters; kwargs...) = nothing

apply_flux_bcs!(::Nothing, c, arch, clock, model_fields) = nothing

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(Adapt.adapt(to, velocities.u),
                             Adapt.adapt(to, velocities.v),
                             Adapt.adapt(to, velocities.w),
                             nothing) # Why are parameters not passed here? They probably should...

on_architecture(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(on_architecture(to, velocities.u),
                             on_architecture(to, velocities.v),
                             on_architecture(to, velocities.w),
                             on_architecture(to, velocities.parameters))

# If the model only tracks particles... do nothing but that!!!
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, G, T, V, B, R, F, P, U, C} where
                 {TS, E, A, S, G, T, V, B, R, F, P<:AbstractLagrangianParticles, U<:PrescribedVelocityFields, C<:NamedTuple{(), Tuple{}}}

function time_step!(model::OnlyParticleTrackingModel, Δt; callbacks = [], kwargs...)
    tick!(model.clock, Δt)
    model.clock.last_Δt = Δt
    step_lagrangian_particles!(model, Δt)
    update_state!(model, callbacks)
end

update_state!(model::OnlyParticleTrackingModel, callbacks) =
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]
