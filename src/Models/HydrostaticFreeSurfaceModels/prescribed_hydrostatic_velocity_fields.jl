#####
##### PrescribedVelocityFields
#####

using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: AbstractField, FunctionField, flatten_tuple
using Oceananigans.TimeSteppers: tick!, step_lagrangian_particles!

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models: extract_boundary_conditions
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
wrap_prescribed_field(X, Y, Z, f, grid; kwargs...) = f

function HydrostaticFreeSurfaceVelocityFields(velocities::PrescribedVelocityFields, grid, clock, bcs)

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

function HydrostaticFreeSurfaceTendencyFields(::PrescribedVelocityFields, free_surface, grid, tracer_names)
    tracer_tendencies = TracerFields(tracer_names, grid)
    momentum_tendencies = (u = nothing, v = nothing, η = nothing)
    return merge(momentum_tendencies, tracer_tendencies)
end

function HydrostaticFreeSurfaceTendencyFields(::PrescribedVelocityFields, ::ExplicitFreeSurface, grid, tracer_names)
    tracers = TracerFields(tracer_names, grid)
    return merge((u = nothing, v = nothing, η = nothing), tracers)
end

@inline fill_halo_regions!(::PrescribedVelocityFields, args...) = nothing
@inline fill_halo_regions!(::FunctionField, args...) = nothing

@inline datatuple(obj::PrescribedVelocityFields) = (; u = datatuple(obj.u), v = datatuple(obj.v), w = datatuple(obj.w))

ab2_step_velocities!(::PrescribedVelocityFields, args...) = nothing
ab2_step_free_surface!(::Nothing, model, Δt, χ) = nothing 
compute_w_from_continuity!(::PrescribedVelocityFields, args...; kwargs...) = nothing

validate_velocity_boundary_conditions(grid, ::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

FreeSurfaceDisplacementField(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing

FreeSurface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
FreeSurface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
FreeSurface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = tracers
compute_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields, kernel_parameters; kwargs...) = nothing

apply_flux_bcs!(::Nothing, c, arch, clock, model_fields) = nothing

Adapt.adapt_structure(to, velocities::PrescribedVelocityFields) =
    PrescribedVelocityFields(Adapt.adapt(to, velocities.u),
                             Adapt.adapt(to, velocities.v),
                             Adapt.adapt(to, velocities.w),
                             nothing)

# If the model only tracks particles... do nothing but that!!!
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, G, T, V, B, R, F, P, U, C} where
                 {TS, E, A, S, G, T, V, B, R, F, P<:AbstractLagrangianParticles, U<:PrescribedVelocityFields, C<:NamedTuple{(), Tuple{}}}                 

function time_step!(model::OnlyParticleTrackingModel, Δt; callbacks = [], kwargs...) 
    model.timestepper.previous_Δt = Δt
    tick!(model.clock, Δt)
    step_lagrangian_particles!(model, Δt)
    update_state!(model, Δt, callbacks)
end

update_state!(model::OnlyParticleTrackingModel, Δt, callbacks) = 
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]
