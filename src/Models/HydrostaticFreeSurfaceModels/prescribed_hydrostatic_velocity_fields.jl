#####
##### PrescribedVelocityFields
#####

using Oceananigans: location
using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: FunctionField, Field, field, TracerFields
using Oceananigans.TimeSteppers: tick!, step_lagrangian_particles!
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.OutputReaders: FieldTimeSeries, TimeSeriesInterpolation

import Oceananigans: prognostic_state, restore_prognostic_state!
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Models: extract_boundary_conditions
import Oceananigans.Utils: datatuple, sum_of_velocities
import Oceananigans.TimeSteppers: time_step!

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

materialize_prescribed_velocity(X, Y, Z, f::Function, grid; kwargs...) = FunctionField{X, Y, Z}(f, grid; kwargs...)

function materialize_prescribed_velocity(X, Y, Z, fts::FieldTimeSeries, grid; clock, kwargs...)
    fts_location = location(fts)
    requested_location = (X, Y, Z)
    if fts_location != requested_location
        throw(ArgumentError("FieldTimeSeries location $fts_location does not match " *
                            "the expected velocity location $requested_location"))
    end
    return TimeSeriesInterpolation(fts, grid; clock)
end

materialize_prescribed_velocity(X, Y, Z, f, grid; kwargs...) = field((X, Y, Z), f, grid)

function hydrostatic_velocity_fields(velocities::PrescribedVelocityFields, grid, clock, bcs)

    parameters = velocities.parameters
    u = materialize_prescribed_velocity(Face, Center, Center, velocities.u, grid; clock, parameters)
    v = materialize_prescribed_velocity(Center, Face, Center, velocities.v, grid; clock, parameters)
    w = materialize_prescribed_velocity(Center, Center, Face, velocities.w, grid; clock, parameters)

    fill_halo_regions!((u, v))
    fill_halo_regions!(w)

    return PrescribedVelocityFields(u, v, w, parameters)
end

# Allow u, v, w = velocities when velocities isa PrescribedVelocityFields
function Base.indexed_iterate(p::PrescribedVelocityFields, i::Int, state=1)
    if i == 1
        return p.u, 2
    elseif i == 2
        return p.v, 3
    else
        return p.w, 4
    end
end

hydrostatic_tendency_fields(::PrescribedVelocityFields, free_surface, grid, tracer_names, bcs) =
    merge((u=nothing, v=nothing), TracerFields(tracer_names, grid))

free_surface_names(free_surface, ::PrescribedVelocityFields, grid) = tuple()
free_surface_names(::SplitExplicitFreeSurface, ::PrescribedVelocityFields, grid) = tuple()

@inline BoundaryConditions.fill_halo_regions!(::PrescribedVelocityFields, args...; kwargs...) = nothing
@inline BoundaryConditions.fill_halo_regions!(::FunctionField, args...; kwargs...) = nothing
@inline BoundaryConditions.fill_halo_regions!(::TimeSeriesInterpolation, args...; kwargs...) = nothing

@inline datatuple(obj::PrescribedVelocityFields) = (; u = datatuple(obj.u), v = datatuple(obj.v), w = datatuple(obj.w))
@inline velocities(obj::PrescribedVelocityFields) = (u = obj.u, v = obj.v, w = obj.w)

# Extend sum_of_velocities for `PrescribedVelocityFields`
@inline sum_of_velocities(U1::PrescribedVelocityFields, U2) = sum_of_velocities(velocities(U1), U2)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields) = sum_of_velocities(U1, velocities(U2))

@inline sum_of_velocities(U1::PrescribedVelocityFields, U2, U3) = sum_of_velocities(velocities(U1), U2, U3)
@inline sum_of_velocities(U1, U2::PrescribedVelocityFields, U3) = sum_of_velocities(U1, velocities(U2), U3)
@inline sum_of_velocities(U1, U2, U3::PrescribedVelocityFields) = sum_of_velocities(U1, U2, velocities(U3))

ab2_step_velocities!(::PrescribedVelocityFields, args...) = nothing
rk_substep_velocities!(::PrescribedVelocityFields, args...) = nothing
step_free_surface!(::Nothing, model, timestepper, Δt) = nothing
compute_w_from_continuity!(::PrescribedVelocityFields, args...; kwargs...) = nothing
mask_immersed_velocities!(::PrescribedVelocityFields) = nothing

# No need for extra velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, free_surface) = velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, ::ExplicitFreeSurface) = velocities
transport_velocity_fields(velocities::PrescribedVelocityFields, ::Nothing) = velocities

validate_velocity_boundary_conditions(grid, ::PrescribedVelocityFields) = nothing
extract_boundary_conditions(::PrescribedVelocityFields) = NamedTuple()

free_surface_displacement_field(::PrescribedVelocityFields, ::Nothing, grid) = nothing
HorizontalVelocityFields(::PrescribedVelocityFields, grid) = nothing, nothing

materialize_free_surface(::Nothing,                      ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ExplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::ImplicitFreeSurface{Nothing}, ::PrescribedVelocityFields, grid) = nothing
materialize_free_surface(::SplitExplicitFreeSurface,     ::PrescribedVelocityFields, grid) = nothing

hydrostatic_prognostic_fields(::PrescribedVelocityFields, ::Nothing, tracers) = prognostic_tracers(tracers)
compute_hydrostatic_momentum_tendencies!(model, ::PrescribedVelocityFields, kernel_parameters; kwargs...) = nothing

compute_flux_bcs!(::Nothing, c, arch, clock, model_fields) = nothing

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
const OnlyParticleTrackingModel = HydrostaticFreeSurfaceModel{TS, E, A, S, G, T, V, B, R, F, P, U, W, C} where
                 {TS, E, A, S, G, T, V, B, R, F, P<:AbstractLagrangianParticles, U<:PrescribedVelocityFields, W<:PrescribedVelocityFields, C<:NamedTuple{(), Tuple{}}}

function time_step!(model::OnlyParticleTrackingModel, Δt; callbacks = [], kwargs...)
    tick!(model.clock, Δt)
    step_lagrangian_particles!(model, Δt)
    update_state!(model, callbacks)
end

update_state!(model::OnlyParticleTrackingModel, callbacks) =
    [callback(model) for callback in callbacks if callback.callsite isa UpdateStateCallsite]

#####
##### Checkpointing
#####

prognostic_state(::PrescribedVelocityFields) = nothing
restore_prognostic_state!(::PrescribedVelocityFields, ::Nothing) = nothing

#####
##### PrescribedTracer
#####

struct PrescribedTracer{F, P}
    field :: F
    parameters :: P
end

"""
    PrescribedTracer(field; parameters=nothing)

Wrap a `FieldTimeSeries` or `Function` to indicate that a tracer should be prescribed
(not time-stepped) in the model. Prescribed tracers are available to turbulence closures
for buoyancy computation but are not advanced by the time-stepper.

Supported input types:
- `FieldTimeSeries`: interpolated to the current model time via `TimeSeriesInterpolation`
- `Function(x, y, z, t)` or `Function(x, y, z, t, parameters)`: wrapped in a `FunctionField`

Examples
========

Prescribe buoyancy with a `Function`:

```jldoctest prescribed
julia> using Oceananigans

julia> using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedTracer

julia> b(x, y, z, t) = 1e-7 * x + 1e-5 * z;

julia> PrescribedTracer(b)
PrescribedTracer wrapping b (generic function with 1 method)
```

Prescribe temperature and salinity with `FieldTimeSeries`:

```jldoctest prescribed
julia> grid = RectilinearGrid(size=(4, 4, 4), extent=(1e5, 1e5, 1e3));

julia> T = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 86400.0]);

julia> S = FieldTimeSeries{Center, Center, Center}(grid, [0.0, 86400.0]);

julia> PrescribedTracer(T)
PrescribedTracer wrapping 4×4×4×2 FieldTimeSeries{InMemory} located at (Center, Center, Center) on CPU
```
"""
PrescribedTracer(field; parameters=nothing) = PrescribedTracer(field, parameters)

Base.summary(pt::PrescribedTracer) = "PrescribedTracer wrapping $(summary(pt.field))"
Base.show(io::IO, pt::PrescribedTracer) = print(io, summary(pt))

#####
##### Materialization: convert PrescribedTracer to a concrete field type
#####

materialize_prescribed_tracer(pt::PrescribedTracer{<:FieldTimeSeries}, grid; clock) =
    TimeSeriesInterpolation(pt.field, grid; clock)

materialize_prescribed_tracer(pt::PrescribedTracer{<:Function}, grid; clock) =
    FunctionField{Center, Center, Center}(pt.field, grid; clock, parameters=pt.parameters)

# Fallback: if already a Field or similar, just return it
materialize_prescribed_tracer(pt::PrescribedTracer, grid; clock) = pt.field

#####
##### Type predicates for prescribed tracers
#####

"""
    is_prescribed_tracer(field)

Return `true` if `field` is a prescribed (non-prognostic) tracer in `model.tracers`.
Prescribed tracers are `TimeSeriesInterpolation` or `FunctionField` objects.
"""
is_prescribed_tracer(::TimeSeriesInterpolation) = true
is_prescribed_tracer(::FunctionField) = true
is_prescribed_tracer(::Any) = false

#####
##### Helpers for separating prescribed and prognostic tracers
#####

"""
    prognostic_tracer_names(tracers::NamedTuple)

Return a tuple of tracer names that are prognostic (not prescribed).
"""
function prognostic_tracer_names(tracers::NamedTuple)
    names = propertynames(tracers)
    return Tuple(n for n in names if !is_prescribed_tracer(tracers[n]))
end

"""
    prognostic_tracers(tracers::NamedTuple)

Return a NamedTuple containing only prognostic (non-prescribed) tracer fields.
"""
function prognostic_tracers(tracers::NamedTuple)
    prog_names = prognostic_tracer_names(tracers)
    isempty(prog_names) && return NamedTuple()
    return NamedTuple{prog_names}(Tuple(tracers[n] for n in prog_names))
end

#####
##### Materialize tracer fields: handle mixed prescribed and prognostic tracers
#####

"""
    materialize_tracer_fields(tracers, grid, clock, boundary_conditions)

Create tracer fields from a tracer specification that may contain `PrescribedTracer` entries.
Regular tracers become `CenterField`s via `TracerFields`; prescribed tracers are materialized
into `TimeSeriesInterpolation` or `FunctionField` objects.
"""
function materialize_tracer_fields(tracers::NamedTuple, grid, clock, boundary_conditions)
    all_names = propertynames(tracers)

    # Check if any entries are PrescribedTracer
    has_prescribed = any(tracers[n] isa PrescribedTracer for n in all_names)
    has_prescribed || return TracerFields(tracers, grid, boundary_conditions)

    # Separate prescribed and prognostic tracer entries
    fields = map(all_names) do name
        entry = tracers[name]
        if entry isa PrescribedTracer
            materialize_prescribed_tracer(entry, grid; clock)
        else
            # entry is either a Field (user-provided) or a CenterField from closure tracers
            entry isa Field ? entry : CenterField(grid, boundary_conditions=boundary_conditions[name])
        end
    end

    return NamedTuple{all_names}(Tuple(fields))
end

# Fallback: when tracers is a Tuple of Symbols (no prescribed tracers possible)
materialize_tracer_fields(tracers::Tuple, grid, clock, boundary_conditions) =
    TracerFields(tracers, grid, boundary_conditions)

#####
##### Boundary condition handling for prescribed tracers
#####

extract_boundary_conditions(::PrescribedTracer) = FieldBoundaryConditions()
extract_boundary_conditions(::TimeSeriesInterpolation) = FieldBoundaryConditions()
extract_boundary_conditions(::FunctionField) = FieldBoundaryConditions()

#####
##### Dispatch-based no-ops for prescribed tracers in tendency and stepping loops
#####

compute_flux_bcs!(Gⁿ, ::TimeSeriesInterpolation, arch, args) = nothing
compute_flux_bcs!(Gⁿ, ::FunctionField, arch, args) = nothing

_compute_tracer_tendency!(::TimeSeriesInterpolation, args...; kwargs...) = nothing
_compute_tracer_tendency!(::FunctionField, args...; kwargs...) = nothing

_rk_substep_tracer!(::TimeSeriesInterpolation, args...) = nothing
_rk_substep_tracer!(::FunctionField, args...) = nothing

_ab2_step_tracer!(::TimeSeriesInterpolation, args...) = nothing
_ab2_step_tracer!(::FunctionField, args...) = nothing
